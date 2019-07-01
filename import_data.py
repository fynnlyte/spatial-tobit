import geopandas as gpd
import pandas as pd
import numpy as np
import itertools
import networkx as nx
from pathlib import Path
from pystan import StanModel, check_hmc_diagnostics
from sklearn.preprocessing import MaxAbsScaler
from joblib import dump


#Use crash data to later infere crash counts
mappedCrashData = gpd.read_file(Path('data/NewYorkCrashes_Mapped.shp'))
#Delete values with join_dist = -1 -> they are not mapped
mappedCrashDataClean = mappedCrashData.loc[mappedCrashData['join_dist'] != -1]

# Use to create adjaceny matrix
adjacencyData = gpd.read_file(Path('data/Adjacency.shp'))

#Map the crash counts to the segment data
segmentData =  gpd.read_file(Path('data/Segment_NewYork2017_Mapped.shp'))
#segmentData['Segment_ID'] = segmentData['Segment_ID'].astype(int)
#Remove unnecessary columns
segmentData = segmentData.drop(columns=['Join_Count', 'TARGET_FID', 'FID_Segmen', 
                 'Id', 'FID_NewYor', 'Year_Recor', 'State_Code', 'Route_ID', 'Begin_Poin', 'End_Point', 
                 'Route_Numb', 'Route_Name', 'Urban_Code', 'County_Cod',  'NHS', 'STRAHNET', 'Truck_NN', 
                 'AADT_Singl', 'AADT_Combi', 'Toll_Charg', 'Toll_ID', 'Toll_Type', 'ESRI_OID', 'Shape_Leng',
                 'geometry',])

segmentData = segmentData.rename(columns={'F_System': 'Fun_Class',
                           'Facility_T': 'Facility_Type',
                           'F_System': 'Fun_Class',
                           'Speed_Limi': 'Speed_Limit',
                           'Access_Con': 'Access_Control'
                           })

#Set speed limits to 25 due to Manhattans wide 25 mph regulatory
segmentData["Speed_Limit"] = segmentData["Speed_Limit"].replace(0,25)


###############################################################################################################################
#Map crash amounts to segements
#Infere crashes per segment

#Group crashes by Segment
groupedCrashes = mappedCrashDataClean.groupby('Segment_ID').count()

#Create array that holds all crashes
crashArray= np.zeros((len(segmentData), 1))

#Create list with crashes per segment
for index, row in groupedCrashes.iterrows():
    crashArray[int(index)] = row["Join_Count"]

#Attach crash list to segments data
segmentData['Crashes'] = crashArray.astype(int)

#Calculate crash rate
crashRate = segmentData['Crashes']/(segmentData['AADT']*(segmentData['Length_m']*1000)*365/1000000)
segmentData['CrashRate'] = crashRate


###############################################################################################################################
#Create adjacency matrix
n_segments = len(segmentData)
adjacencyMatrix = np.zeros((n_segments, n_segments), dtype=int)
intersection_list = adjacencyData["Intsec_ID"].unique().tolist()

#Loop over all intersections
for i in range(len(intersection_list)):
    #Get all segments that are connected via this intersection
    intersec = intersection_list[i]
    
    connectedSegments = adjacencyData.loc[adjacencyData['Intsec_ID'] == intersec]["Segment_ID"].tolist()   
    #Combine all segments to tuples
    combineSegments = list(itertools.permutations(connectedSegments, 2))
    #Loop over all tuples and set to 1 (connection between these to segments)   
    for j in range(len(combineSegments)):
        tup = combineSegments[j]
        #print(tup)
        adjacencyMatrix[int(tup[0]),int(tup[1])] = 1
        
#######################################################################################################
#Use segmentData and adjacencyMatrix
# Matrix needs to be cleaned up before model can be used :(
# - May not contain any vertices without neighbors!

segmentDF = pd.DataFrame(segmentData)
desc = segmentDF.describe()

# first very simply approach: run tobit model on a fraction of the dataset with 
# Three non-categorical vals as predictors (ok, Through_La technically is...)
predictors = ['ones', 'Length_m', 'AADT', 'Through_La']

# some verifications before using the data: 
# - is the adjacency matrix symmetric? Necessary for generating a sparse
#   representation of it.
# - no ones on the diagonals?
for i in range(n_segments):
    for j in range(n_segments):
        if i != j and adjacencyMatrix[i,j] != adjacencyMatrix[j,i]:
            print('Error: from {} to {}: {}, but from {} to {}: {}'
                  .format(i,j, adjacencyMatrix[i,j], j, i, adjacencyMatrix[j,i]))
        elif i == j and adjacencyMatrix[i,j] != 0:
            print('Error: encountered value {} in row/col {}'
                  .format(adjacencyMatrix[i,i], i))
# - is the adjacency graph connected or are there some weird segments?
empty_row_count = (adjacencyMatrix.sum(axis=0) == 0).sum()
if empty_row_count > 0:
    print('adj matrix has %s rows/cols without any edge.' % empty_row_count)



###
# model begins here
###
filtered_df = segmentDF
filtered_matrix = adjacencyMatrix
filtered_df['ones'] = np.ones(filtered_df.shape[0])
tobit_model = StanModel(file=Path('models/crash_tobit.stan').open(),
                        extra_compile_args=["-w"])
car_model = StanModel(file=Path('models/crash_CAR.stan').open(),
                      extra_compile_args=["-w"])

trans = MaxAbsScaler().fit_transform(filtered_df[predictors + ['CrashRate']])
data_centered = pd.DataFrame(trans, columns=predictors + ['CrashRate'])
threshold = 0.0000000001
is_cens = data_centered['CrashRate'] < threshold
not_cens = data_centered['CrashRate'] >= threshold
ii_obs = filtered_df[not_cens].index + 1
ii_cens = filtered_df[is_cens].index + 1
tobit_dict = {'n_obs': not_cens.sum(), 'n_cens': is_cens.sum(), 'p': len(predictors),
              'ii_obs': ii_obs, 'ii_cens': ii_cens,
              'y_obs': filtered_df[not_cens]['CrashRate'], 'U': threshold,
              'X': filtered_df[predictors]}
tobit_params = {'adapt_delta': 0.95, 'max_treedepth': 15}
tobit_fit = tobit_model.sampling(data=tobit_dict, iter=20000, warmup=4000,
                                 control=tobit_params)
tobit_info = tobit_fit.stansummary()
with open(Path('data/crash_tobit.log'), 'w') as t_log:
    t_log.write(tobit_info)
dump(tobit_fit, Path('data/crash_tobit.joblib'))

# 95% divergence
# 5% max depth of 10
# low E-BFMI (0.039 - 0.07)

# now with adapt_delta: 0.9 and depth: 12
# 89% divergence
# 10% max depth
# low E-BFMI (0.005 - 0.03)

# seems like reparametrisation is really necessary...

# as comparison: without the vectorisation
#t_old_dict = {'n': data_centered.shape[0], 'p': len(predictors), 'X': filtered_df[predictors],
#              'y': filtered_df['CrashRate'], 'U': threshold, 'n_cens': is_cens.sum()}
#t_old_model = StanModel(file=Path('models/crash_tobit_old.stan').open(),
#                        extra_compile_args=["-w"])
#t_old_fit = t_old_model.sampling(data=t_old_dict, iter=20000, warmup=4000)
#dump(t_old_fit, Path('data/crash_old.joblib'))
#old_info = t_old_fit.stansummary()
#with open(Path('data/crash_old.log'), 'w') as old_log:
#    old_log.write(old_info)
# 96% div
# 3.8% max tree depth
# low E-BFMI (0.36 - 0.08)
# but nice. The parameters β are roughly same. Vectorisation seems OK.


# WARNING: this will take ages
c_params = {'adapt_delta': 0.95, 'max_treedepth': 15}
car_dict = tobit_dict.copy()
car_dict['W'] = filtered_matrix
car_dict['W_n'] = filtered_matrix.sum()//2
car_dict['n'] = data_centered.shape[0]
car_fit = car_model.sampling(data=car_dict, iter=20000, warmup=4000, control=c_params)
car_info = car_fit.stansummary()
with open('data/crash_car.log', 'w') as c_log:
    c_log.write(car_info)
dump(car_fit, 'data/car_tobit.joblib')


########################################################################################################################
# Calculate Moran's I

print('data:', segmentData)
print('adj:', adjacencyMatrix)
print(len(adjacencyMatrix), len(adjacencyMatrix[1]))
print(adjacencyMatrix[0][2])
print(adjacencyMatrix[19])
print(segmentData['CrashRate'])

print(len(segmentData))

def morans_i(number_road_segments, adjacencyMatrix, crash_rates):
    global_avg_crash_rate = crash_rates.mean()
    numerator = 0
    sum_adjacency_weights = 0
    denominator = 0
    for i in range(number_road_segments):
        for j in range(number_road_segments):
            numerator += adjacencyMatrix[i][j] *\
                         (crash_rates[i] - global_avg_crash_rate) *\
                         (crash_rates[j] - global_avg_crash_rate)
            sum_adjacency_weights += adjacencyMatrix[i][j]
        denominator += np.square(crash_rates[i] - global_avg_crash_rate)
    return (number_road_segments * numerator) / (sum_adjacency_weights * denominator)

print(morans_i(len(segmentData), adjacencyMatrix, segmentData['CrashRate']))
