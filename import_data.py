import geopandas as gpd
import pandas as pd
import numpy as np
import itertools
import networkx as nx
from pathlib import Path
from pystan import StanModel
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
<<<<<<< HEAD
empty_row_count = adjacencyMatrix.sum(axis=0) == 0).sum()
if empty_row_count > 0:
    print('adj matrix has %s rows/cols without any edge.' % empty_row_count)

#adj_graph = nx.Graph(adjacencyMatrix)
#if not nx.is_connected(adj_graph):
#    print('Need to remove nodes:')
#    conn_comp = [c for c in sorted(nx.connected_components(adj_graph), reverse=True, key=len)]
#    isolated_nodes = [n for c in conn_comp[1:len(conn_comp)] for n in c ]
#    print(isolated_nodes)

# segment ids and also rows/ cols in adjacency-matrix are 0-indexed!!!
# WTF - there is one entry with one!! That may not happen.
(adjacencyMatrix.sum(axis=1) == 1).sum()
(adjacencyMatrix.sum(axis=0) == 1).sum()
filtered_df = segmentDF[adjacencyMatrix.sum(axis=0) > 0].reset_index()
n_filt = filtered_df.shape[0]
filtered_matrix = np.zeros((n_filt, n_filt), dtype=int)
segmentID_to_index = dict()
for i, s_id in filtered_df[['Segment_ID']].itertuples(index=True):
    segmentID_to_index[int(s_id)] = i

for i in range(n_filt):
    isec_id = int(filtered_df.iloc[i]['Segment_ID'])
    intersec = intersection_list[isec_id]

    connectedSegments = adjacencyData.loc[adjacencyData['Intsec_ID'] == intersec]["Segment_ID"].tolist()   
    #Combine all segments to tuples
    combineSegments = list(itertools.permutations(connectedSegments, 2))
    #Loop over all tuples and set to 1 (connection between these to segments)   
    for j in range(len(combineSegments)):
        tup = combineSegments[j]
        #print(tup)
        s_id = int(tup[0])
        d_id = int(tup[1])
        if s_id in segmentID_to_index and d_id in segmentID_to_index:
            s = segmentID_to_index[s_id]
            d = segmentID_to_index[d_id]
            if s in isolated_nodes or d in isolated_nodes:
                print('Skipping isolated: %s - %d' % (s,d))
            else:
                filtered_matrix[s,d] = 1
        else:
            print('skipping edge: %s - %d' % (s_id, d_id))
           
print('how many zero rows in my matrix? %s. WTF??' % (filtered_matrix.sum(axis=0) == 0).sum())

#filtered_graph = nx.Graph(filtered_matrix)
#if not nx.is_connected(filtered_graph):
#    print('adj graph is not connected! need to remove nodes:')
#    conn_comp = [c for c in sorted(nx.connected_components(adj_graph), reverse=True, key=len)]
#    isolated_nodes = [n for c in conn_comp[1:len(conn_comp)] for n in c ]
#    print(isolated_nodes)


###
# model begins here
###
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
tobit_fit = tobit_model.sampling(data=tobit_dict, iter=4000, warmup=500)
tobit_info = tobit_fit.stansummary()
print(tobit_fit.stansummary())
dump(tobit_fit, 'data/crash_tobit.joblib')
with open('data/crash_tobit.log', 'w') as t_log:
    t_log.write(tobit_info)


car_dict = tobit_dict.copy()
car_dict['W'] = filtered_matrix
car_dict['W_n'] = filtered_matrix.sum()//2
car_dict['n'] = data_centered.shape[0]
car_fit = car_model.sampling(data=car_dict, iter=4000, warmup=500)
car_info = car_fit.stansummary()
dump(car_fit, 'data/car_tobit.joblib')
with open('data/crash_tobit.log', 'w') as c_log:
    c_log.write(car_info)
=======
adj_graph = nx.Graph(adjacencyMatrix)
if not nx.is_connected(adj_graph):
    print('adj graph is not connected! need to remove nodes:')
    conn_comp = [c for c in sorted(nx.connected_components(adj_graph), reverse=True, key=len)]
    isolated_nodes = [n for c in conn_comp[1:len(conn_comp)] for n in c ]
    print(isolated_nodes)
    
    
#Check of the isolated nodes
#Isolated nodes are road segment thats are not correctly connected in the traffic shape files. We can remove them since they are not too much
    
>>>>>>> f0b158d6ad917ea1db229c1470ed482aa01776f6
