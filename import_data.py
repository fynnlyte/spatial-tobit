import os
import sys
import geopandas as gpd
import pandas as pd
import numpy as np
import itertools
import arviz as az
import matplotlib.pyplot as plt
from pathlib import Path
from pystan import StanModel, check_hmc_diagnostics
from joblib import dump, load


# os.chdir(sys.path[0]) # seems to be needed on windows?
for folder in ['cache', 'logs']:
    os.makedirs(folder, exist_ok=True)
        
plt.rcParams["figure.figsize"] = (16,12)
plt.rcParams["font.size"] = 20

########################################################################################
# Data Loading and preprocessing
###

# Use crash data to later infere crash counts
mappedCrashData = gpd.read_file(Path('data/NewYorkCrashes_Mapped.shp'))
# Delete values with join_dist = -1 -> they are not mapped
mappedCrashDataClean = mappedCrashData.loc[mappedCrashData['join_dist'] != -1]

# Use to create adjaceny matrix
adjacencyData = gpd.read_file(Path('data/Adjacency.shp'))

# Map the crash counts to the segment data
segmentData = gpd.read_file(Path('data/Segment_NewYork2017_Mapped.shp'))
#segmentData['Segment_ID'] = segmentData['Segment_ID'].astype(int)
# Remove unnecessary columns
segmentData = segmentData.drop(columns=['Join_Count', 'TARGET_FID', 'FID_Segmen',
                                        'Id', 'FID_NewYor', 'Year_Recor', 'State_Code', 'Route_ID', 'Begin_Poin', 'End_Point',
                                        'Route_Numb', 'Route_Name', 'Urban_Code', 'County_Cod', 'STRAHNET', 'Truck_NN',
                                        'AADT_Singl', 'AADT_Combi', 'Toll_Charg', 'Toll_ID', 'Toll_Type', 'ESRI_OID', 'Shape_Leng',
                                        'geometry', 'PSR', 'Route_Qual', 'Route_Sign', 'IRI', 'Access_Con'])

segmentData = segmentData.rename(columns={'F_System': 'Fun_Class',
                                          'Facility_T': 'Facility_Type',
                                          'Speed_Limi': 'Speed_Limit',
                                          # 'Access_Con': 'Access_Control',
                                          # 'Route_Qual': 'Route_Qualifier',
                                          'Through_La': 'Through_Lanes'
                                          })

segmentData = pd.DataFrame(segmentData)

# Set speed limits to 25 due to Manhattans wide 25 mph regulatory
segmentData["Speed_Limit"] = segmentData["Speed_Limit"].replace(0, 25)


# Remap faciltiy type #Remap facility type 1= One-Way Roadway.2=Two-Way Roadway. 3 = Other (4=Ramp,5=Non-Mainline,6=Non-Inventory Direction)

# Remap functional class (1 interstate, 2 Principal Arterial (2,3), 3 Minor Arterial(4), 4 Others (5,6,7))
segmentData['Fun_Class'] = segmentData['Fun_Class'].astype(str)
segmentData['Fun_Class'] = segmentData['Fun_Class'].replace('1', 'Interstate')
segmentData['Fun_Class'] = segmentData['Fun_Class'].replace(['2', '3'], 'Principal_Arterial')
segmentData['Fun_Class'] = segmentData['Fun_Class'].replace(['4'], 'Minor_Arterial')
segmentData['Fun_Class'] = segmentData['Fun_Class'].replace(['5', '6', '7'], 'Others')

# Facility type (one-way road, two-way road etc.)
segmentData['Facility_Type'] = segmentData['Facility_Type'].astype(str)
segmentData['Facility_Type'] = segmentData['Facility_Type'].replace('1', 'One-Way')
segmentData['Facility_Type'] = segmentData['Facility_Type'].replace('2', 'Two-Way')
segmentData['NHS'] = segmentData['NHS']


# not used for computation:
segmentData['Segment_ID'] = segmentData['Segment_ID'].astype(int)

# Ownership (Values: 32 Local Toll Authority ,1 State Hwy Agency,4 City or Municipal Hwy Agency, 12 Local Park, Forest, or Reservation Agency.)
segmentData['Ownership'] = segmentData['Ownership'].astype(str)
for i, name in (('1', 'State_Hwy'), ('32', 'Local_Toll_Authority'), ('4', 'Local_Hwy'), ('12', 'Other_Auth')):
    segmentData['Ownership'] = segmentData['Ownership'].replace(i, name)

# Additional variables
# AADT
# (Length_m)

# Map crash amounts to segements
# Infere crashes per segment

# Group crashes by Segment
groupedCrashes = mappedCrashDataClean.groupby('Segment_ID').count()

# Create array that holds all crashes
crashArray = np.zeros((len(segmentData), 1))

# Create list with crashes per segment
for index, row in groupedCrashes.iterrows():
    crashArray[int(index)] = row["Join_Count"]

# Attach crash list to segments data
segmentData['Crashes'] = crashArray.astype(int)

# Calculate crash rate
crashRate = segmentData['Crashes'] / \
    (segmentData['AADT']*(segmentData['Length_m']/1000)*365/1000000)
segmentData['CrashRate'] = crashRate

# Create adjacency matrix
n_segments = len(segmentData)
adjacencyMatrix = np.zeros((n_segments, n_segments), dtype=int)
intersection_list = adjacencyData["Intsec_ID"].unique().tolist()

# Loop over all intersections
for i in range(len(intersection_list)):
    # Get all segments that are connected via this intersection
    intersec = intersection_list[i]

    connectedSegments = adjacencyData.loc[adjacencyData['Intsec_ID']
                                          == intersec]["Segment_ID"].tolist()
    # Combine all segments to tuples
    combineSegments = list(itertools.permutations(connectedSegments, 2))
    # Loop over all tuples and set to 1 (connection between these to segments)
    for j in range(len(combineSegments)):
        tup = combineSegments[j]
        # print(tup)
        adjacencyMatrix[int(tup[0]), int(tup[1])] = 1

########################################################################################

########################################################################################
# Variable selection and verification
# 
        
for kind in ['Interstate', 'Principal_Arterial', 'Minor_Arterial', 'Others']:
    segmentData[kind] = (segmentData['Fun_Class'] == kind).astype(int)
for kind in ['One-Way', 'Two-Way']:
    segmentData[kind] = (segmentData['Facility_Type'] == kind).astype(int)
for kind in ['State_Hwy','Local_Toll_Authority','Local_Hwy','Other_Auth']:
    segmentData[kind] = (segmentData['Ownership'] == kind).astype(int)
segmentDF = pd.DataFrame(segmentData)
desc = segmentDF.describe()
for kind in desc.columns:
    desc[kind][0] = (segmentData[kind] > 0.0000000001).sum()
desc.to_excel(Path('logs/description.xlsx'))

segmentDF.drop(['Other_Auth', 'Fun_Class', 'Facility_Type', 'Ownership'], axis=1, inplace=True)


# first very simply approach: run tobit model on a fraction of the dataset with
# Three non-categorical vals as predictors (ok, Through_La technically is...)
# Not considering 
predictors = ['Length_m', 'AADT', 'Through_Lanes', 'NHS', 'Interstate', 'Principal_Arterial', 
              'Minor_Arterial', 'Two-Way']

# calculate pearson correlation:
corr_mat = np.ones((len(segmentDF.columns), len(segmentDF.columns)))
for i, row in enumerate(segmentDF.columns):
    for j, col in enumerate(segmentDF.columns):
        if i !=j:
            corr_mat[i,j] = segmentDF[row].corr(segmentDF[col])
corr_df = pd.DataFrame(corr_mat, index=segmentDF.columns, columns=segmentDF.columns)
corr_df.to_excel(Path('logs/correlation.xlsx'))
# Basically, NHS means interstate. Also dropping Through_Lanes like the researchers
# due to high correlation with aadt.
#predictors.remove('NHS')
predictors.remove('Interstate')
predictors.remove('Principal_Arterial')
predictors.remove('Minor_Arterial')
predictors.remove('Through_Lanes')

# some verifications before using the data:
# - is the adjacency matrix symmetric? Necessary for generating a sparse
#   representation of it.
# - no ones on the diagonals?
for i in range(n_segments):
    for j in range(n_segments):
        if i != j and adjacencyMatrix[i, j] != adjacencyMatrix[j, i]:
            print('Error: from {} to {}: {}, but from {} to {}: {}'
                  .format(i, j, adjacencyMatrix[i, j], j, i, adjacencyMatrix[j, i]))
        elif i == j and adjacencyMatrix[i, j] != 0:
            print('Error: encountered value {} in row/col {}'
                  .format(adjacencyMatrix[i, i], i))
# - is the adjacency graph connected or are there some weird segments?
empty_row_count = (adjacencyMatrix.sum(axis=0) == 0).sum()
if empty_row_count > 0:
    print('adj matrix has %s rows/cols without any edge.' % empty_row_count)


########################################################################################
#  Calculate Moran's I and Z score
###
    
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

def z_score(n, adjacencyMatrix, crash_rates, morans_i):
    mean =  -1/(n - 1)

    # helper functions for calculating the variance
    # calc for variance from https://en.wikipedia.org/wiki/Moran%27s_I
    s1_helper = 0
    s2_helper = 0
    s2 = 0
    sum_adjacency_weights = 0
    global_avg_crash_rate = crash_rates.mean()
    s3_helper1 = 0
    s3_helper2 = 0

    for i in range(n):
        for j in range(n):
            s1_helper += np.square(2 * adjacencyMatrix[i][j])
            s2_helper += 2 * adjacencyMatrix[i][j]
            sum_adjacency_weights += adjacencyMatrix[i][j]
        s2 += np.square(s2_helper)
        s3_helper1 += np.power(crash_rates[i] - global_avg_crash_rate, 4)
        s3_helper2 += np.power(crash_rates[i] - global_avg_crash_rate, 2)
    s1 = (1 / 2) * s1_helper
    s3 = s3_helper1 / \
         (np.square(s3_helper2 / n) * n)
    s4 = (np.square(n) - 3 * n + 3) * s1 - \
         n * s2 + 3 * np.square(sum_adjacency_weights)
    s5 = (np.square(n) - n) * s1 - \
         2 * n * s2 + 6 * np.square(sum_adjacency_weights)
    # print('s1', s1, 's2', s2, 's3', s3, 's4', s4, 's5', s5)

    variance = ((n * s4) - (s3 * s5)) / \
               ((n - 1) * (n - 2) * (n - 3) * np.square(sum_adjacency_weights)) \
               - np.square(mean)

    print(np.sqrt(variance))
    return (morans_i - mean) / np.sqrt(variance)

def show_morans_and_z():
    moransi = morans_i(len(segmentData), adjacencyMatrix, segmentData['CrashRate'])
    zscore = z_score(len(segmentData), adjacencyMatrix, segmentData['CrashRate'], moransi)
    
    print('morans i:', moransi, 'z-score:', zscore)

########################################################################################


########################################################################################
# Running the models
###

def get_tobit_dict(df: pd.DataFrame):
    threshold = 0.0000000001
    is_cens = df['CrashRate'] < threshold
    not_cens = df['CrashRate'] >= threshold
    ii_obs = df[not_cens].index + 1
    ii_cens = df[is_cens].index + 1

    tobit_dict = {'n_obs': not_cens.sum(), 'n_cens': is_cens.sum(), 'p': len(predictors),
                  'ii_obs': ii_obs, 'ii_cens': ii_cens,
                  'y_obs': df[not_cens]['CrashRate'], 'U': threshold,
                  'X': df[predictors]}
    return tobit_dict

def add_car_info_to_dict(tobit_dict, filtered_matrix):
    car_dict = tobit_dict.copy()
    car_dict['W'] = filtered_matrix
    car_dict['W_n'] = filtered_matrix.sum()//2
    car_dict['n'] = tobit_dict['X'].shape[0]
    return car_dict

def get_full_dict(segmentDF, adjMatrix):
    ret = get_tobit_dict(segmentDF)
    ret['y'] = segmentDF['CrashRate']
    ret = add_car_info_to_dict(ret, adjMatrix)
    return ret

def run_or_load_model(m_type, m_dict, iters, warmup, c_params):
    if m_type not in ['car', 'tobit']:
        raise Exception('Invalid model type!')
    name = 'crash_{}_{}-{}_delta_{}_max_{}'.format(m_type,iters, warmup, 
                                                           c_params['adapt_delta'],
                                                           c_params['max_treedepth'])
    try:
        model = load(Path('cache/' + name + '_model.joblib'))
    except:
        model = StanModel(file=Path('models/crash_{}.stan'.format(m_type)).open(),
                          extra_compile_args=["-w"], model_name=name.split('-')[0])
        dump(model, Path('cache/' + name + '_model.joblib'))
    try:
        fit = load(Path('cache/' + name + '_fit.joblib'))
    except:
        fit = model.sampling(data=m_dict, iter=iters, warmup=warmup,
                             control=c_params, check_hmc_diagnostics=True)
        info = fit.stansummary()
        with open(Path('logs/' + name + '.log'), 'w') as c_log:
            c_log.write(info)
        dump(fit, Path('cache/' + name + '_fit.joblib'))
    return model, fit

def compare_runtimes(segmentDF, adjMatrix):
    """
    usage: adjust main, run the script as `python3 import_data.py > log.txt` and then
    call `grep -E "running model|Gradient|(Total)" log.txt` to read the relevant lines
    most likely won't work on Windows as there seems to be less output.
    """
    iters = 5000
    warmup = 1000
    # 'models/comparison/CAR_simple.stan' -> this model runs forever even with iters = 100
    models = ['models/comparison/tobit_for_loop.stan', 'models/comparison/tobit_vectorised.stan',
              
              'models/crash_tobit.stan', 'models/crash_car.stan', 'models/comparison/CAR_QR.stan', 
              'models/comparison/CAR_simple.stan' ]
    data = get_full_dict(segmentDF, adjMatrix) # n_obs, n_cens, p, ii_obs, ii_cens, y_obs, U, X
    model_and_fit = []
    
    for i, m_file in enumerate(models):
        print(f'running model: {m_file}')
        m_name = m_file.split('/')[-1].split('.')[0]
        try:
            model = load(Path(f'cache/profiling_{m_name}.joblib'))
        except:
            model = StanModel(model_name=m_name, file=Path(m_file).open(),
                              extra_compile_args=['-w'])
            dump(model, Path(f'cache/{m_name}.joblib'))
        fit = model.sampling(data=data, iter=iters, warmup=warmup, 
                             control= {'adapt_delta': 0.95, 'max_treedepth': 15})
        model_and_fit.append((model, fit))
    return model_and_fit


def run_and_plot_models(segmentDF, adjacencyMatrix, iters, warmup):

    tobit_dict = get_tobit_dict(segmentDF)
    
    # TOBIT MODEL:
    t_c_params = {'adapt_delta': 0.95, 'max_treedepth': 15}
    tobit_model, tobit_fit = run_or_load_model('tobit', tobit_dict, iters, warmup, t_c_params)
    check_hmc_diagnostics(tobit_fit)
    
    plt.hist(tobit_fit['sigma'], bins=int(iters*4/100))
    plt.title('tobit')
    tob_vars = ['sigma', 'beta_zero', 'theta' ]
    az.plot_trace(tobit_fit, tob_vars)
    
    
    # SPATIAL TOBIT MODEL:
    c_c_params = {'adapt_delta': 0.95, 'max_treedepth': 15}
    car_dict = add_car_info_to_dict(tobit_dict, adjacencyMatrix)
    car_model, car_fit = run_or_load_model('car', car_dict, iters, warmup, c_c_params)
    check_hmc_diagnostics(car_fit)
    
    plt.hist(car_fit['sigma'], bins=int(iters*4/100))
    plt.title('car')
    car_vars = ['sigma', 'beta_zero', 'theta', 'alpha', 'tau']
    az.plot_trace(car_fit, compact=False, var_names=car_vars)
    
    az.plot_pair(car_fit, ['tau', 'alpha', 'sigma'], divergences=True)
    plt.scatter(car_fit['lp__'], car_fit['sigma'])
    plt.hist(car_fit['phi'].mean(axis=0),bins=50)


show_morans_and_z()
### comparing several approaches 
#compare_runtimes(segmentDF, adjacencyMatrix)

### running the models (change to lower values for faster computation)
run_and_plot_models(segmentDF, adjacencyMatrix, 30000, 5000)