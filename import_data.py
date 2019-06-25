import geopandas as gpd
import pandas as pd
import numpy as np
import itertools
import networkx as nx
from pathlib import Path


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
        
segmentDF = pd.DataFrame(segmentData)
desc = segmentDF.describe()

# first very simply approach: run tobit model on a fraction of the dataset with 
# Three non-categorical vals as predictors (ok, Through_La technically is...)
predictors = ['Length_m', 'AADT', 'Throug_La']

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
adj_graph = nx.Graph(adjacencyMatrix)
if not nx.is_connected(adj_graph):
    print('adj graph is not connected! need to remove nodes:')
    conn_comp = [c for c in sorted(nx.connected_components(adj_graph), reverse=True, key=len)]
    isolated_nodes = [n for c in conn_comp[1:len(conn_comp)] for n in c ]
    print(isolated_nodes)
    
    
#Check of the isolated nodes
#Isolated nodes are road segment thats are not correctly connected in the traffic shape files. We can remove them since they are not too much
    
