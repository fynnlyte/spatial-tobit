#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 14 14:44:57 2019

@author: fynn
"""

import pandas as pd

aadt_ny = pd.read_csv('data/Annual_Average_Daily_Traffic__AADT___Beginning_1977.csv')
collisions = pd.read_csv('data/NYPD_Motor_Vehicle_Collisions.csv')

aadt_ny.columns
collisions.columns

coll_man = collisions[collisions['BOROUGH'] == 'MANHATTAN']

municips = set(aadt_ny['Municipality'].unique())

aadt_nyc = aadt_ny[aadt_ny['Municipality'] == 'NEW YORK CITY']

aadt_gis_codes = set(aadt_nyc['GIS Code'])

# Challenge is now:
# 1. select the segments of Manhattan that I want to investigate.
#    e.g. 34th - 66th.

aadt_nyc['Roadway Begin Description'].isnull().sum()
len(aadt_nyc['Roadway Begin Description'])
aadt_nyc['Roadway End Description'].isnull().sum()
len(aadt_nyc['Roadway End Description'])
# Note: for 124 of 18156 roads, the begin is not given, 
#       for 158 the end is not given. Missing or blind alley? 
# looking at Jewel Ave -> ends at Park East at one point, other blind alley.
test = aadt_nyc[aadt_nyc['Roadway Begin Description'].isnull()]

# I could restrict the crash sites to only the relevant ones by plucing four markers
# and requiring that their coordinates are within that square.

# but in order to connect with the collisions, would need to somehow map...

