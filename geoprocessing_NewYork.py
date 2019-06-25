######NOTE#####
## Only working if ARCGIS Pro is installed

import arcpy

#Used shapefiles
workPath = os.getcwd()
trafficShapeFile = workPath + r'\data\NewYork2017.shp'
crashCSVFile = workPath + r'\data\NYPD_Motor_Vehicle_Collisions_Clean.csv'
crashShapeFile = workPath +r'\data\NewYorkCrashes.shp'

mappingTrafficShapeFile = workPath + r'\data\NewYork2017_mapping.shp'
segmentShapeFile = workPath + r'\data\Segment_NewYork2017_FullBlob.shp'
segmentShapeFile2 = workPath + r'\data\Segment_NewYork2017_Intersected.shp'
segmentShapeFile3 = workPath + r'\data\Segment_NewYork2017_Mapped.shp'
segmentShapeFile3Name = "Segment_NewYork2017_Mapped"




#Import crash data
XPoints = "LONGITUDE"
YPoints = "LATITUDE"

arcpy.MakeXYEventLayer_management(crashCSVFile, XPoints , YPoints , "New York CrashData","", "")
arcpy.CopyFeatures_management("New York CrashData", crashShapeFile)

#Clean traffic shape files
#Delete everything besides data from Manhattan
with arcpy.da.UpdateCursor(trafficShapeFile, "County_Cod") as cursor:
    for row in cursor:
        if row[0] != 61:
            cursor.deleteRow()


#Create mapping traffic shape file -> this file is used to later correctly map the AADTs
#Background: some segments are to long or to short, that leads to unwanted mini segments
#Split at intersections
arcpy.FeatureToLine_management(trafficShapeFile,
                               mappingTrafficShapeFile,
                               "0.001 Meters")


#Create segment shape file
arcpy.Dissolve_management(mappingTrafficShapeFile, segmentShapeFile)
	##Now we have one big connected polygon

#Intersect segment shape file
arcpy.FeatureToLine_management(segmentShapeFile,
                              segmentShapeFile2,
                               "0.001 Meters")

#Add fields segment shape file, length and id
arcpy.AddField_management(in_table= segmentShapeFile2, field_name = "Length_m", field_type="DOUBLE")
arcpy.AddField_management(in_table= segmentShapeFile2, field_name = "Segment_ID", field_type="DOUBLE")


#Calculate segment length of splitted traffic shape file
arcpy.CalculateGeometryAttributes_management (in_features = segmentShapeFile2, geometry_property = [["Length_m", "LENGTH_GEODESIC"]], length_unit = "METERS")
#Calculate segment ID
arcpy.CalculateField_management (in_table = segmentShapeFile2, field = "Segment_ID", expression="!FID!")



#Spatial join segment file with traffic mapping file
fieldmappings = arcpy.FieldMappings()
fieldmappings.addTable(segmentShapeFile2)
fieldmappings.addTable(mappingTrafficShapeFile)

arcpy.SpatialJoin_analysis (target_features = segmentShapeFile2, join_features = mappingTrafficShapeFile, 
out_feature_class = segmentShapeFile3, 
join_operation = "JOIN_ONE_TO_ONE", join_type = "KEEP_ALL", field_mapping = fieldmappings, 
match_option ="SHARE_A_LINE_SEGMENT_WITH")


#Spatial join crash 
#Use distance to determine if map is valid or not!
#Avoid double attaching of 
outName = workPath + r'\data\NewYorkCrashes_Mapped.shp'

fieldmappings = arcpy.FieldMappings()

# Add all fields from inputs.
fieldmappings.addTable(segmentShapeFile3)
fieldmappings.addTable(crashShapeFile)

# Fields that shall remain
keepers = ["Segment_ID","join_dist"] 

# Remove all output fields you don't want.
for field in fieldmappings.fields:
    if field.name not in keepers:
        fieldmappings.removeFieldMap(fieldmappings.findFieldMapIndex(field.name))



arcpy.SpatialJoin_analysis (target_features = crashShapeFile, join_features = segmentShapeFile3, 
out_feature_class = outName, 
join_operation = "JOIN_ONE_TO_ONE", join_type = "KEEP_ALL", field_mapping = fieldmappings, 
match_option ="CLOSEST", search_radius="0.0002",distance_field_name ="join_dist")




##Create points at intersects -> mistakes can occur with bridges etc.
intersectionShapeFile = workPath + r'\data\Intersections.shp'
arcpy.Intersect_analysis (in_features =segmentShapeFile3Name, out_feature_class=intersectionShapeFile, join_attributes="ONLY_FID", output_type="POINT")
#Create point id
#Add fields segment shape file, length and id
arcpy.AddField_management(in_table= intersectionShapeFile, field_name = "Intsec_ID", field_type="Long")

#Calculate intersection id
arcpy.CalculateField_management (in_table = intersectionShapeFile, field = "Intsec_ID", expression="!FID!")




##Map points to cornering roads
joinedIntersectionsShapeFile = crashShapeFile = workPath + r'\data\Adjacency.shp'
fieldmappings = arcpy.FieldMappings()

# Add all fields from inputs.
fieldmappings.addTable(intersectionShapeFile)
fieldmappings.addTable(segmentShapeFile3)

# Fields that shall remain
keepers = ["Intsec_ID","Segment_ID"] 

# Remove all output fields you don't want.
for field in fieldmappings.fields:
    if field.name not in keepers:
        fieldmappings.removeFieldMap(fieldmappings.findFieldMapIndex(field.name))


arcpy.SpatialJoin_analysis (target_features = intersectionShapeFile
, join_features = segmentShapeFile3, field_mapping=fieldmappings,
out_feature_class = joinedIntersectionsShapeFile, 
join_operation = "JOIN_ONE_TO_MANY", join_type = "KEEP_ALL",
match_option ="INTERSECT")



