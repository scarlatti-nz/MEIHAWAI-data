import os
import geopandas as gpd
import yaml
import sys
import rtree
from utils import load_shapefile_data_to_selection, load_config, save_empty_dataset, consistent_hash

config = load_config("config.yaml")
grid_identifier = sys.argv[1]
output_directory = f"{config['output_dir']}/grid_{grid_identifier}"

def remove_overlaps():
    """
        Remove overlaps between farms in the current grid and farms in the surrounding grids.
        Do this by checking if a farm in the current grid overlaps with a farm in a surrounding grid.
        If it does, then the farm with the smaller hash of farm_id is kept.
    """
    gdf1 = gpd.read_file(output_directory+"/farms.gpkg")
    if(len(gdf1) == 0):
        save_empty_dataset(output_directory+"/farms_aligned.gpkg")
        sys.exit(0)

    lat = grid_identifier.split("_")[0]
    lon = grid_identifier.split("_")[1]
    print(f"Filtering out farms for lat {lat} lon {lon}")
    lat_minus = str(round(float(lat)-config["grid_size"],4))
    lat_plus = str(round(float(lat)+config["grid_size"],4))
    lon_minus = str(round(float(lon)-config["grid_size"],4))
    lon_plus = str(round(float(lon)+config["grid_size"],4))
    left_folder = config["output_dir"]+"/grid_"+lat+"_"+lon_minus
    upper_left_folder = config["output_dir"]+"/grid_"+lat_plus+"_"+lon_minus
    upper_folder = config["output_dir"]+"/grid_"+lat_plus+"_"+lon
    upper_right_folder = config["output_dir"]+"/grid_"+lat_plus+"_"+lon_plus
    right_folder = config["output_dir"]+"/grid_"+lat+"_"+lon_plus
    lower_right_folder = config["output_dir"]+"/grid_"+lat_minus+"_"+lon_plus
    lower_folder = config["output_dir"]+"/grid_"+lat_minus+"_"+lon
    lower_left_folder = config["output_dir"]+"/grid_"+lat_minus+"_"+lon_minus
    for other_folder in [left_folder, upper_left_folder, upper_folder, upper_right_folder, right_folder, lower_right_folder, lower_folder, lower_left_folder]:
        print("Checking for "+other_folder)
        if os.path.exists(other_folder+"/farms.gpkg"):
            print("Removing overlaps looking at "+other_folder)
            gdf2 = gpd.read_file(other_folder+"/farms.gpkg")
            
            # create index for other grid. Here we are using rtree to speed up the process.
            index = rtree.index.Index()
            for idx2, feature2 in gdf2.iterrows():
                index.insert(idx2, feature2['geometry'].bounds)
            for idx1, feature1 in gdf1.iterrows():
                geometry1 = feature1['geometry']
                for idx2 in index.intersection(geometry1.bounds):
                    feature2 = gdf2.iloc[idx2]
                    geometry2 = feature2['geometry']
                    # check if a farm overlaps.
                    intersection_between = geometry1.intersection(geometry2).area
                    if(intersection_between > ((geometry1.area+geometry2.area) * config["minimum_overlap_percentage_for_intersection"]/100.0)):
                        change_id = consistent_hash(feature1['farm_id']) < consistent_hash(feature2['farm_id'])
                        if(change_id):
                            gdf1.at[idx1,'farm_id'] = feature2['farm_id'] # change the farm_id to that of the other farm.
    gdf1.to_file(output_directory+"/farms_aligned.gpkg", driver="GPKG",overwrite=True)

if __name__ == "__main__":
    remove_overlaps()
