import geopandas as gpd
import yaml
import sys
from shapely.geometry import Polygon, MultiPolygon, shape
import os
from utils import load_shapefile_data_to_selection, load_config, save_empty_dataset
import numpy as np
from shapely.ops import unary_union
import uuid

config = load_config("config.yaml")
grid_identifier = sys.argv[1]
output_directory = f"{config['output_dir']}/grid_{grid_identifier}"

def split_into_hectares():
    """
        Create a grid of 100m by 100m squares.
        Filter all hectares that are inside the grid selection.
        Filter all hectares that are inside the farms.
    """
    farms = gpd.read_file(output_directory + "/farms_aligned.gpkg")
    grid_selection = gpd.read_file(output_directory + "/grid_selection_without_buffer.gpkg").to_crs("EPSG:2193") # don't use the buffer
    if(len(farms) == 0):
        save_empty_dataset(output_directory + "/hectares.gpkg")
        sys.exit(0)

    min_x, min_y, max_x, max_y = grid_selection.total_bounds
    # first round min to nearest lower 100m, max to nearest higher 100m
    min_x = np.floor(min_x/100)*100
    min_y = np.floor(min_y/100)*100
    max_x = np.ceil(max_x/100)*100
    max_y = np.ceil(max_y/100)*100
    # make a grid of 100m by 100m squares
    cell_size_hectares = 100
    grid_polygons = []
    for x in range(int(min_x), int(max_x), cell_size_hectares):
        for y in range(int(min_y), int(max_y), cell_size_hectares):
            polygon = Polygon([
                (x, y),
                (x + cell_size_hectares, y),
                (x + cell_size_hectares, y + cell_size_hectares),
                (x, y + cell_size_hectares)
            ])
            grid_polygons.append(polygon)

    grid_gdf = gpd.GeoDataFrame(geometry=grid_polygons, crs=farms.crs)
    grid_gdf['centroid'] = grid_gdf.centroid
    centroid_gdf = gpd.GeoDataFrame(grid_gdf, geometry='centroid', crs=farms.crs)
    # make sure all hectares are inside the grid selection
    centroid_gdf = gpd.sjoin(centroid_gdf, grid_selection, how="inner").drop(columns=['index_right'])
    hectares_gdf = gpd.sjoin(centroid_gdf, farms, how="inner")
    hectares_gdf = hectares_gdf.set_geometry('geometry')
    hectares_gdf = hectares_gdf.drop(columns=['centroid', 'index_right'])

    # save the full hectares file too
    centroid_gdf = centroid_gdf.set_geometry('geometry')
    centroid_gdf = centroid_gdf.drop(columns=['centroid'])
    centroid_gdf.to_file(output_directory + "/hectares_all.gpkg",overwrite=True)

    if(hectares_gdf.empty):
        print("No hectares found")
        save_empty_dataset(output_directory + "/hectares.gpkg")
        sys.exit(0)
    else:
        hectares_gdf.to_file(output_directory + "/hectares.gpkg",overwrite=True)

if __name__ == "__main__":    
    split_into_hectares()