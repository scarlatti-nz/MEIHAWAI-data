import geopandas as gpd
import yaml
import sys
from shapely.geometry import Polygon
import os
from utils import load_shapefile_data_to_selection, save_empty_dataset, make_clean_folder, load_config
import numpy as np
from shapely.ops import unary_union
import uuid

config = load_config("config.yaml")
grid_identifier = sys.argv[1]
output_directory = f"{config['output_dir']}/grid_{grid_identifier}"

def build_selection_box():
    """
        Turn the grid identifier into a geospatial selection box, both with and without a buffer.
    """
    lat_start = grid_identifier.split("_")[-2]
    lon_start = grid_identifier.split("_")[-1]
    grid_size = config["grid_size"]
    buffer = config["buffer"]
    lat_end = round(float(lat_start) + grid_size, 4)
    lon_end = round(float(lon_start) + grid_size, 4)
    print(f"Building selection box on grid: {lat_start},{lon_start} to {lat_end},{lon_end}")
    rectangle_latlon_with_buffer = Polygon([
        (float(lon_start) - buffer, float(lat_start) - buffer),
        (float(lon_start) - buffer, float(lat_end) + buffer),
        (float(lon_end) + buffer, float(lat_end) + buffer),
        (float(lon_end) + buffer, float(lat_start) - buffer)
    ])
    rectangle_latlon_without_buffer = Polygon([
        (float(lon_start), float(lat_start)),
        (float(lon_start), float(lat_end)),
        (float(lon_end), float(lat_end)),
        (float(lon_end), float(lat_start))
    ])
    selection_with_buffer = gpd.GeoDataFrame(geometry=[rectangle_latlon_with_buffer], crs="EPSG:4326")
    selection_with_buffer.to_file(os.path.join(output_directory, "grid_selection_with_buffer.gpkg"),overwrite=True)
    selection_without_buffer = gpd.GeoDataFrame(geometry=[rectangle_latlon_without_buffer], crs="EPSG:4326")
    selection_without_buffer.to_file(os.path.join(output_directory, "grid_selection_without_buffer.gpkg"),overwrite=True)

def load_linz():
    print("Loading linz data")
    linz_success = load_shapefile_data_to_selection(
        shapefile_path = 'raw_data/lds-nz-property-titles-including-owners-SHP_oct2021/nz-property-titles-including-owners.shp',
        selection_path = os.path.join(output_directory, "grid_selection_with_buffer.gpkg"),
        shapefile_crs = "EPSG:4326",
        output_directory = output_directory,
        output_name = "linz_selection"
    )
    if not linz_success:
        print("Halting as no linz data could be loaded")
        save_empty_dataset(output_directory + "/farms.gpkg")
        sys.exit(0)

def load_lucas():
    print("Loading lucas data")
    lucas_success = load_shapefile_data_to_selection(
        shapefile_path = 'raw_data/mfe-lucas-nz-land-use-map-1990-2008-2012-2016-v011-SHP/lucas-nz-land-use-map-1990-2008-2012-2016-v011.shp',
        selection_path = os.path.join(output_directory, "grid_selection_with_buffer.gpkg"),
        shapefile_crs = "EPSG:2193",
        output_directory = output_directory,
        output_name = "lucas_selection"
    )
    if not lucas_success:
        print("Halting as no lucas data could be loaded")
        save_empty_dataset(output_directory + "/farms.gpkg")
        sys.exit(0)

def create_farms():
    """
        Generate the farms from the LINZ and LUCAS data.
        Start by filtering to sizable land parcels with farmland.
        Then find neighbours based on ownership.
        Join the neighbours into farms.
        Finally, remove farms which are overlapping other farms.
    """
    print("Creating farms")
    lucas = gpd.read_file(output_directory + "/lucas_selection.gpkg")
    farmland_codes = config["lucas_LUCNA_values"]
    lucas = lucas[lucas["LUCNA_2016"].isin(farmland_codes)]

    if(len(lucas)==0):
        print("Halting as no farmland data")
        save_empty_dataset(output_directory + "/farms.gpkg")
        sys.exit(0)

    farmland = lucas.unary_union

    if(farmland.is_empty):
        print("Halting as no farmland data")
        save_empty_dataset(output_directory + "/farms.gpkg")
        sys.exit(0)

    linz = gpd.read_file(output_directory + "/linz_selection.gpkg")
    linz_filtered = linz[linz.geometry.area > config["minimum_land_parcel_size"]]
    linz_filtered = linz_filtered.explode(index_parts=True)
    minimum_percent_farmland = config["minimum_percentage_farmland_for_land_parcel_to_be_considered_farmland"]/100
    linz_farmland = linz_filtered[linz_filtered["geometry"].intersection(farmland).area > linz_filtered["geometry"].area * minimum_percent_farmland]

    if(len(linz_farmland)>0):
        linz_farmland.to_file(output_directory + "/linz_farmland.gpkg",overwrite=True)
    else:
        print("Halting as no land parcels were found to be farmland")
        save_empty_dataset(output_directory + "/farms.gpkg")
        sys.exit(0)
    
    # small function to get a clean "owners" string, lowering the case & stipping whitespace etc.
    clean_owners = lambda owners:  "," + ",".join(sorted([owner.strip().lower() for owner in str(owners).split(",") if len(owner.strip()) > 0])) + ","
    linz_farmland = linz_farmland.copy()
    linz_farmland.loc[:,"owners_clean"] = linz_farmland["owners"].apply(clean_owners).copy()
    owners_list = [set(linz_farmland.iloc[i]["owners_clean"].split(",")[1:-1]) for i in range(len(linz_farmland))]

    # find the neighbours for each land parcel
    neighbours = {}
    print("Finding neighbours...")
    print(len(owners_list))
    for i in range(len(linz_farmland)):
        neighbours[i] = linz_farmland.sindex.query(linz_farmland.iloc[i].geometry.buffer(config["neighbour_distance_threshold"]), predicate='intersects') # get close neighbours
        neighbours[i] = [j for j in neighbours[i] if (len(owners_list[i] & owners_list[j])>0  or (linz_farmland.iloc[j].geometry.area < 100000))] # only keep neighbours which have overlapping owners, unless the neighbouring parcel is less than 10ha, then join it up regardless.
   
    # join the neighbours together into farms
    print("Joining neighbours into farms...")
    new_geometries = []
    farm_owners = []
    farm_ids = []
    added = np.zeros(len(linz_farmland)) # keep track of which land parcels have been added to a farm already
    for i in range(len(linz_farmland)): # go through every land parcel
        if added[i] == 0:
            # search the network of neighbours, up to a number of levels deep.
            # owners must have names overlapping with the initial parcel. 
            # So if the owners of 3 parcels are [A,A+B,B+C] and I start at the first parcel, I will get A+B as the new owners.
            # If I start at the second parcel I will get [A+B+C].
            neighbours_set = set([n for n in neighbours[i] if added[n] == 0])
            for depth in range(100): # maximum number of levels deep to search. This is plenty. Not expecting to find farmland which is a string of land parcels 100 long.
                new_neighbours_set = set()
                for j in neighbours_set: # go through current list of neighbours & update with their neighbours, if the names align, or if they are a small parcel at very shallow depth.
                    new_neighbours_set.update([n for n in neighbours[j] if ( (len(owners_list[i] & owners_list[n])>0 or (linz_farmland.iloc[n].geometry.area < 100000 and depth < 3))  and added[n] == 0 )])
                current_length = len(neighbours_set)
                neighbours_set.update(new_neighbours_set)
                if(len(neighbours_set) == current_length): # if no new neighbours were added, end the search.
                    break
            owners_set = set()
            for j in neighbours_set:
                added[j] = 1
                owners_set.update(owners_list[j])
            polygons_in_group = [linz_farmland.iloc[idx].geometry for idx in neighbours_set]
            multipolygon = unary_union(polygons_in_group)
            new_geometries.append(multipolygon)
            farm_ids.append(str(uuid.uuid4())) # generate a unique id for the farm
            farm_owners.append(",".join(sorted(list(owners_set))))

    farms = gpd.GeoDataFrame({"farm_id":farm_ids, "owners":farm_owners, "geometry": new_geometries}, crs=linz_farmland.crs)
    farms = farms[farms.geometry.area > config["minimum_farm_size"]]

    # remove farms which are overlapping other farms.
    keep = np.zeros(len(farms))
    for i in range(len(farms)):
        remove = False
        for j in range(len(farms)):
            if(not i==j):
                if(keep[j]==1):
                    if(farms.iloc[i].geometry.intersects(farms.iloc[j].geometry)):
                        area_intersection = farms.iloc[i].geometry.intersection(farms.iloc[j].geometry).area
                        if(area_intersection > (config["minimum_overlap_percentage_for_intersection"]/100.0) * farms.iloc[i].geometry.area):
                            remove = True
                            break
        if(not remove):
            keep[i] = 1
    farms = farms[keep==1]

    if(len(farms)>0):
        farms.to_file(output_directory + "/farms.gpkg",overwrite=True)
    else:
        print("Halting as no farms could be made")
        save_empty_dataset(output_directory + "/farms.gpkg")
        sys.exit(0)

if __name__ == "__main__":
    make_clean_folder(output_directory)
    build_selection_box()
    load_linz()
    load_lucas()
    create_farms()