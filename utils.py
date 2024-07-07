import geopandas as gpd
import yaml
import sys
from shapely.geometry import Polygon
import os
import rasterio as rio
from rasterio.transform import Affine
from rasterio.features import shapes, geometry_mask
import numpy as np
import hashlib

def save_empty_dataset(output_path):
    empty = gpd.GeoDataFrame(geometry=[], crs="EPSG:2193")
    empty.to_file(output_path,overwrite=True)

def make_clean_folder(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    else:
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
            except Exception as e:
                print(e)

def load_config(config_file):
    with open(config_file, "r") as f:
        config = yaml.safe_load(f)
    return config

def load_shapefile_data_to_selection(shapefile_path, selection_path, shapefile_crs, output_directory,output_name,data_can_be_empty=False):
    try:
        selection = gpd.read_file(selection_path)
        rectangle = selection.to_crs(shapefile_crs).iloc[0].geometry
        minx, miny, maxx, maxy = rectangle.bounds
        data = gpd.read_file(shapefile_path, bbox=(minx, miny, maxx, maxy))
        data = data.to_crs("EPSG:2193")
        if not data.empty:
            data.to_file(os.path.join(output_directory, f"{output_name}.gpkg"), driver="GPKG",overwrite=True)
            success = len(data) > 0
            return success
        else:
            return data_can_be_empty
    except Exception as e:
        print(f"Error loading data: {e}")
        return False

def raster_helper_nan_mean(data):
    """
        Custom function for raster data to get the mean ignoring nan values. 
    """
    masked_data = np.ma.masked_invalid(data)
    mean_value = masked_data.mean() if masked_data.count() > 0 else np.nan
    return mean_value

def raster_helper_sum_range(lower,upper):
    """
        Custom function for raster data to get the count of values between lower, upper. Here we input lower, upper and get the function back.
    """
    def nan_sum_range(data):
        masked_data = np.ma.masked_invalid(data)
        mask = np.logical_and(masked_data >= lower, masked_data < upper)
        sum_value = np.sum(mask) if masked_data.count() > 0 else np.nan
        return sum_value
    return nan_sum_range

def load_image_to_selection(image_path, selection_path, image_crs, output_directory, output_name):
    try:
        selection = gpd.read_file(selection_path)
        rectangle = selection.to_crs(image_crs).iloc[0].geometry
        minx, miny, maxx, maxy = rectangle.bounds
        with rio.open(image_path) as dataset:
            minx, miny, maxx, maxy = rectangle.bounds
            minx = max(minx, dataset.bounds.left)
            maxx = min(maxx, dataset.bounds.right)
            miny = max(miny, dataset.bounds.bottom)
            maxy = min(maxy, dataset.bounds.top)
            if(minx<maxx and miny<maxy):
                x_res = dataset.res[0]
                y_res = dataset.res[1]
                start_col = int((minx - dataset.bounds.left) / x_res)
                start_row = int((dataset.bounds.top - maxy) / y_res)
                window_width = int((maxx - minx) / x_res)+1
                window_height = int((maxy - miny) / y_res)+1
                file_data = dataset.read(1, window=rio.windows.Window(start_col, start_row, window_width, window_height))
                # check if the size > 0
                if(file_data.size == 0):
                    return False
                else:
                    new_transform = Affine(dataset.transform.a, dataset.transform.b, minx, dataset.transform.d, dataset.transform.e, maxy)
                    with rio.open(
                        os.path.join(output_directory, f"{output_name}.tif"), 'w',
                        driver='GTiff', 
                        width=window_width, 
                        height=window_height, 
                        count=1, 
                        dtype=file_data.dtype, 
                        crs=dataset.crs, 
                        transform=new_transform,
                        nodata=dataset.nodata
                    ) as dst:
                        dst.write(file_data, 1)
            else:
                return False
        return True
    except Exception as e:
        print(f"Error loading data: {e}")
        return False

def read_image_polygonized(image_path):
    with rio.Env():
        with rio.open(image_path) as src:
            image = src.read(1)
            image[image == src.nodata] = np.nan
            height, width = image.shape
            polys = []
            raster_vals = []
            for row in range(height):
                for col in range(width):
                    if not np.isnan(image[row, col]):
                        ulx, uly = src.transform * (col, row)
                        lrx, lry = src.transform * (col + 1, row + 1)
                        polygon = {
                            'type': 'Polygon',
                            'coordinates': [[(ulx, uly), (lrx, uly), (lrx, lry), (ulx, lry)]]
                        }
                        polys.append(polygon)
                        raster_vals.append(image[row, col])
        if(len(polys)>0):
            results = [{'properties': {'raster_val': raster_vals[p]}, 'geometry': polys[p]} for p in range(len(polys))]
            gpd_polygonized_raster = gpd.GeoDataFrame.from_features(results, crs=src.crs)
            return gpd_polygonized_raster
        else:
            return None

# helper function to hash the grid positions into a number. Needed to break ties. 
def consistent_hash(text):
    sha256 = hashlib.sha256()
    sha256.update(text.encode())
    hash_value = int(sha256.hexdigest(), 16)
    return hash_value