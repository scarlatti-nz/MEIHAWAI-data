import geopandas as gpd
import rasterio as rio
import numpy as np
from utils import load_config
from rasterio.mask import mask
from skimage.measure import block_reduce
import os
import pandas as pd

config = load_config("config.yaml")
lucas = gpd.read_file("raw_data/mfe-lucas-nz-land-use-map-1990-2008-2012-2016-v011-SHP/lucas-nz-land-use-map-1990-2008-2012-2016-v011.shp")
farmland = lucas[lucas["LUCNA_2016"].isin(config["lucas_LUCNA_values"])]
dairy = farmland[farmland["SUBID_2016"]==502]
not_dairy = farmland[farmland["SUBID_2016"]!=502]

names = []
dairy_means = []
not_dairy_means = []
for file in os.listdir("raw_data/whitiwhiti-ora/pasture"):
    if(file.endswith(".tif")):
        names.append(file[:-4])
        with rio.open("raw_data/whitiwhiti-ora/pasture/"+file) as src:
            dairy_data, _ = mask(src, dairy.geometry, crop=True)
            not_dairy_data, _ = mask(src, not_dairy.geometry, crop=True)
            # downsample
            downsampled_data_dairy = block_reduce(dairy_data, block_size=(1, 4, 4), func=np.mean)
            downsampled_data_not_dairy = block_reduce(not_dairy_data, block_size=(1, 4, 4), func=np.mean)
            # flatten
            downsampled_data_dairy = downsampled_data_dairy.flatten()
            downsampled_data_not_dairy = downsampled_data_not_dairy.flatten()
            dairy_data[dairy_data==src.nodata] = np.nan
            not_dairy_data[not_dairy_data==src.nodata] = np.nan
            dairy_mean = np.nanmean(dairy_data)
            not_dairy_mean = np.nanmean(not_dairy_data)
            dairy_means.append(dairy_mean)
            not_dairy_means.append(not_dairy_mean)

dairy_means = np.array(dairy_means)
not_dairy_means = np.array(not_dairy_means)
df = pd.DataFrame({"dairy": dairy_means, "not_dairy": not_dairy_means}, index=names)
df.to_csv("pasture_means.csv")