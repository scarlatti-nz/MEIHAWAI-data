import geopandas as gpd
import sys
import os
from utils import *
import numpy as np
import rasterio as rio
from rasterio.merge import merge
import rasterstats as rs
from scipy.ndimage import convolve
import pandas as pd
import pyogrio

config = load_config("config.yaml")
grid_identifier = sys.argv[1]
output_directory = f"{config['output_dir']}/grid_{grid_identifier}"

def load_MAL_data():
    """
        Load maximum allowable load data and model for each hectare. 
        We first combine the catchment geometric data (MAL and catchment data.shp) with the rainfall data (totann7216.tif) to get the rainfall for each catchment.
        Then we merge with data from the csv files (MAL and catchment data.csv and Sediment25Nov2023_RECout.csv).
        The csvs have data on the terminal segments, nitrogen, phosphorous, sediment, load ratios, watershed area, yield etc for each catchment.
        We go through each terminal segment and use it's values to get the MAL we want to spread across all catchments feeding into it. 
        We use the rainfall data (normalised to sum to 1) to get the MAL for each catchment (MAL we say is proportional to rainfall).
        We also get the maximum load ratio for all catchments feeding into the terminal segment.
        Finally, the load for each catchment is calculated as the catchment yield * catchment watershed area.
    """
    print("Loading MAL data")
    loading_success = load_shapefile_data_to_selection(
        shapefile_path = 'raw_data/whitiwhiti-ora/MAL and catchment data/MAL and catchment data.shp',
        selection_path = os.path.join(output_directory, "grid_selection_with_buffer.gpkg"),
        shapefile_crs = "EPSG:2193",
        output_directory = output_directory,
        output_name = "MAL_and_catchment_data"
    )
    if(not loading_success):
        print("MAL data failed to load")
        save_empty_dataset(output_directory + "/hectares_with_stuff_final.gpkg")
        sys.exit(0)
    MAL_and_catchment_data = pyogrio.read_dataframe(output_directory+"/MAL_and_catchment_data.gpkg")
    MAL_and_catchment_data.rename(columns={"REC2_Termi":"REC2_TerminalSegment"}, inplace=True)
    polygonised_rainfall = read_image_polygonized("raw_data/rainfall/totann7216.tif")
    polygonised_rainfall = polygonised_rainfall.to_crs(MAL_and_catchment_data.crs)
    polygonised_rainfall["rainfall"] = polygonised_rainfall["raster_val"]
    pyogrio.write_dataframe(polygonised_rainfall, output_directory+"/polygonised_rainfall.gpkg")
    N_and_P_csv = pd.read_csv("raw_data/whitiwhiti-ora/MAL and catchment data/MAL and catchment data.csv")[['nzsegment','REC2_TerminalSegment', 'MAL_TN_rv_m', 'MAL_TP_rv', 'LoadMALRatio_TN', 'LoadMALRatio_TP', 'WaterShedAreaKM2', 'TPYield', 'TNYield', 'upcoordX', 'upcoordY', 'downcoordX', 'downcoordY']]
    sediment_csv = pd.read_csv("raw_data/whitiwhiti-ora/MAL and catchment data/Sediment25Nov2023_RECout.csv")[['CurrentSedLoad', 'Pressure','CurrentSedYield','LoadEx']].rename(columns={"Pressure":"LoadMALRatio_TS", 'LoadEx':'LoadEx_sediment'})
    sediment_csv["MAL_TS"] = sediment_csv["CurrentSedLoad"] - sediment_csv["LoadEx_sediment"]
    all_catchments_csv = pd.concat([N_and_P_csv, sediment_csv], axis=1)
    terminal_nodes = MAL_and_catchment_data["REC2_TerminalSegment"].unique() # get the terminal nodes for the current selection.
    for t in terminal_nodes:
        MAL_terminal_N = all_catchments_csv.loc[all_catchments_csv["nzsegment"]==t]["MAL_TN_rv_m"].values[0]
        MAL_terminal_P = all_catchments_csv.loc[all_catchments_csv["nzsegment"]==t]["MAL_TP_rv"].values[0]
        MAL_terminal_TS = all_catchments_csv.loc[all_catchments_csv["nzsegment"]==t]["MAL_TS"].values[0]
        S_t = all_catchments_csv[all_catchments_csv["REC2_TerminalSegment"]==t].copy()
        mid_x = (S_t["downcoordX"]+S_t["upcoordX"]) / 2
        mid_y = (S_t["downcoordY"]+S_t["upcoordY"]) / 2
        points = gpd.GeoDataFrame(geometry=gpd.points_from_xy(mid_x, mid_y), crs="EPSG:2193").to_crs(MAL_and_catchment_data.crs)
        # get the value in the rainfall gpkg for each point
        points = gpd.sjoin(points, polygonised_rainfall, how="left")
        S_t.loc[:, "rainfall_sum"] = points["rainfall"].values  * S_t["WaterShedAreaKM2"].values
        rainfall_ratios = S_t["rainfall_sum"].values / S_t["rainfall_sum"].sum()
        MAL_Nitrogen = MAL_terminal_N * rainfall_ratios
        MAL_Phosphorous = MAL_terminal_P * rainfall_ratios
        MAL_Sediment = MAL_terminal_TS * rainfall_ratios
        Load_Ratio_N = S_t["LoadMALRatio_TN"].max()
        Load_Ratio_P = S_t["LoadMALRatio_TP"].max()
        Load_Ratio_TS = S_t["LoadMALRatio_TS"].max()
        Load_N = S_t["WaterShedAreaKM2"].values * S_t["TNYield"].values
        Load_P = S_t["WaterShedAreaKM2"].values * S_t["TPYield"].values
        Load_TS = S_t["WaterShedAreaKM2"].values * S_t["CurrentSedYield"].values
        # join back to MAL_and_catchment_data
        for i, segment in enumerate(S_t["nzsegment"]):
            if segment in MAL_and_catchment_data["nzsegment"].unique():
                MAL_and_catchment_data.loc[MAL_and_catchment_data["nzsegment"]==segment, "MAL_Nitrogen"] = MAL_Nitrogen[i]
                MAL_and_catchment_data.loc[MAL_and_catchment_data["nzsegment"]==segment, "MAL_Phosphorous"] = MAL_Phosphorous[i]
                MAL_and_catchment_data.loc[MAL_and_catchment_data["nzsegment"]==segment, "MAL_Sediment"] = MAL_Sediment[i]
                MAL_and_catchment_data.loc[MAL_and_catchment_data["nzsegment"]==segment, "Max_Load_Ratio_N"] = Load_Ratio_N
                MAL_and_catchment_data.loc[MAL_and_catchment_data["nzsegment"]==segment, "Max_Load_Ratio_P"] = Load_Ratio_P
                MAL_and_catchment_data.loc[MAL_and_catchment_data["nzsegment"]==segment, "Max_Load_Ratio_TS"] = Load_Ratio_TS
                MAL_and_catchment_data.loc[MAL_and_catchment_data["nzsegment"]==segment, "Load_N"] = Load_N[i]
                MAL_and_catchment_data.loc[MAL_and_catchment_data["nzsegment"]==segment, "Load_P"] = Load_P[i]
                MAL_and_catchment_data.loc[MAL_and_catchment_data["nzsegment"]==segment, "Load_TS"] = Load_TS[i]
                MAL_and_catchment_data.loc[MAL_and_catchment_data["nzsegment"]==segment, "rainfall"] = points["rainfall"].iloc[i]
                MAL_and_catchment_data.loc[MAL_and_catchment_data["nzsegment"]==segment, "rainfall_sum"] = S_t["rainfall_sum"].iloc[i]
                MAL_and_catchment_data.loc[MAL_and_catchment_data["nzsegment"]==segment, "WaterShedAreaKM2"] = S_t["WaterShedAreaKM2"].iloc[i]
                MAL_and_catchment_data.loc[MAL_and_catchment_data["nzsegment"]==segment, "CurrentSedYield"] = S_t["CurrentSedYield"].iloc[i]
    pyogrio.write_dataframe(MAL_and_catchment_data, output_directory+"/MAL_and_catchment_data_processed.gpkg")

def join_to_MAL_data():
    print("Joining to MAL data")
    hectares = pyogrio.read_dataframe(output_directory+"/hectares.gpkg")
    if(len(hectares) == 0):
        save_empty_dataset(output_directory + "/hectares_with_stuff_final.gpkg")
        sys.exit(0)
    
    MAL_and_catchment_data = pyogrio.read_dataframe(output_directory+"/MAL_and_catchment_data_processed.gpkg")
    renaming_dict = dict(zip(MAL_and_catchment_data.columns, ["catchment_"+column if column != "geometry" else column for column in MAL_and_catchment_data.columns]))
    MAL_and_catchment_data = MAL_and_catchment_data.rename(columns=renaming_dict)
    hectares["centroid"] = hectares.centroid
    hectares = hectares.set_geometry("centroid")
    hectares = gpd.sjoin(hectares, MAL_and_catchment_data, how="left")
    hectares = hectares.set_geometry("geometry")
    hectares = hectares.drop(columns=["centroid", "index_right"])
    pyogrio.write_dataframe(hectares, output_directory+"/hectares_with_stuff_v1.gpkg")

def load_dairy_typology():
    print("Loading dairy typology")
    dairy_typology_success = load_shapefile_data_to_selection(
        shapefile_path = 'raw_data/whitiwhiti-ora/dairy-typology/dairy_typology.gdb',
        selection_path = os.path.join(output_directory, "grid_selection_without_buffer.gpkg"),
        shapefile_crs = "EPSG:2193",
        output_directory = output_directory,
        output_name = "dairy_typology_selection"
    )
    if(not dairy_typology_success):
        print("Dairy typology failed to load")
        save_empty_dataset(output_directory + "/hectares_with_stuff_final.gpkg")
        sys.exit(0)

def join_to_dairy_typology():
    print("Joining to dairy typology")
    hectares = pyogrio.read_dataframe(output_directory+"/hectares_with_stuff_v1.gpkg")
    if(len(hectares) == 0):
        save_empty_dataset(output_directory + "/hectares_with_stuff_final.gpkg")
        sys.exit(0)

    hectares["centroid"] = hectares.centroid
    hectares = hectares.set_geometry("centroid")
    dairy_typology = pyogrio.read_dataframe(output_directory+"/dairy_typology_selection.gpkg")
    dairy_typology = dairy_typology.to_crs(hectares.crs)
    dairy_typology.columns = ["dairy_typology_"+column if column != "geometry" else column for column in dairy_typology.columns]
    hectares = gpd.sjoin(hectares, dairy_typology, how="left")
    hectares = hectares[hectares["index_right"].notna()]
    hectares = hectares.set_geometry("geometry")
    hectares = hectares.drop(columns=["centroid", "index_right"])
    pyogrio.write_dataframe(hectares, output_directory+"/hectares_with_stuff_v2.gpkg")

def load_height_data():
    print("Loading height data")
    height_north_regions = ["aklnd", "ecape", "hksbay", "nthcape", "taranaki", "waikato", "well", "whngrei"]
    height_south_regions = ["chch", "dunedin", "greymth", "invcgll", "kaik", "mtcook", "nelson", "teanau", "waitaki"]
    make_clean_folder(output_directory + "/height_maps")
    for island in ["north", "south"]:
        regions = height_north_regions if island == "north" else height_south_regions
        for region in regions:
            file_path = "raw_data/lris-nzdem-" + island + "-island-25-metre-GTiff/" + region + "_25r.tif"
            load_image_to_selection(
                image_path=file_path,
                selection_path=output_directory + "/grid_selection_without_buffer.gpkg",
                image_crs="EPSG:27200",
                output_directory=output_directory + "/height_maps",
                output_name="height-"+region
            )

def join_to_height_data():
    """
        Here we use the height maps to get the gradient in each hectare using a convolution kernel.
    """
    print("Joining to height data")
    hectares = pyogrio.read_dataframe(output_directory+"/hectares_with_stuff_v2.gpkg")
    if(len(hectares) == 0):
        save_empty_dataset(output_directory + "/hectares_with_stuff_final.gpkg")
        sys.exit(0)

    files = [output_directory + "/height_maps/" + file for file in os.listdir(output_directory + "/height_maps")]
    src_files_to_merge = [rio.open(file) for file in files]
    nodata_value = rio.open(files[0]).nodata
    height_data, affine = merge(src_files_to_merge)
    height_data = height_data[0]

    # do processing on merged height map.
    kernel = np.array(
        [
            [-3 - 3j, 0 - 10j, +3 - 3j],
            [-10 + 0j, 0 + 0j, +10 + 0j],
            [-3 + 3j, 0 + 10j, +3 + 3j],
        ]
    )
    angle_data = np.abs(convolve(height_data, kernel)).astype(np.float32)
    # the convolution will return values in units of (1/25) * (1/32) so we should divide by 800 to get the gradient
    # 1/25 because the height data is in metres, with 25m resolution
    # 1/32 because the sum of the absolute values of the real parts of kernel matrix is 32
    angle_data /= 800.0
    # convert gradient to degrees
    np.arctan(angle_data, out=angle_data)
    np.degrees(angle_data, out=angle_data)
    np.round(angle_data, out=angle_data)

    # save angle data
    with rio.open(
        output_directory + "/angle_data.tif",
        "w",
        driver="GTiff",
        height=angle_data.shape[0],
        width=angle_data.shape[1],
        count=1,
        dtype=angle_data.dtype,
        crs=rio.open(files[0]).crs,
        transform=affine,
        nodata=nodata_value,
    ) as dst:
        dst.write(angle_data, 1)
    
    hectares = hectares.to_crs("EPSG:27200")

    # get the height statistics
    height_stat_functions = {'avg_height': raster_helper_nan_mean}
    height_stats = rs.zonal_stats(
        hectares, 
        height_data, 
        affine=affine, 
        stats="mean", 
        nodata=np.nan,
        add_stats=height_stat_functions
    )
    
    for key in height_stat_functions.keys():
        hectares[key] = [stat[key] for stat in height_stats]
    
    # get the angle statistics
    angle_stat_functions = {
        "angl_avg": raster_helper_nan_mean,
        "angl_0_5": raster_helper_sum_range(0,5),
        "angl_5_10": raster_helper_sum_range(5,10),
        "angl_10_15": raster_helper_sum_range(10,15),
        "angl_15_20": raster_helper_sum_range(15,20),
        "angl_20_25": raster_helper_sum_range(20,25),
        "angl_25_30": raster_helper_sum_range(25,30),
        "angl_30+": raster_helper_sum_range(30,90) 
    }
    angle_stats = rs.zonal_stats(
        hectares,
        angle_data,
        affine=affine,
        stats="mean",
        nodata=np.nan,
        add_stats=angle_stat_functions
    )
    for key in angle_stat_functions.keys():
        hectares[str(key)] = [stat[key] for stat in angle_stats]
    pyogrio.write_dataframe(hectares, output_directory+"/hectares_with_stuff_v3.gpkg")

def load_lri():
    print("Loading lri")
    lri_success = load_shapefile_data_to_selection(
        shapefile_path = 'raw_data/lris-nzlri-land-use-capability-2021-SHP/nzlri-land-use-capability-2021.shp',
        selection_path = os.path.join(output_directory, "grid_selection_without_buffer.gpkg"),
        shapefile_crs = "EPSG:2193",
        output_directory = output_directory,
        output_name = "lri_selection"
    )
    if(not lri_success):
        print("Lri failed to load")
        save_empty_dataset(output_directory + "/hectares_with_stuff_final.gpkg")
        sys.exit(0)

def join_to_lri():
    print("Joining to lri")
    hectares = pyogrio.read_dataframe(output_directory + "/hectares_with_stuff_v3.gpkg")
    if(len(hectares) == 0):
        save_empty_dataset(output_directory + "/hectares_with_stuff_final.gpkg")
        sys.exit(0)

    lri = pyogrio.read_dataframe(output_directory+"/lri_selection.gpkg")
    lri = lri.to_crs(hectares.crs)
    lri.columns = ["lri_"+column if column != "geometry" else column for column in lri.columns]
    hectares["centroid"] = hectares.centroid
    hectares = hectares.set_geometry("centroid")
    hectares = gpd.sjoin(hectares, lri, how="left")
    hectares = hectares.set_geometry("geometry")
    hectares = hectares.drop(columns=["centroid", "index_right"])
    pyogrio.write_dataframe(hectares, output_directory+"/hectares_with_stuff_v4.gpkg")

def join_to_regions():
    print("Joining to regions")
    hectares = pyogrio.read_dataframe(output_directory+"/hectares_with_stuff_v4.gpkg")
    regional_gdf = pyogrio.read_dataframe("raw_data/regional_councils/regional-council-2023-clipped-generalised.shp")
    regional_gdf = regional_gdf.to_crs(hectares.crs)
    regional_gdf = regional_gdf[["REGC2023_1", "geometry"]]
    hectares["centroid"] = hectares.centroid
    hectares = hectares.set_geometry("centroid")
    hectares = gpd.sjoin(hectares, regional_gdf, how="left")
    hectares = hectares.set_geometry("geometry")
    hectares.drop(columns=["centroid", "index_right"], inplace=True)
    pyogrio.write_dataframe(hectares, output_directory+"/hectares_with_stuff_v5.gpkg")

def load_whitiwhiti_ora_economic_indicators():
    print("Loading whitiwhiti ora economic indicators")
    make_clean_folder(output_directory + "/whitiwhiti_ora_economic_indicators")
    for root, dirs, files in os.walk("raw_data/whitiwhiti-ora/economic-indicators/"):
        for file in files:
            if file.endswith(".tif"):
                load_image_to_selection(
                    image_path=root+"/"+file,
                    selection_path=output_directory+"/grid_selection_without_buffer.gpkg",
                    image_crs="EPSG:4326",
                    output_directory=output_directory+"/whitiwhiti_ora_economic_indicators",
                    output_name=file[:-4]
                )

def join_to_whitiwhiti_ora_economic_indicators():
    print("Joining to whitiwhiti ora economic indicators")
    hectares = pyogrio.read_dataframe(output_directory+"/hectares_with_stuff_v5.gpkg")
    hectares["temp_id"] = hectares.index
    for file in os.listdir(output_directory+"/whitiwhiti_ora_economic_indicators"):
        if(file.endswith(".tif")):
            gpd_polygonized_raster = read_image_polygonized(output_directory+"/whitiwhiti_ora_economic_indicators/"+file) # turn the raster into polygons
            if(not gpd_polygonized_raster is None):
                gpd_polygonized_raster = gpd_polygonized_raster.to_crs(hectares.crs)
                pyogrio.write_dataframe(gpd_polygonized_raster, output_directory+"/whitiwhiti_ora_economic_indicators/"+file[:-4]+".gpkg")
                gpd_polygonized_raster["raster_area"]=gpd_polygonized_raster.area
                intersection = gpd.overlay(hectares, gpd_polygonized_raster, how="intersection")
                intersection["area"] = intersection.area
                intersection.loc[intersection["raster_val"].isna(), "area"] = 0
                intersection["value"]=intersection["raster_val"]*(intersection["area"]/10000) # area is in m^2, we want per hectare share
                stats = intersection.groupby("temp_id")[["value"]].sum()
                if("per-10ha" in file):
                    stats["value"] = stats["value"] / 10
                    stats = stats.rename(columns={"value":"economic_indicator_"+file[:-4].replace("per-10ha","")+"_value"})
                else:
                    stats = stats.rename(columns={"value":"economic_indicator_"+file[:-4]+"_value"})
                hectares = hectares.merge(stats, left_on="temp_id", right_index=True, how="left")
    pyogrio.write_dataframe(hectares, output_directory+"/hectares_with_stuff_v6.gpkg")

def load_forestry_data():
    print("Loading forestry data")
    file_path = "raw_data/whitiwhiti-ora/forestry/Pradiata_300Index.tif"
    load_image_to_selection(
        image_path=file_path,
        selection_path=output_directory+"/grid_selection_without_buffer.gpkg",
        image_crs="EPSG:2193",
        output_directory=output_directory,
        output_name="Radiata_300Index"
    )

def join_to_forestry_data():
    print("Joining to forestry data")
    hectares = pyogrio.read_dataframe(output_directory+"/hectares_with_stuff_v6.gpkg")
    with rio.open(output_directory+"/Radiata_300Index.tif") as src:
        Radiata300_data = src.read(1)
        Radiata300_data[Radiata300_data == src.nodata] = np.nan
        affine = src.transform
        raster_crs = src.crs

    hectares = hectares.to_crs(raster_crs)
    raster_stats = rs.zonal_stats(
        hectares,
        Radiata300_data,
        affine=affine,
        stats="mean",
        add_stats={'avg_Radiata300_m3_per_ha': raster_helper_nan_mean}
    )
    hectares["economic_indicator_avg_Radiata300_m3_per_ha"] = [stat["avg_Radiata300_m3_per_ha"] for stat in raster_stats]
    hectares["economic_indicator_Radiata300_revenue_per_ha"] = hectares["economic_indicator_avg_Radiata300_m3_per_ha"] * config["Radiata300_dollars_per_m3"]
    pyogrio.write_dataframe(hectares, output_directory+"/hectares_with_stuff_v7.gpkg")

def load_pasture_data():
    print("Loading pasture data")
    make_clean_folder(output_directory + "/pasture")
    for file in os.listdir("raw_data/whitiwhiti-ora/pasture"):
        if(file.endswith(".tif")):
            load_image_to_selection(
                image_path="raw_data/whitiwhiti-ora/pasture/"+file,
                selection_path=output_directory+"/grid_selection_without_buffer.gpkg",
                image_crs="EPSG:2193",
                output_directory=output_directory+"/pasture",
                output_name=file[:-4]
            )

def join_to_pasture_data():
    """
        Here we join to pasture data. 
    """
    print("Joining to pasture data")
    hectares = pyogrio.read_dataframe(output_directory+"/hectares_with_stuff_v7.gpkg")
    for file in os.listdir(output_directory+"/pasture"):
        if(file.endswith(".tif")):
            with rio.open(output_directory+"/pasture/"+file) as src:
                pasture_data = src.read(1)
                pasture_data[pasture_data == src.nodata] = np.nan
                affine = src.transform
                raster_crs = src.crs
            hectares = hectares.to_crs(raster_crs)
            raster_stats = rs.zonal_stats(
                hectares,
                pasture_data,
                affine=affine,
                stats="mean",
                add_stats={"avg_pasture_yield": raster_helper_nan_mean}
            )
            hectares[file[:-4]] = [stat["avg_pasture_yield"] for stat in raster_stats]
    pyogrio.write_dataframe(hectares, output_directory+"/hectares_with_stuff_v8.gpkg")

def join_to_dairy_and_BLNZ_profitability():
    """
        We assume profitability and revenue per hectare is proportional to the regional averages with +-10% variation based on pasture quality.
        Pasture quality is measured relative to all other pasture used for the same purpose (dairy or not dairy).
    """
    print("Joining to Dairy and BLNZ profitability")
    hectares = pyogrio.read_dataframe(output_directory+"/hectares_with_stuff_v8.gpkg")

    # S&B
    BLNZ_data = pd.read_excel("raw_data/profitability/benchmarks.xlsx", sheet_name="BLNZ profit")
    BLNZ_all_classes = BLNZ_data.loc[BLNZ_data["Class"]=="All classes"]
    BLNZ_profit_per_ha_all_classes_by_region = dict(zip(BLNZ_all_classes["Region"], BLNZ_all_classes["profit provisional 2021-22"]))
    BLNZ_revenue_per_ha_all_classes_by_region = dict(zip(BLNZ_all_classes["Region"], BLNZ_all_classes["revenue provisional 2021-22"]))
    regional_gdf_regions_to_BLNZ_regions = {
        "Northland Region": "Northland / Waikato / BoP",
        "Auckland Region": "Northland / Waikato / BoP",
        "Waikato Region": "Northland / Waikato / BoP",
        "Bay of Plenty Region": "Northland / Waikato / BoP",
        "Gisborne Region": "East Coast",
        "Hawke's Bay Region": "East Coast",
        "Taranaki Region": "Taranaki / Manawatu",
        "Manawatū-Whanganui Region": "Taranaki / Manawatu",
        "Wellington Region": "Taranaki / Manawatu",
        "West Coast Region": "All New Zealand",
        "Canterbury Region": "Marlborough / Canterbury",
        "Otago Region": "Otago / Southland",
        "Southland Region": "Otago / Southland",
        "Tasman Region": "All New Zealand",
        "Nelson Region": "All New Zealand",
        "Marlborough Region": "Marlborough / Canterbury",
        "Area Outside Region": "All New Zealand"
    }
    hectares["B+LNZ class region"] = hectares["REGC2023_1"].map(regional_gdf_regions_to_BLNZ_regions)
    hectares["B+LNZ profit per ha"] = hectares["B+LNZ class region"].map(BLNZ_profit_per_ha_all_classes_by_region)
    hectares["B+LNZ revenue per ha"] = hectares["B+LNZ class region"].map(BLNZ_revenue_per_ha_all_classes_by_region)
    
    # Dairy
    Dairy_data = pd.read_excel("raw_data/profitability/benchmarks.xlsx", sheet_name="DairyNZ profit")
    Dairy_profit_per_ha_all_classes_by_region = dict(zip(Dairy_data["Region"], Dairy_data["Operating Profit ($/ha) 2021/22"]))
    Dairy_revenue_per_ha_all_classes_by_region = dict(zip(Dairy_data["Region"], Dairy_data["Operating Revenue ($/ha) 2021/22 estimate"]))
    regional_gdf_regions_to_DairyNZ_regions = {
        "Northland Region": "Northland",
        "Auckland Region": "Northland",
        "Waikato Region": "Waikato",
        "Bay of Plenty Region": "Bay of Plenty",
        "Gisborne Region": "Lower North Island",
        "Hawke's Bay Region": "Lower North Island",
        "Taranaki Region": "Taranaki",
        "Manawatū-Whanganui Region": "Lower North Island",
        "Wellington Region": "Lower North Island",
        "West Coast Region": "West Coast - Top of the South",
        "Canterbury Region": "Canterbury",
        "Otago Region": "Otago - Southland",
        "Southland Region": "Otago - Southland",
        "Tasman Region": "West Coast - Top of the South",
        "Nelson Region": "West Coast - Top of the South",
        "Marlborough Region": "West Coast - Top of the South",
        "Area Outside Region": "National"
    }
    hectares["DairyNZ region"] = hectares["REGC2023_1"].map(regional_gdf_regions_to_DairyNZ_regions)
    hectares["DairyNZ profit per ha"] = hectares["DairyNZ region"].map(Dairy_profit_per_ha_all_classes_by_region)
    hectares["DairyNZ revenue per ha"] = hectares["DairyNZ region"].map(Dairy_revenue_per_ha_all_classes_by_region)
    pyogrio.write_dataframe(hectares, output_directory+"/hectares_with_stuff_v9.gpkg")

def join_to_lucas_data():
    print("Joining to lucas data")
    lucas = pyogrio.read_dataframe(output_directory + "/lucas_selection.gpkg")
    hectares = pyogrio.read_dataframe(output_directory+"/hectares_with_stuff_v9.gpkg")
    lucas.columns = ["lucas_"+column if column != "geometry" else column for column in lucas.columns]
    hectares["centroid"] = hectares.centroid
    hectares = hectares.set_geometry("centroid")
    hectares = gpd.sjoin(hectares, lucas, how="left")
    hectares = hectares.set_geometry("geometry")
    hectares = hectares.drop(columns=["centroid", "index_right"])
    if(len(hectares) == 0):
        save_empty_dataset(output_directory + "/hectares_with_stuff_final.gpkg")
        sys.exit(0)
    pyogrio.write_dataframe(hectares, output_directory+"/hectares_with_stuff_v10.gpkg")


def load_soil_classification():
    soil_classification_success = load_shapefile_data_to_selection(
        shapefile_path = 'raw_data/lris-fsl-particle-size-classification-SHP/fsl-particle-size-classification.shp',
        selection_path = os.path.join(output_directory, "grid_selection_without_buffer.gpkg"),
        shapefile_crs = "EPSG:2193",
        output_directory = output_directory,
        output_name = "soil_classification_selection"
    )
    if(not soil_classification_success):
        print("Soil classification failed to load")
        save_empty_dataset(output_directory + "/hectares_with_stuff_final.gpkg")
        sys.exit(0)


def load_land_cover_data():
    print("Loading land cover data")
    land_cover_success = load_shapefile_data_to_selection(
        shapefile_path = 'raw_data/lris-lcdb-v30-deprecated-land-cover-database-version-30-mainland--SHP/lcdb-v30-deprecated-land-cover-database-version-30-mainland-.shp',
        selection_path = os.path.join(output_directory, "grid_selection_without_buffer.gpkg"),
        shapefile_crs = "EPSG:2193",
        output_directory = output_directory,
        output_name = "land_cover_selection"
    )
    if(not land_cover_success):
        print("Land cover failed to load")
        save_empty_dataset(output_directory + "/hectares_with_stuff_final.gpkg")
        sys.exit(0)

def run_sediment_calculations():
    """
        Here we calculate the sediment loads for each hectare, under a few different scenarios.
        We want to know what the values are for different land uses.
        We also want to consider different settings for the different land uses.
        To do this we need to first calculate the sediment equation e dataset (see documentation for more details).
        Once we have that we can calculate the sediment loads for each hectare.
        Then we test the different settings for the different scenarios. 
    """
    print("Calculating sediment equation e dataset")
    land_cover_class_to_erosion_terrain_function_group = {
        "Depleted Grassland": "Herbaceous",
        "Herbaceous Freshwater Vegetation": "Herbaceous",
        "Herbaceous Saline Vegetation": "Herbaceous",
        "High Producing Exotic Grassland": "Herbaceous",
        "Low Producing Grassland": "Herbaceous",
        "Short-rotation Cropland": "Herbaceous",
        "Broadleaved Indigenous Hardwoods": "Trees / scrub",
        "Deciduous Hardwoods": "Trees / scrub",
        "Exotic Forest": "Trees / scrub",
        "Fernland": "Trees / scrub",
        "Flaxland": "Trees / scrub",
        "Forest - Harvested": "Trees / scrub",
        "Gorse and/or Broom": "Trees / scrub",
        "Indigenous Forest": "Trees / scrub",
        "Mangrove": "Trees / scrub",
        "Manuka and/or Kanuka": "Trees / scrub",
        "Matagouri or Grey Scrub": "Trees / scrub",
        "Mixed Exotic Shrubland": "Trees / scrub",
        "Orchard Vineyard & Other Perennial Crops": "Trees / scrub",
        "Sub Alpine Shrubland": "Trees / scrub",
        "Alpine Grass/Herbfield": "Tussock and alpine herbaceous",
        "Tall Tussock Grassland": "Tussock and alpine herbaceous",
        "Gravel and Rock": "Other erodible",
        "Landslide": "Other erodible",
        "Permanent Snow and Ice": "Other erodible",
        "Coastal Sand and Gravel": "Other erodible",
        "Surface Mines and Dumps": "Other erodible",
        "Estuarine Open Water": "Water",
        "Lake and Pond": "Water",
        "River": "Water",
        "Transport Infrastructure": "Urban",
        "Built-up Area (settlement)": "Urban",
        "Urban Parkland/Open Space": "Urban",
    }
    # now get the erosion values, from https://environment.govt.nz/assets/Publications/Files/updated-sediment-load-estimator-for-nz.pdf
    erosion_terrain_function_group_to_coefficients = {
        "Herbaceous": 1.0,
        "Trees / scrub": 0.466,
        "Tussock and alpine herbaceous": 0.512,
        "Other erodible": 1.213,
        "Water": 0.0,
        "Urban": 0.0
    }

    # minimal sediment loss under optimal management i.e., herbaceous -> trees
    erosion_terrain_function_group_to_coefficients_minimal = {
        "Herbaceous": 0.466,
        "Trees / scrub": 0.466,
        "Tussock and alpine herbaceous": 0.466,
        "Other erodible": 1.213,
        "Water": 0.0,
        "Urban": 0.0
    }


    # a different set of coefficients for the alternative scenario
    erosion_terrain_function_group_to_coefficients_alt = {
        "Herbaceous": 1.0,
        "Trees / scrub": 0.1,
        "Tussock and alpine herbaceous": 0.512,
        "Other erodible": 1.213,
        "Water": 0.0,
        "Urban": 0.0
    }

    erosion_terrain_function_group_to_coefficients_alt_minimal = {
        "Herbaceous": 0.1,
        "Trees / scrub": 0.1,
        "Tussock and alpine herbaceous": 0.1,
        "Other erodible": 1.213,
        "Water": 0.0,
        "Urban": 0.0
    }

    # constants for the sediment equation
    c_slope = 0.755
    c_rainfall = 1.311
    alpha = 10.039

    # load the data. We need to consider ALL hectares in a grid to recover our e dataset.
    hectares = pyogrio.read_dataframe(output_directory+"/hectares_with_stuff_v10.gpkg")
    hectares["centroid"] = hectares.centroid
    hectares = hectares.set_geometry("centroid")
    all_hectares_in_grid = pyogrio.read_dataframe(output_directory+"/hectares_all.gpkg")

    # mark each hectare_all whether it is in our modelled farm hectares set or not
    all_hectares_in_grid = gpd.sjoin(all_hectares_in_grid, hectares[["centroid"]], how="left")
    all_hectares_in_grid["in_set"] = ~all_hectares_in_grid["index_right"].isna()
    all_hectares_in_grid = all_hectares_in_grid.drop(columns=["index_right"])

    # join all_hectares to the angle data
    original_crs = all_hectares_in_grid.crs
    with rio.open(output_directory+"/angle_data.tif") as src:
        angle_data = src.read(1)
        affine = src.transform
        raster_crs = src.crs
    all_hectares_in_grid = all_hectares_in_grid.to_crs(raster_crs)
    raster_stats = rs.zonal_stats(all_hectares_in_grid,angle_data,affine=affine,stats="mean",nodata=np.nan,add_stats={'angl_avg': raster_helper_nan_mean})
    all_hectares_in_grid["angl_avg"] = [stat["angl_avg"] for stat in raster_stats]
    all_hectares_in_grid = all_hectares_in_grid.to_crs(original_crs)

    # now load in the current land cover data and work out what the land cover group is
    land_cover_data = pyogrio.read_dataframe(output_directory+"/land_cover_selection.gpkg")
    land_cover_data = land_cover_data.to_crs(all_hectares_in_grid.crs)
    land_cover_data["erosion_terrain_function_group"] = land_cover_data["Name_2008"].map(land_cover_class_to_erosion_terrain_function_group)

    # join both all_hectares and hectares to the land cover data
    all_hectares_in_grid["centroid"] = all_hectares_in_grid.centroid
    all_hectares_in_grid = all_hectares_in_grid.set_geometry("centroid")
    all_hectares_in_grid = gpd.sjoin(all_hectares_in_grid, land_cover_data, how="left")
    all_hectares_in_grid = all_hectares_in_grid.drop(columns=["index_right"])
    hectares = gpd.sjoin(hectares, land_cover_data, how="left")
    hectares = hectares.drop(columns=["index_right"])

    # load catchment data to get the sediment yields
    catchment_data = pyogrio.read_dataframe(output_directory+"/MAL_and_catchment_data_processed.gpkg")
    catchment_data = catchment_data.rename(columns=dict(zip(catchment_data.columns, ["catchment_"+column if column != "geometry" else column for column in catchment_data.columns])))

    # join all_hectares to catchment data
    all_hectares_in_grid = gpd.sjoin(all_hectares_in_grid, catchment_data, how="left")
    all_hectares_in_grid = all_hectares_in_grid.drop(columns=["index_right"])

    # fetch the rise as % of run in all_hectares and hectares, excluding values that are too high or too low
    all_hectares_in_grid["rise_as_percentage_of_run"] = np.tan(np.maximum(np.minimum(all_hectares_in_grid["angl_avg"],85),0.1) * np.pi / 180)
    hectares["rise_as_percentage_of_run"] = np.tan(np.maximum(np.minimum(hectares["angl_avg"],85),0.1) * np.pi / 180)

    # fetch the sediment erosion coefficients for each CURRENT land cover type
    all_hectares_in_grid["current_sediment_erosion_land_cover_coefficient"] = all_hectares_in_grid["erosion_terrain_function_group"].map(erosion_terrain_function_group_to_coefficients)
    all_hectares_in_grid["minimal_sediment_erosion_land_cover_coefficient"] = all_hectares_in_grid["erosion_terrain_function_group"].map(erosion_terrain_function_group_to_coefficients_minimal)

    # get the right hand side average for all hectares (sediment loads assuming current land uses, ignoring e)
    all_hectares_in_grid["loads_hectares_no_e"] = alpha * ((all_hectares_in_grid["catchment_rainfall"] / 1000) ** c_rainfall) * (all_hectares_in_grid["rise_as_percentage_of_run"] ** c_slope) * all_hectares_in_grid["current_sediment_erosion_land_cover_coefficient"]
    
    # aggregate up to the catchment level to get the mean loads for each catchment (ignoring the e factor, which we aim to calcualte)
    mean_loads_hectares_no_e_catchment = all_hectares_in_grid.groupby("catchment_nzsegment")[["loads_hectares_no_e"]].mean().fillna(0)
    mean_loads_hectares_no_e_catchment = mean_loads_hectares_no_e_catchment.rename(columns={"loads_hectares_no_e":"mean_loads_hectares_no_e_catchment"})
    
    # join to catchment data, calculate values for e as the ratio of the actual catchment yield to the mean yield for the hectares 
    catchment_data = catchment_data.merge(mean_loads_hectares_no_e_catchment, left_on="catchment_nzsegment", right_index=True, how="left")
    catchment_data["yield_no_e"] = catchment_data["mean_loads_hectares_no_e_catchment"] * 100 # 100 hectares in a km2 (we are getting the mean load per km2). 
    catchment_data["sediment_e_values"] = catchment_data["catchment_CurrentSedYield"] / catchment_data["yield_no_e"] # KEY STEP: reconciling left side (actual catchment yield) with right side (expected hectares in catchment yield).
    catchment_data["sediment_e_values"] = catchment_data["sediment_e_values"].replace([np.inf, -np.inf], np.nan)
    
    # NOW we have the e values, lets run the model for all hectares in their current land use
    all_hectares_in_grid = gpd.sjoin(all_hectares_in_grid, catchment_data[["sediment_e_values", "geometry"]], how="left")
    all_hectares_in_grid = all_hectares_in_grid.drop(columns=["index_right"])
    all_hectares_in_grid["current_sediment_load"] = alpha * ((all_hectares_in_grid["catchment_rainfall"] / 1000) ** c_rainfall) * (all_hectares_in_grid["rise_as_percentage_of_run"] ** c_slope) * all_hectares_in_grid["sediment_e_values"] * all_hectares_in_grid["current_sediment_erosion_land_cover_coefficient"]
    all_hectares_in_grid["min_out_of_model_load"] = alpha * ((all_hectares_in_grid["catchment_rainfall"] / 1000) ** c_rainfall) * (all_hectares_in_grid["rise_as_percentage_of_run"] ** c_slope) * all_hectares_in_grid["sediment_e_values"] * all_hectares_in_grid["minimal_sediment_erosion_land_cover_coefficient"]
    # Now we have the current sediment loads, lets calculate the catchment load not in our model
    load_not_in_model = all_hectares_in_grid[all_hectares_in_grid["in_set"] == False].groupby("catchment_nzsegment")[["current_sediment_load","min_out_of_model_load"]].sum()
    load_not_in_model = load_not_in_model.rename(columns={"current_sediment_load":"sediment_catchment_load_total_not_in_model"})
    catchment_data = catchment_data.merge(load_not_in_model, left_on="catchment_nzsegment", right_index=True, how="left")

    # Finally we look at the actual hectares, runing the sediment formula for the different land cover scenarios and joining to the load not in the model
    hectares = gpd.sjoin(hectares, catchment_data[["sediment_e_values", "sediment_catchment_load_total_not_in_model", "min_out_of_model_load", "geometry"]], how="left")
    hectares = hectares.drop(columns=["index_right"])
    hectares["current_sediment_total_grid_load"] = all_hectares_in_grid["current_sediment_load"].sum()
    sediment_formula_without_land_cover = alpha * ((hectares["catchment_rainfall"] / 1000) ** c_rainfall) * (hectares["rise_as_percentage_of_run"] ** c_slope) * hectares["sediment_e_values"]
    hectares["current_sediment_erosion_land_cover_coefficient"] = hectares["erosion_terrain_function_group"].map(erosion_terrain_function_group_to_coefficients)
    hectares["current_LCDB_sediment_load"] = sediment_formula_without_land_cover * hectares["current_sediment_erosion_land_cover_coefficient"]
    hectares["sediment_load_Herbaceous"] = sediment_formula_without_land_cover * erosion_terrain_function_group_to_coefficients["Herbaceous"]
    hectares["sediment_load_Trees_scrub"] = sediment_formula_without_land_cover * erosion_terrain_function_group_to_coefficients["Trees / scrub"]
    hectares["sediment_load_Tussock_and_alpine_herbaceous"] = sediment_formula_without_land_cover * erosion_terrain_function_group_to_coefficients["Tussock and alpine herbaceous"]
    hectares["sediment_load_Other_erodible"] = sediment_formula_without_land_cover * erosion_terrain_function_group_to_coefficients["Other erodible"]
    hectares["sediment_load_Water"] = sediment_formula_without_land_cover * erosion_terrain_function_group_to_coefficients["Water"]
    hectares["sediment_load_Urban"] = sediment_formula_without_land_cover * erosion_terrain_function_group_to_coefficients["Urban"]

    # now we do the 3 above steps, but for the alternative coefficients. This assumes the e values are the same, and we scale to match the overall load. 
    all_hectares_in_grid["current_sediment_erosion_land_cover_coefficient_alt"] = all_hectares_in_grid["erosion_terrain_function_group"].map(erosion_terrain_function_group_to_coefficients_alt)
    all_hectares_in_grid["minimal_sediment_erosion_land_cover_coefficient_alt"] = all_hectares_in_grid["erosion_terrain_function_group"].map(erosion_terrain_function_group_to_coefficients_alt_minimal)

    all_hectares_in_grid["current_sediment_load_alt"] = alpha * ((all_hectares_in_grid["catchment_rainfall"] / 1000) ** c_rainfall) * (all_hectares_in_grid["rise_as_percentage_of_run"] ** c_slope) * all_hectares_in_grid["sediment_e_values"] * all_hectares_in_grid["current_sediment_erosion_land_cover_coefficient_alt"]
    all_hectares_in_grid["min_out_of_model_load_alt"] = alpha * ((all_hectares_in_grid["catchment_rainfall"] / 1000) ** c_rainfall) * (all_hectares_in_grid["rise_as_percentage_of_run"] ** c_slope) * all_hectares_in_grid["sediment_e_values"] * all_hectares_in_grid["minimal_sediment_erosion_land_cover_coefficient_alt"]
    
    load_not_in_model_alt = all_hectares_in_grid[all_hectares_in_grid["in_set"] == False].groupby("catchment_nzsegment")[["current_sediment_load_alt","min_out_of_model_load_alt"]].sum()
    load_not_in_model_alt = load_not_in_model_alt.rename(columns={"current_sediment_load_alt":"sediment_catchment_load_total_not_in_model_alt"})
    catchment_data = catchment_data.merge(load_not_in_model_alt, left_on="catchment_nzsegment", right_index=True, how="left")
    hectares = gpd.sjoin(hectares, catchment_data[["sediment_catchment_load_total_not_in_model_alt", "min_out_of_model_load_alt", "geometry"]], how="left")
    hectares = hectares.drop(columns=["index_right"])
    hectares["current_sediment_total_grid_load_alt"] = all_hectares_in_grid["current_sediment_load_alt"].sum()
    hectares["current_sediment_erosion_land_cover_coefficient_alt"] = hectares["erosion_terrain_function_group"].map(erosion_terrain_function_group_to_coefficients_alt)
    hectares["current_LCDB_sediment_load_alt"] = sediment_formula_without_land_cover * hectares["current_sediment_erosion_land_cover_coefficient_alt"]
    hectares["sediment_load_Herbaceous_alt"] = sediment_formula_without_land_cover * erosion_terrain_function_group_to_coefficients_alt["Herbaceous"]
    hectares["sediment_load_Trees_scrub_alt"] = sediment_formula_without_land_cover * erosion_terrain_function_group_to_coefficients_alt["Trees / scrub"]
    hectares["sediment_load_Tussock_and_alpine_herbaceous_alt"] = sediment_formula_without_land_cover * erosion_terrain_function_group_to_coefficients_alt["Tussock and alpine herbaceous"]
    hectares["sediment_load_Other_erodible_alt"] = sediment_formula_without_land_cover * erosion_terrain_function_group_to_coefficients_alt["Other erodible"]
    hectares["sediment_load_Water_alt"] = sediment_formula_without_land_cover * erosion_terrain_function_group_to_coefficients_alt["Water"]
    hectares["sediment_load_Urban_alt"] = sediment_formula_without_land_cover * erosion_terrain_function_group_to_coefficients_alt["Urban"]
    hectares = hectares.set_geometry("geometry")
    hectares = hectares.drop(columns=["centroid"])
    pyogrio.write_dataframe(hectares, output_directory+"/hectares_with_stuff_final.gpkg")
    all_hectares_in_grid = all_hectares_in_grid.set_geometry("geometry")
    all_hectares_in_grid = all_hectares_in_grid.drop(columns=["centroid"])
    pyogrio.write_dataframe(all_hectares_in_grid, output_directory+"/all_hectares_in_grid.gpkg")
    pyogrio.write_dataframe(catchment_data, output_directory+"/catchment_data_sed.gpkg")

if __name__ == "__main__":
    load_MAL_data()
    join_to_MAL_data()
    load_dairy_typology()
    join_to_dairy_typology()
    load_height_data()
    join_to_height_data()
    load_lri()
    join_to_lri()
    join_to_regions()
    load_whitiwhiti_ora_economic_indicators()
    join_to_whitiwhiti_ora_economic_indicators()
    load_forestry_data()
    join_to_forestry_data()
    load_pasture_data()
    join_to_pasture_data()
    join_to_dairy_and_BLNZ_profitability()
    join_to_lucas_data()
    load_soil_classification()
    load_land_cover_data()
    run_sediment_calculations()