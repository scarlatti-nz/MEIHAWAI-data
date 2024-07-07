# Snakefile
import numpy as np
np.random.seed(12908)
import math

output_directory = config["output_dir"]

lat_grid_size = math.ceil((config["lat_end"] - config["lat_start"]) / config["grid_size"])
lon_grid_size = math.ceil((config["lon_end"] - config["lon_start"]) / config["grid_size"])
lat_values = [str(round(config["lat_start"] + x * config["grid_size"], 4)) for x in np.arange(0, lat_grid_size)]
lon_values = [str(round(config["lon_start"] + x * config["grid_size"], 4)) for x in np.arange(0, lon_grid_size)]

# make grid of all options
lat_lon_strings = [lat + "_" + lon for lat in lat_values for lon in lon_values]
lat_lon_strings = np.random.choice(lat_lon_strings, min(len(lat_lon_strings), config["grid_max_samples"]), replace=False)

rule all:
    input:
        expand(
            output_directory+"/grid_{lat_lon}/hectares_with_stuff_final.gpkg",
            lat_lon=lat_lon_strings
        )
    shell:
        "rm -rf logs"

rule create_farms_in_grid:
    input: "create_farms_in_grid.py"
    output: output_directory+"/grid_{lat_lon}/farms.gpkg"
    shell: "python create_farms_in_grid.py {wildcards.lat_lon}"

rule aggregate_create_farms:
    input: 
        expand(output_directory+"/grid_{lat_lon}/farms.gpkg", lat_lon=lat_lon_strings)
    output: output_directory+"/create_farms_complete.txt"
    shell:
        """
        python -c "open('{output}', 'a').close()"
        """

rule align_farms_between_grids:
    input: 
        "align_farms_between_grids.py", 
        output_directory+"/create_farms_complete.txt"
    output: output_directory+"/grid_{lat_lon}/farms_aligned.gpkg"
    shell: "python align_farms_between_grids.py {wildcards.lat_lon}"

rule split_farms_into_hectares:
    input: 
        "split_farms_into_hectares.py",
        output_directory+"/grid_{lat_lon}/farms_aligned.gpkg"
    output: output_directory+"/grid_{lat_lon}/hectares.gpkg"
    shell: "python split_farms_into_hectares.py {wildcards.lat_lon}"

rule join_hectares_to_stuff:
    input: 
        "join_hectares_to_stuff.py",
        output_directory+"/grid_{lat_lon}/hectares.gpkg"
    output: output_directory+"/grid_{lat_lon}/hectares_with_stuff_final.gpkg"
    shell: "python join_hectares_to_stuff.py {wildcards.lat_lon}"
