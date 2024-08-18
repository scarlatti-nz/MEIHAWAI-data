# MEIHAWAI Farm Data Model

## Overview
This repository creates a hectare-level dataset describing New Zealand farms as an input to the [MEIHAWAI](https://github.com/scarlatti-nz/MEIHAWAI) model. Due to the large computing power needed to do this, we structure it to run in parallel on small grids (measuring 0.2 by 0.2 degrees). The *snakemake* python package is used to manage the workflow and help with running on a cluster computer.
The dataset is created in 4 key stages:

1. **Generation of farms**: Within each grid land parcels are grouped into farms based on ownership information and the land use.

2. **Removal of overlaps**: Given the grid level nature of the data, some farms may be split between grids. This stage removes these overlaps.

3. **Splitting into hectares**: Each farm is split into hectares.

4. **Data merging**: Hectares are joined to a variety of relevant datasets.


This project is distributed under the Creative Commons Attribution-ShareAlike 4.0 International License. See LICENSE.md for details.

## Input data
Due to filesize and license restrictions, the raw input data is not included in this repository, however, this can be recreated by downloading these datasets from source.
To run this code, create a folder called `raw_data` and populate the following directory structure with the relevant datasets:
### Directory Structure

```
- mfe-updated-suspended-sediment-yield-estimator-and-estuarine-tra-SHP
- lris-fsl-particle-size-classification-SHP
- whitiwhiti-ora
    - dairy-typology
    - economic-indicators
    - forestry
    - MAL and catchment data
    - N_and_P
    - pasture
    - sediment
- lris-lcdb-v30-deprecated-land-cover-database-version-30-mainland--SHP
- rainfall
- mfe-lucas-nz-land-use-map-1990-2008-2012-2016-v011-SHP
- lds-nz-property-titles-including-owners-SHP_oct2021
- regional_councils
- lris-nzdem-south-island-25-metre-GTiff
- lris-nzdem-north-island-25-metre-GTiff
- lris-nzlri-land-use-capability-2021-SHP
```


## Project Structure
The code files are:

- `create_farms_in_grid.py`: This script generates farm boundaries within a specified grid.
  
- `align_farms_between_grids.py`: This script makes sure that a farm overlapping between two grids is asigned the same id in each.  

- `split_farms_into_hectares.py`: This script splits farm boundaries into hectare-sized parcels.

- `join_hectares_to_data.py`: This script merges hectare-level data with other relevant datasets.

- `utils.py`: This script contains utility functions used by the main scripts.

- `get_pasture_means.py`: This script calculates the mean of pasture yield across NZ, split by farm type (S&B and Dairy).

In addition, the project includes the following configuration / workflow files:

- `Snakefile`: This file defines the workflow for running the project using Snakemake.

- `config.yaml`: This YAML file contains configuration settings for the project, such as grid size, buffer size, minimum land parcel size, and minimum farm size.

- `cluster-nesi-mahuika.yaml`: This YAML file contains configuration settings for running the project on the NeSI Mahuika cluster.

- `aliases.sh`: This file contains aliases for running the project and moving data to the NeSI Mahuika cluster.

- `environment.yaml`: This file contains the conda environment for running the project. To activate it use `conda env create -f environment.yaml`


## Explanation of scripts

### create_farms_in_grid.py

Example usage: `python create_farms_in_grid.py -37.9_175.5`

This script builds the farm dataset for a single grid. It does the following:

1. Define the grid cell given, with and without the buffer. The coordinates specified in the argument are the bottom left corner of the grid. The grid size is defined in the config file. 

2. Load LINZ data (property titles including owners) and LUCAS data (land use map).

3. Filter out parcels which are too small, or don't contain enough farmland (LUCAS data is used to identify farmland).

4. Go through each unique land owner and merge parcels where the owners are the same. This creates the farm boundaries.

5. Remove overlapping farms.

### align_farms_between_grids.py

Example usage: `python align_farms_between_grids.py -37.9_175.5`

This script aligns farm ids between neighboring grids to ensure that the same farm is assigned the same id in each grid. It does the following:

1. Load farm data for the current grid.

2. Identify and load farm data for neighboring grids.

3. For each farm check if it overlaps with another in a neighboring grid.

4. If it does, do a hash of the two farms and if the farm in the neighboring grid has a higher hash, set the current farm id to the neighboring farm id.

### split_farms_into_hectares.py

Example usage: `python split_farms_into_hectares.py -37.9_175.5`

This script splits farm boundaries into hectare-sized parcels. It does the following:

1. Load the grid without the buffer.

2. Divide the grid into 1 hectare parcels.

3. Save the full grid and the grid where it intersects with the farm boundaries to two seperate files.

### join_hectares_to_data.py

Example usage: `python join_hectares_to_data.py -37.9_175.5`

This script merges hectare-level data with other relevant datasets. It does the following:

1. Combine with maximum allowable load (MAL) data for sediment, N, and P.

2. Combine with dairy typology data.

3. Combine with height rasters to get the average slope of each hectare.

4. Combine with land resource inventory (LRI) data.

5. Combine with regional map information.

6. Combine with Whitiwhiti Ora economic indicator rasters.

7. Combine with forestry return rasters.

8. Combine with pasture yield information to get how each hectare sits relative to the distribution of pasture yields (S&B and Dairy seperated).

9. Combine with BLNZ and Dairy profitability data.

10. Combine with Lucas data.

11. Combine with soil classification data.

12. Combine with land cover data.

13. Run sediment calculations to work out the sediment load of each hectare under different scenarios. 

### get_pasture_means.py

Example usage: `python get_pasture_means.py`

This script calculates the mean pasture yield across NZ. It does this split by the (current) farm type.

## Usage

You can run a single grid by just running the python scripts as described above. However, to run the entire workflow, you can use Snakemake.
In the snakemake file we first load in all grids according to the configuration file. 
We specify that each script just needs the previous script to be completed before it can run.
The only exception is the `align_farms_between_grids.py` script which needs all the `create_farms_in_grid.py` scripts to be completed before it can run.
We add a dummy rule to the snakemake file to ensure that all farms in grids are completed before doing alignment. 

After installing snakemake you would run the entire workflow with something like:
`snakemake --jobs 400 --printshellcmds --rerun-incomplete --configfile config.yaml --cluster-config cluster-nesi-mahuika.yaml --cluster "sbatch --account={cluster.account} --partition={cluster.partition} --mem={cluster.mem} --ntasks={cluster.ntasks} --cpus-per-task={cluster.cpus-per-task} --time={cluster.time} --hint={cluster.hint} --output={cluster.output} --error={cluster.error}"`

Here --jobs is the number of seperate processes to be run.

## Contact

Please contact kenny.bell@scarlatti.co.nz with any questions or issues.
