alias connect_to_nesi='ssh -Y mahuika'
alias upload_folder='rsync -rcvP --info=progress2 --exclude="output/*" /mnt/c/Users/GiottoFrean/Desktop/farm_data/* mahuika:/nesi/project/nesi03886/projects/farm_data/'
alias download_folder='rsync -rcvP --info=progress2 --include="*/" --include="hectares_with_stuff_final.*" --exclude="*" mahuika:/nesi/nobackup/nesi03886/farm_data2/* /mnt/c/Users/GiottoFrean/Desktop/farm_data/output/'
