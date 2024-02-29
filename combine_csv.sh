# Grab the header from the first file and create the combined file
head -n 1 splitfile_part_1.csv > combined.csv

# Concatenate the rest of the files, excluding their headers
tail -n +2 -q splitfile_part_*.csv >> combined.csv
