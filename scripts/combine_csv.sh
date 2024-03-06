FILE1="02-15-2018.csv"
FILE2="02-16-2018.csv"
FILE3="02-21-2018.csv"

# Grab the header from the first file and create the combined file
head -n 1 $FILE1 > combined.csv

# Concatenate the rest of the files, excluding their headers
tail -n +2 -q $FILE2 $FILE3 >> combined.csv
