#!/bin/bash
FILE1="data/02-15-2018.csv"
FILE2="data/02-16-2018.csv"
FILE3="data/02-21-2018.csv"

# Merge the files together
cat $FILE1 $FILE2 $FILE3 > data/combined.csv
# Remove duplicates
awk 'NR == 1 || !seen[$0]++' data/combined.csv > data/cleaned_combined.csv
awk 'NR == 1 {print; next} !(seen[$0]++ && NR == 1046156)' temp.csv > data/cleaned_combined.csv
rm data/combined.csv temp.csv
