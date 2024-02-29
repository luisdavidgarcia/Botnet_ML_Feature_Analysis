import pandas as pd
import os

def split_csv_file(file_path, chunk_size):
    # Determine the base name for output files
    base_name = os.path.splitext(file_path)[0]
    
    # Initialize a chunk counter
    chunk_counter = 1
    
    # Read the CSV file in chunks
    for chunk in pd.read_csv(file_path, chunksize=chunk_size):
        # Define the output file name based on the base name and chunk counter
        output_file = f"splitfile_part_{chunk_counter}.csv"
        
        # Write the chunk to a new CSV file
        chunk.to_csv(output_file, index=False)
        
        # Check the file size, if it's over 25 MB, you may need to adjust chunk_size
        if os.path.getsize(output_file) > 25 * 1024 * 1024:
            print(f"Warning: {output_file} is larger than 25 MB")
        
        # Increment the chunk counter
        chunk_counter += 1

# Replace 'your_file.csv' with the path to your large CSV file
file_path = '02-15-2018_1.csv'
# Estimate the chunk size (number of rows per chunk)
chunk_size = 50000  # You might need to adjust this based on your data

split_csv_file(file_path, chunk_size)
