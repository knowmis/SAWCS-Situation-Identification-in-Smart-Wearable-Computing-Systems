"""Preprocessing_Activity
"""
"""***********************Checking Empty Values For every user*********************"""

import csv # use to remove rows that have empty values

input_file = 'Preprocessing/FDAA70A1-42A3-4E3F-9AE3-3FDA412E03BF.features_labels.csv'
output_file = 'Preproceessing_Step2/updated_FDAA70A1-42A3-4E3F-9AE3-3FDA412E03BF.features_labels.csv'

with open(input_file, 'r') as f_in, open(output_file, 'w', newline='') as f_out:
    reader = csv.reader(f_in)
    writer = csv.writer(f_out)

    header = next(reader)  # get the header row
    writer.writerow(header)  # write the header row to the output file

    for row in reader:
        if any(cell.strip() == '' for cell in row):#
            continue  # skip this row if any cell is empty

        writer.writerow(row)  # write the row to the output file

print(f"Filtered data saved in {output_file}")

"""**********************************Merge All 60 Users*************************************"""

import pandas as pd
import glob

# Define the file pattern to read CSV files
file_pattern = "Preproceessing_Step#02/*.csv"

# Use glob to get a list of all the CSV files matching the pattern
csv_files = glob.glob(file_pattern)

# Create an empty list to hold the dataframes
dfs = []

# Loop through each CSV file, read it into a pandas dataframe, and append it to the dfs list
for csv_file in csv_files:
    df = pd.read_csv(csv_file)
    dfs.append(df)

# Concatenate all the dataframes in the dfs list into a single dataframe
result = pd.concat(dfs, ignore_index=True)

# Write the concatenated dataframe to a new CSV file
result.to_csv("Final_Merge_60_Files.csv", index=False)

"""*******************************Grouping Activities****************************"""
import pandas as pd
# Load the CSV file into a pandas DataFrame
df = pd.read_csv('Final_Merge_60_Files.csv')

# Create a new column that contains the activity label for each sample
df['activity'] = ''
df.loc[df['LYING_DOWN'] == 1, 'activity'] = 'LYING_DOWN'
df.loc[df['SITTING'] == 1, 'activity'] = 'SITTING'
df.loc[df['walking'] == 1, 'activity'] = 'walking'
df.loc[df['running'] == 1, 'activity'] = 'running'
df.loc[df['standing'] == 1, 'activity'] = 'standing'

# Drop the original activity columns
df.drop(['LYING_DOWN', 'SITTING', 'walking', 'running', 'standing'], axis=1, inplace=True)

# Save the modified DataFrame to a new CSV file
df.to_csv('Final_Merge_60_Files_ActivityLabel', index=False)

"""**********************Considering only smartphone and smartwatch********************"""
# use for removal of features only consider 78 features

def remove_columns(input_file, output_file, ranges):
    """
    Removes columns from a CSV file based on given ranges.

    Args:
        input_file (str): Input CSV file name.
        output_file (str): Output CSV file name.
        ranges (list): List of ranges to be removed, each range should be a tuple of start and end column indices.
    """
    with open(input_file, 'r') as infile, open(output_file, 'w', newline='') as outfile:
        reader = csv.reader(infile)
        writer = csv.writer(outfile)
        for row in reader:
            # Remove columns within the given ranges
            new_row = []
            for i in range(len(row)):
                remove = False
                for start, end in ranges:
                    if i >= start and i <= end:
                        remove = True
                        break
                if not remove:
                    new_row.append(row[i])
            writer.writerow(new_row)

# Example usage:
input_file = "InputFiles/FDAA70A1-42A3-4E3F-9AE3-3FDA412E03BF.features_labels.csv"
output_file = 'OutputFiles/FDAA70A1-42A3-4E3F-9AE3-3FDA412E03BF.features_labels.csv'
ranges = [(27,83),(130,225),(230,269),(271,277)]
remove_columns(input_file, output_file, ranges)
print(f"Columns within ranges {ranges} removed from {input_file}. Result saved to {output_file}.")

"""*****************************Standard activity names**********************************"""
#Script use for Label name changing

import os
import pandas as pd

# Define the input and output folder paths
input_folder_path = 'InputFiles_10'
output_folder_path = 'OutputFiles'

# Define the mapping for column names
column_mapping = {
    'label:FIX_walking': 'walking',
    'label:LYING_DOWN': 'LYING_DOWN',
    'label:SITTING': 'SITTING',
    'label:FIX_running': 'running',
    'label:OR_standing': 'standing'
}

# Loop through each file in the input folder
for file_name in os.listdir(input_folder_path):
    if file_name.endswith('.csv'):
        # Read the CSV file into a DataFrame
        input_file_path = os.path.join(input_folder_path, file_name)
        df = pd.read_csv(input_file_path)

        # Rename the columns using the mapping
        df = df.rename(columns=column_mapping)

        # Save the updated DataFrame to a new CSV file in the output folder
        output_file_name = f'updated_{file_name}'
        #print()
        output_file_path = os.path.join(output_folder_path, output_file_name)
        df.to_csv(output_file_path, index=False)
       # print(output_file_name)