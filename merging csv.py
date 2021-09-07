# import csv, os
#
#
# base_dir = r'C:\Users\Ramstein\Downloads\B. Disease Grading\2. Groundtruths'
# # list_of_files = [file for file in glob('*.{}'.format(file_pattern))]
# file2 =  'b. IDRiD_Disease Grading_Testing Labels - Copy.csv'
#
# r = csv.reader(open(os.path.join(base_dir, file2))) # Here your csv file
#
# lines = list(r)
#
# writer = csv.writer(open(os.path.join(base_dir, 'output.csv'), 'w', newline=''))
#
# for i in range(len(lines)-1):
#     lines[i+1][0] = 'IDRiD_{}'.format(414+i)
#
# writer.writerows(lines)





"""
 Python Script:
  Combine/Merge multiple CSV files using the Pandas library
"""
from os import chdir
from glob import glob
import pandas as pdlib
import csv, os

# Produce a single CSV after combining all files
def produceOneCSV(list_of_files, file_out):
   # Consolidate all CSV files into one object
   result_obj = pdlib.concat([pdlib.read_csv(file) for file in list_of_files])
   # Convert the above object into a csv file and export
   result_obj.to_csv(file_out, index=False, encoding="utf-8")

# Move to the path that holds our CSV files
csv_file_path = r'C:\Users\Ramstein\Downloads\B. Disease Grading\2. Groundtruths'
chdir(csv_file_path)

# List all CSV files in the working dir

file_pattern = ".csv"
base_dir = r'C:\Users\Ramstein\Downloads\B. Disease Grading\2. Groundtruths'
# list_of_files = [file for file in glob('*.{}'.format(file_pattern))]
file1 = 'a. IDRiD_Disease Grading_Training Labels.csv'
file2 =  'output.csv'


file_out = "AppendedOutput.csv"
produceOneCSV([file1, file2], file_out)