import darshan
import pandas as pd
import os
from glob import glob

# Path to the directory containing .darshan log files
data_dir = "data"

# Path to the directory where the CSV files will be saved
output_dir = "CSVs/"
os.makedirs(output_dir, exist_ok=True)

# Recursively find all .darshan files in the data directory
darshan_files = glob(os.path.join(data_dir, "**/*.darshan"), recursive=True)

for darshan_file in darshan_files:
    # Create the output directory structure in CSVs directory
    output_subdir = os.path.relpath(os.path.dirname(darshan_file), data_dir)
    os.makedirs(os.path.join(output_dir, output_subdir), exist_ok=True)

    # Open a Darshan log file and read all data stored in it
    with darshan.DarshanReport(darshan_file, read_all=True) as report:
        # Export POSIX module records to a pandas DataFrame
        posix_df = report.records["POSIX"].to_df()

        # Export the DataFrame to a CSV file
        output_csv_path_counters = os.path.join(
            output_dir,
            output_subdir,
            os.path.basename(darshan_file).replace(".darshan", "_counters.csv"),
        )

        output_csv_path_fcounters = os.path.join(
            output_dir,
            output_subdir,
            os.path.basename(darshan_file).replace(".darshan", "_fcounters.csv"),
        )

        posix_df["counters"].to_csv(output_csv_path_counters, index=False)
        posix_df["fcounters"].to_csv(output_csv_path_fcounters, index=False)
