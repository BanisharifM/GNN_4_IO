import darshan
import pandas as pd
import os

# Path to the .darshan log file
log_file_path = "data/hdf5_diagonal_write_1_byte_dxt.darshan"


# Path to the directory where the CSV files will be saved
output_dir = "CSVs/"
os.makedirs(output_dir, exist_ok=True)

# open a Darshan log file and read all data stored in it
with darshan.DarshanReport(log_file_path, read_all=True) as report:

    # print the metadata dict for this log
    print("metadata: ", report.metadata)
    # print job runtime and nprocs
    print("run_time: ", report.metadata["job"]["run_time"])
    print("nprocs: ", report.metadata["job"]["nprocs"])

    # print modules contained in the report
    print("modules: ", list(report.modules.keys()))

    # export POSIX module records to DataFrame and print
    posix_df = report.records["POSIX"].to_df()
    print("POSIX df: ", posix_df)

    print(type(posix_df))
    print(posix_df.keys())


# Open the Darshan log file
with darshan.DarshanReport(log_file_path, read_all=True) as report:
    # Export POSIX module records to a pandas DataFrame
    posix_df = report.records["POSIX"].to_df()

    # Export the DataFrame to a CSV file
    # posix_df.to_csv("output.csv", index=False)
    posix_df["counters"].to_csv(os.path.join(output_dir, "output1.csv"), index=False)
    posix_df["fcounters"].to_csv(os.path.join(output_dir, "output2.csv"), index=False)
