import darshan
import pandas as pd
import os
from glob import glob

darshan_file = os.path.join(
    "data", "hdf5_diagonal_write_only", "hdf5_diagonal_write_1_byte_dxt.darshan"
)

# open a Darshan log file and read all data stored in it
with darshan.DarshanReport(darshan_file, read_all=True) as report:

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
