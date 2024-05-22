import os
import darshan
import pandas as pd

# log_file = os.path.join(
#     "data", "hdf5_diagonal_write_only", "hdf5_diagonal_write_1_byte_dxt.darshan"
# )

# log_file = os.path.join(
#     "data",
#     "acme_fcase_trace",
#     "snyder_acme.exe_id1253318_9-27-24239-1515303144625770178_2.darshan",
# )

log_file = os.path.join(
    "data",
    "imbalanced_io",
    "imbalanced-io.darshan",
)


def calculate_performance_metrics(
    module_df,
    read_counters,
    write_counters,
    read_time_counters,
    write_time_counters,
    meta_time_counters,
):
    total_bytes_read = 0
    total_bytes_written = 0
    total_read_time = 0
    total_write_time = 0
    total_meta_time = 0

    for counter in read_counters:
        if counter in module_df:
            total_bytes_read += module_df[counter].sum()
            print(f"Reading counter {counter}: {module_df[counter].sum()}")

    for counter in write_counters:
        if counter in module_df:
            total_bytes_written += module_df[counter].sum()
            print(f"Writing counter {counter}: {module_df[counter].sum()}")

    for counter in read_time_counters:
        if counter in module_df:
            total_read_time += module_df[counter].sum()
            print(f"Read time counter {counter}: {module_df[counter].sum()}")

    for counter in write_time_counters:
        if counter in module_df:
            total_write_time += module_df[counter].sum()
            print(f"Write time counter {counter}: {module_df[counter].sum()}")

    for counter in meta_time_counters:
        if counter in module_df:
            total_meta_time += module_df[counter].sum()
            print(f"Metadata time counter {counter}: {module_df[counter].sum()}")

    read_throughput = total_bytes_read / total_read_time if total_read_time > 0 else 0
    write_throughput = (
        total_bytes_written / total_write_time if total_write_time > 0 else 0
    )

    return {
        "total_bytes_read": total_bytes_read,
        "total_bytes_written": total_bytes_written,
        "total_read_time": total_read_time,
        "total_write_time": total_write_time,
        "total_meta_time": total_meta_time,
        "read_throughput": read_throughput,
        "write_throughput": write_throughput,
    }


# Open the Darshan log file and read all data stored in it
with darshan.DarshanReport(log_file, read_all=True) as report:
    # Print the metadata dict for this log
    print("Metadata:", report.metadata)
    # Print job runtime and number of processes
    print("Run time:", report.metadata["job"]["run_time"])
    print("Number of processes (nprocs):", report.metadata["job"]["nprocs"])

    # Print modules contained in the report
    print("Modules:", list(report.modules.keys()))

    performance_metrics = {}

    # POSIX module performance metrics
    if "POSIX" in report.records:
        posix_df = report.records["POSIX"].to_df()
        print("POSIX DataFrame:", posix_df)
        posix_metrics = calculate_performance_metrics(
            posix_df,
            [
                "POSIX_BYTES_READ",
                "POSIX_SIZE_READ_0_100",
                "POSIX_SIZE_READ_100_1K",
                "POSIX_SIZE_READ_1K_10K",
                "POSIX_SIZE_READ_100K_1M",
            ],
            [
                "POSIX_BYTES_WRITTEN",
                "POSIX_SIZE_WRITE_0_100",
                "POSIX_SIZE_WRITE_100_1K",
                "POSIX_SIZE_WRITE_1K_10K",
                "POSIX_SIZE_WRITE_10K_100K",
                "POSIX_SIZE_WRITE_100K_1M",
            ],
            ["POSIX_F_READ_TIME"],
            ["POSIX_F_WRITE_TIME"],
            ["POSIX_F_META_TIME"],
        )
        performance_metrics["POSIX"] = posix_metrics

    # MPI-IO module performance metrics
    if "MPI-IO" in report.records:
        mpiio_df = report.records["MPI-IO"].to_df()
        print("MPI-IO DataFrame:", mpiio_df)
        mpiio_metrics = calculate_performance_metrics(
            mpiio_df,
            ["MPIIO_BYTES_READ"],
            ["MPIIO_BYTES_WRITTEN"],
            ["MPIIO_F_READ_TIME"],
            ["MPIIO_F_WRITE_TIME"],
            ["MPIIO_F_META_TIME"],
        )
        performance_metrics["MPI-IO"] = mpiio_metrics

    # STDIO module performance metrics
    if "STDIO" in report.records:
        stdio_df = report.records["STDIO"].to_df()
        print("STDIO DataFrame:", stdio_df)
        stdio_metrics = calculate_performance_metrics(
            stdio_df,
            ["STDIO_BYTES_READ"],
            ["STDIO_BYTES_WRITTEN"],
            ["STDIO_READ_TIME"],
            ["STDIO_WRITE_TIME"],
            ["STDIO_META_TIME"],
        )
        performance_metrics["STDIO"] = stdio_metrics

    # General handling for all other modules
    for module_name, module_record in report.records.items():
        if module_name not in performance_metrics and module_name not in [
            "POSIX",
            "MPI-IO",
            "STDIO",
        ]:
            module_df = module_record.to_df()
            if isinstance(module_df, pd.DataFrame):
                read_counters = [col for col in module_df.columns if "READ" in col]
                write_counters = [col for col in module_df.columns if "WRITE" in col]
                read_time_counters = [
                    col for col in module_df.columns if "F_READ_TIME" in col
                ]
                write_time_counters = [
                    col for col in module_df.columns if "F_WRITE_TIME" in col
                ]
                meta_time_counters = [
                    col for col in module_df.columns if "F_META_TIME" in col
                ]

                module_metrics = calculate_performance_metrics(
                    module_df,
                    read_counters,
                    write_counters,
                    read_time_counters,
                    write_time_counters,
                    meta_time_counters,
                )
                performance_metrics[module_name] = module_metrics

    # Print performance metrics
    for module, metrics in performance_metrics.items():
        print(f"{module} Module Performance Metrics:")
        print(f"Total bytes read: {metrics['total_bytes_read']}")
        print(f"Total bytes written: {metrics['total_bytes_written']}")
        print(f"Total read time (seconds): {metrics['total_read_time']}")
        print(f"Total write time (seconds): {metrics['total_write_time']}")
        print(f"Total metadata time (seconds): {metrics['total_meta_time']}")
        print(f"Read throughput (bytes/second): {metrics['read_throughput']}")
        print(f"Write throughput (bytes/second): {metrics['write_throughput']}")
