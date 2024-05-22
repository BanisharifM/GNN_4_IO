import os
import pandas as pd

# Load the CSV file
file_path = os.path.join("CSVs", "sample_train_100.csv")
df = pd.read_csv(file_path)


# Function to calculate all performance metrics
def calculate_all_metrics(df):
    metrics = {}
    metrics["total_bytes_read"] = df["POSIX_BYTES_READ"].sum()
    metrics["total_bytes_written"] = df["POSIX_BYTES_WRITTEN"].sum()
    metrics["total_read_operations"] = df["POSIX_READS"].sum()
    metrics["total_write_operations"] = df["POSIX_WRITES"].sum()
    metrics["total_seek_operations"] = df["POSIX_SEEKS"].sum()
    metrics["total_stat_operations"] = df["POSIX_STATS"].sum()
    metrics["total_consecutive_reads"] = df["POSIX_CONSEC_READS"].sum()
    metrics["total_consecutive_writes"] = df["POSIX_CONSEC_WRITES"].sum()
    metrics["total_sequential_reads"] = df["POSIX_SEQ_READS"].sum()
    metrics["total_sequential_writes"] = df["POSIX_SEQ_WRITES"].sum()
    metrics["total_rw_switches"] = df["POSIX_RW_SWITCHES"].sum()
    metrics["total_mem_not_aligned"] = df["POSIX_MEM_NOT_ALIGNED"].sum()
    metrics["total_file_not_aligned"] = df["POSIX_FILE_NOT_ALIGNED"].sum()
    metrics["total_reads_size_0_100"] = df["POSIX_SIZE_READ_0_100"].sum()
    metrics["total_reads_size_100_1K"] = df["POSIX_SIZE_READ_100_1K"].sum()
    metrics["total_reads_size_1K_10K"] = df["POSIX_SIZE_READ_1K_10K"].sum()
    metrics["total_reads_size_100K_1M"] = df["POSIX_SIZE_READ_100K_1M"].sum()
    metrics["total_writes_size_0_100"] = df["POSIX_SIZE_WRITE_0_100"].sum()
    metrics["total_writes_size_100_1K"] = df["POSIX_SIZE_WRITE_100_1K"].sum()
    metrics["total_writes_size_1K_10K"] = df["POSIX_SIZE_WRITE_1K_10K"].sum()
    metrics["total_writes_size_10K_100K"] = df["POSIX_SIZE_WRITE_10K_100K"].sum()
    metrics["total_writes_size_100K_1M"] = df["POSIX_SIZE_WRITE_100K_1M"].sum()
    metrics["total_stride1"] = df["POSIX_STRIDE1_STRIDE"].sum()
    metrics["total_stride2"] = df["POSIX_STRIDE2_STRIDE"].sum()
    metrics["total_stride3"] = df["POSIX_STRIDE3_STRIDE"].sum()
    metrics["total_stride4"] = df["POSIX_STRIDE4_STRIDE"].sum()
    metrics["total_stride1_count"] = df["POSIX_STRIDE1_COUNT"].sum()
    metrics["total_stride2_count"] = df["POSIX_STRIDE2_COUNT"].sum()
    metrics["total_stride3_count"] = df["POSIX_STRIDE3_COUNT"].sum()
    metrics["total_stride4_count"] = df["POSIX_STRIDE4_COUNT"].sum()
    metrics["total_access1"] = df["POSIX_ACCESS1_ACCESS"].sum()
    metrics["total_access2"] = df["POSIX_ACCESS2_ACCESS"].sum()
    metrics["total_access3"] = df["POSIX_ACCESS3_ACCESS"].sum()
    metrics["total_access4"] = df["POSIX_ACCESS4_ACCESS"].sum()
    metrics["total_access1_count"] = df["POSIX_ACCESS1_COUNT"].sum()
    metrics["total_access2_count"] = df["POSIX_ACCESS2_COUNT"].sum()
    metrics["total_access3_count"] = df["POSIX_ACCESS3_COUNT"].sum()
    metrics["total_access4_count"] = df["POSIX_ACCESS4_COUNT"].sum()

    # Calculating throughput
    metrics["read_throughput"] = (
        metrics["total_bytes_read"] / metrics["total_read_operations"]
        if metrics["total_read_operations"] > 0
        else 0
    )
    metrics["write_throughput"] = (
        metrics["total_bytes_written"] / metrics["total_write_operations"]
        if metrics["total_write_operations"] > 0
        else 0
    )

    return metrics


# Function to analyze and interpret performance metrics
def analyze_performance(metrics):
    analysis = {}

    # Identifying Bottlenecks
    if metrics["total_rw_switches"] > 100:
        analysis["rw_switches"] = (
            "High number of read/write switches detected. This could indicate inefficient I/O patterns."
        )

    if metrics["total_mem_not_aligned"] > 100:
        analysis["mem_alignment"] = (
            "High number of memory not aligned operations detected. Consider aligning memory for better performance."
        )

    if metrics["total_file_not_aligned"] > 100:
        analysis["file_alignment"] = (
            "High number of file not aligned operations detected. Consider aligning file accesses for better performance."
        )

    # Understanding Access Patterns
    if metrics["total_sequential_reads"] > metrics["total_consecutive_reads"]:
        analysis["read_pattern"] = (
            "More sequential reads than consecutive reads detected. Consider optimizing read patterns."
        )

    if metrics["total_sequential_writes"] > metrics["total_consecutive_writes"]:
        analysis["write_pattern"] = (
            "More sequential writes than consecutive writes detected. Consider optimizing write patterns."
        )

    # Optimizing I/O
    if metrics["read_throughput"] < 1.0:
        analysis["read_throughput"] = (
            "Low read throughput detected. Consider optimizing read operations."
        )

    if metrics["write_throughput"] < 1.0:
        analysis["write_throughput"] = (
            "Low write throughput detected. Consider optimizing write operations."
        )

    # Resource Utilization
    total_operations = (
        metrics["total_read_operations"]
        + metrics["total_write_operations"]
        + metrics["total_seek_operations"]
        + metrics["total_stat_operations"]
    )
    if total_operations > 1000:
        analysis["resource_utilization"] = (
            "High number of I/O operations detected. Consider optimizing resource utilization."
        )

    return analysis


# Function to print the metrics and analysis line by line
def print_metrics_and_analysis(metrics, analysis):
    print("Performance Metrics:")
    for key, value in metrics.items():
        print(f"{key}: {value}")

    print("\nPerformance Analysis and Recommendations:")
    for key, value in analysis.items():
        print(f"{key}: {value}")


# Get all performance metrics
all_metrics = calculate_all_metrics(df)

# Analyze the performance metrics
performance_analysis = analyze_performance(all_metrics)

# Print the metrics and analysis
print_metrics_and_analysis(all_metrics, performance_analysis)
