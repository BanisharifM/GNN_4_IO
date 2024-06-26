import os
import pandas as pd
import networkx as nx

# Load the CSV file
file_path = "CSVs/sample_train_100.csv"
df = pd.read_csv(file_path)

# Define categories and their corresponding counters
categories = {
    "POSIX_ACCESS": [
        "POSIX_ACCESS1_ACCESS",
        "POSIX_ACCESS2_ACCESS",
        "POSIX_ACCESS3_ACCESS",
        "POSIX_ACCESS4_ACCESS",
        "POSIX_ACCESS1_COUNT",
        "POSIX_ACCESS2_COUNT",
        "POSIX_ACCESS3_COUNT",
        "POSIX_ACCESS4_COUNT",
    ],
    "POSIX_STRIDE": [
        "POSIX_STRIDE1_STRIDE",
        "POSIX_STRIDE2_STRIDE",
        "POSIX_STRIDE3_STRIDE",
        "POSIX_STRIDE4_STRIDE",
        "POSIX_STRIDE1_COUNT",
        "POSIX_STRIDE2_COUNT",
        "POSIX_STRIDE3_COUNT",
        "POSIX_STRIDE4_COUNT",
    ],
    "POSIX_SIZE": [
        "POSIX_SIZE_READ_0_100",
        "POSIX_SIZE_READ_100_1K",
        "POSIX_SIZE_READ_1K_10K",
        "POSIX_SIZE_READ_100K_1M",
        "POSIX_SIZE_WRITE_0_100",
        "POSIX_SIZE_WRITE_100_1K",
        "POSIX_SIZE_WRITE_1K_10K",
        "POSIX_SIZE_WRITE_10K_100K",
        "POSIX_SIZE_WRITE_100K_1M",
    ],
    "POSIX_OPERATIONS": [
        "POSIX_OPENS",
        "POSIX_FILENOS",
        "POSIX_READS",
        "POSIX_WRITES",
        "POSIX_SEEKS",
        "POSIX_STATS",
    ],
    "POSIX_DATA_TRANSFERS": ["POSIX_BYTES_READ", "POSIX_BYTES_WRITTEN"],
    "POSIX_PATTERNS": [
        "POSIX_CONSEC_READS",
        "POSIX_CONSEC_WRITES",
        "POSIX_SEQ_READS",
        "POSIX_SEQ_WRITES",
    ],
    "POSIX_ALIGNMENTS": [
        "POSIX_MEM_ALIGNMENT",
        "POSIX_FILE_ALIGNMENT",
        "POSIX_MEM_NOT_ALIGNED",
        "POSIX_FILE_NOT_ALIGNED",
    ],
    "POSIX_RESOURCE_UTILIZATION": ["POSIX_RW_SWITCHES"],
    "LUSTRE_CONFIGURATION": ["LUSTRE_STRIPE_SIZE", "LUSTRE_STRIPE_WIDTH"],
    "GENERAL": ["nprocs"],
}

# Directory to save the graph representations
output_dir = "Graph3"
os.makedirs(output_dir, exist_ok=True)
print(f"Output directory created at: {output_dir}")

# Initialize the Excel writer
excel_file_path = f"{output_dir}/graphs.xlsx"
writer = pd.ExcelWriter(excel_file_path, engine="xlsxwriter")


# Function to adjust the column widths
def adjust_column_widths(writer, sheet_name, df):
    worksheet = writer.sheets[sheet_name]
    for i, col in enumerate(df.columns):
        max_len = df[col].astype(str).map(len).max()
        worksheet.set_column(i, i, max_len + 2)


# Function to create and save graph for each row
def create_and_save_graph(row, index):
    G = nx.DiGraph()
    root_node = "Tag"
    G.add_node((root_node, row["tag"]))

    for category, counters in categories.items():
        G.add_node((category, ""))
        G.add_edge((root_node, row["tag"]), (category, ""))
        for counter in counters:
            if counter in row:
                counter_node = (counter, row[counter])
                G.add_node(counter_node)
                G.add_edge((category, ""), counter_node)

    # Save the graph to a DataFrame
    edges = list(G.edges(data=True))
    edges_df = pd.DataFrame(
        [(source[0], source[1], target[0], target[1]) for source, target, _ in edges],
        columns=["Source", "Source_Value", "Target", "Target_Value"],
    )
    sheet_name = f"Graph_{index}"
    edges_df.to_excel(writer, sheet_name=sheet_name, index=False)
    adjust_column_widths(writer, sheet_name, edges_df)
    print(f"Graph {index} saved to sheet {sheet_name}")


# Create and save graphs for the first 10 rows
for index, row in df.iterrows():
    if index < 10:
        create_and_save_graph(row, index)
    else:
        break

# Save the Excel file
writer.close()
print(f"Graphs saved to Excel file at: {excel_file_path}")
