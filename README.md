# Graph4IO: Heterogeneous Graph Neural Networks for I/O Performance Bottleneck Diagnosis

![Graph4IO](https://github.com/user-attachments/assets/aa194384-24b2-4fe0-a373-cc89076a750c)


## Overview

Graph4IO is a project developed as part of Google Summer of Code (GSoC) 2024. The goal of this project is to enhance the AIIO framework by integrating Graph Neural Networks (GNNs) to improve the diagnosis of I/O performance bottlenecks in High-Performance Computing (HPC) systems. The project focuses on developing a data pre-processing pipeline, constructing a GNN model tailored for I/O performance analysis, and evaluating the model's accuracy and interpretability.

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgements](#acknowledgements)

## Introduction

High-Performance Computing (HPC) applications often suffer from I/O performance bottlenecks. Identifying these bottlenecks is crucial for optimizing I/O operations and improving cost efficiency. Graph4IO utilizes the power of Graph Neural Networks to model complex relationships in I/O operations, providing a more detailed and accurate diagnosis of I/O performance issues.

## Features

- **Data Pre-Processing Pipeline**: Convert Darshan I/O log files into graph structures suitable for GNN processing.
- **Graph Neural Network Model**: Train and evaluate a GNN model specifically designed to analyze I/O performance.
- **SHAP Value Integration**: Use SHAP values to interpret model predictions and understand the contribution of each node and feature.
- **Performance Evaluation**: Test and validate the model using AIIO's test cases.

## Installation

### Prerequisites

- **Python 3.8+**
- **PyTorch**
- **PyTorch Geometric**
- **Pandas**
- **Scipy**
- **Docker** (optional, for a containerized environment)

### Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/BanisharifM/Graph4IO.git
   cd Graph4IO
   ```
2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
3.  **(Optional) To run the project in a Docker container**:
   ```bash
   docker-compose up
   ```

## Usage

### 1. Converting Darshan Log Files to CSV

Use the `convert_to_csv.py` script to convert Darshan log files to CSV format:
```bash
python convert_to_csv.py --input-dir /path/to/darshan/logs --output-dir /path/to/output/csvs
```

### 2. Generating Graph Structures

Generate graph structures from the converted CSV files:
```bash
python generate_graph.py --input-csv /path/to/csv --output-graph /path/to/output/graphs
```

### 3. Training the GNN Model

Train the GNN model using the prepared graph data:
```bash
python train_gnn.py --config /path/to/config.yaml   
```


## Project Structure

- `src/`: Contains the source code for data processing, graph generation, and GNN training.
- `configs/`: Configuration files for different stages of the project.
- `notebooks/`: Jupyter notebooks for exploratory data analysis and experiments.
- `CSVs/`: Directory containing CSV files generated from Darshan logs.
- `graphs/`: Directory containing graph structures ready for GNN input.
- `models/`: Trained GNN models and their checkpoints.
- `images/`: Images and visualizations used in the project.

## Results

### Performance Metrics

- **Train Loss**: 0.5751
- **Train RMSE**: 0.7927
- **Test RMSE**: 0.7907
- **Train MAE**: 0.5640
- **Test MAE**: 0.5646
- **Train R²**: 0.3395
- **Test R²**: 0.3400

![loss_chart](https://github.com/user-attachments/assets/9d85a25d-125f-4334-9dd1-b39c9e911a56)
![rmse_chart](https://github.com/user-attachments/assets/3ceb9fc1-89b4-44c9-af06-fb01483ba7ef)
![mae_chart](https://github.com/user-attachments/assets/874b2b3f-4b6c-4c2a-8eaa-e0b40fa32a7a)
![r2_chart](https://github.com/user-attachments/assets/6386e975-4943-468f-9ddd-86cbed305a33)

## Contributors

<div style="display: flex; align-items: center; justify-content: center; flex-wrap: wrap; gap: 20px;">

<a href="https://github.com/banisharifm" style="text-decoration: none; color: initial; text-align: center;">
    <img src="https://avatars.githubusercontent.com/u/41099498?v=4" width="100" style="border-radius: 50%; padding-top: 10px;" alt="BanisharifM"/>
    <br />
    <span style="font-size: 16px; font-weight: bold; color: #000;">Mahdi Banisharif</span>
    <br />
    <span style="font-size: 14px; color: #555;">PhD student in Computer Science<br/>Iowa State University</span>
</a>

</div>


## License

This project is licensed under the MIT License.

## Acknowledgements

This project was developed as part of the Google Summer of Code (GSoC) 2024.

Special thanks to my supervisor [Ali Jannesari](https://www.cs.iastate.edu/jannesar) at Iowa State University for his continuous support and guidance throughout this project.

Additional thanks to the mentors [Bin Dong](https://github.com/BinDong314) and [Suren Byna](https://github.com/sbyna), as well as the open-source community for their invaluable contributions.


