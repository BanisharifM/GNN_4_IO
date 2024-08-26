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

- **Train Accuracy**: 50.41%
- **Test Accuracy**: 51.31%
- **Train RMSE**: 0.9509
- **Test RMSE**: 0.9429

![loss_chart](https://github.com/user-attachments/assets/9d85a25d-125f-4334-9dd1-b39c9e911a56)
![mae_chart](https://github.com/user-attachments/assets/874b2b3f-4b6c-4c2a-8eaa-e0b40fa32a7a)
![rmse_chart](https://github.com/user-attachments/assets/3ceb9fc1-89b4-44c9-af06-fb01483ba7ef)
![r2_chart](https://github.com/user-attachments/assets/6386e975-4943-468f-9ddd-86cbed305a33)


## License

This project is licensed under the MIT License.

## Acknowledgements

This project was developed as part of the Google Summer of Code (GSoC) 2024. Special thanks to the mentors and the open-source community for their support and guidance throughout the project.
