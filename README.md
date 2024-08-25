# Graph4IO: Heterogeneous Graph Neural Networks for I/O Performance Bottleneck Diagnosis

![Graph4IO](path_to_your_image) 

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
   cd Graph4IO```
2.  **Install dependencies**:
   ```bash
   pip install -r requirements.txt```
