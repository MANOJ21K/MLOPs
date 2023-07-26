# MLOps with NYC Taxi Data

![MLOps](./images/mlops_image.jpg)

This repository showcases an end-to-end MLOps (Machine Learning Operations) implementation for data science tasks using NYC Taxi data. It demonstrates experiment tracking with MLflow and workflow orchestration with Prefect.

## Overview

The repository contains the following components:

1. **Data Science Tasks:** Jupyter notebooks (`*.ipynb`) with data exploration, preprocessing, feature engineering, modeling, and evaluation using NYC Taxi data.

2. **MLflow:** Experiment tracking and model versioning with MLflow. The `mlflow` directory contains the necessary scripts and configurations for logging experiments.

3. **Prefect Orchestration:** Workflow orchestration using Prefect. The `prefect` directory contains the Prefect flow for automating the end-to-end ML pipeline.

## Dataset

The NYC Taxi dataset is used for this project. The data includes taxi trip records, which are used for various data science tasks, such as prediction, classification, or clustering.

## Getting Started

To get started with the project, follow these steps:

1. Clone the repository:


3. Run the Jupyter notebooks in the `notebooks` directory to perform data science tasks.

4. For MLflow experiment tracking, navigate to the `mlflow` directory and execute the `mlflow_tracking.py` script.

5. For Prefect workflow orchestration, go to the `prefect` directory and run the `main_flow.py` script.

## Experiment Tracking with MLflow

MLflow is used to track experiments, log metrics, and save models. It provides a user-friendly web interface to visualize and compare experiments.

![MLflow](./images/mlflow_image.jpg)

## Workflow Orchestration with Prefect

Prefect is employed to automate and orchestrate the data science workflow. It manages dependencies, scheduling, and retries for robust and reliable pipelines.

![Prefect](./images/prefect_image.jpg)

## Contributing

Pull requests and contributions are welcome. For major changes, please open an issue first to discuss the changes.

## License

This project is licensed under the [MIT License](LICENSE).
