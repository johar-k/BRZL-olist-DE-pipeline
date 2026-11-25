
# Olist E-Commerce Data Engineering Pipeline

An end-to-end Data Engineering project built using
Apache Spark, PySpark, Delta Lake and Apache Airflow.

## Project Overview

This project processes the Brazilian Olist E-Commerce
dataset through a Medallion Architecture:

- Bronze: Raw data ingestion
- Silver: Data cleaning and integration
- Gold: Dimensional data warehouse and analytics

## Technology Stack

- Python
- Apache Spark
- PySpark
- Apache Airflow
- Delta Lake
- PostgreSQL (database learning)
- Pytest
- Git and GitHub

## Architecture

Raw CSV Data
    |
    v
Bronze Layer
    |
    v
Silver Layer
    |
    v
Gold Layer
    |
    v
Business Analytics

Apache Airflow orchestrates the pipeline.

## Project Structure

- airflow/dags: Airflow workflows
- src: PySpark processing scripts
- tests: Automated tests
- data: Local datasets and generated storage
- output: Local pipeline reports

## Dataset

Brazilian E-Commerce Public Dataset by Olist:

https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce

Download the dataset separately and place the CSV files
inside data/raw/olist/.

## Local Setup

Install uv and Java 17.

Create the environment:

uv venv --python 3.13

Activate:

source .venv/bin/activate

Install project dependencies:

uv sync

Run a Spark script:

SPARK_LOCAL_IP=127.0.0.1 python src/bronze_ingestion.py

Run tests:

SPARK_LOCAL_IP=127.0.0.1 python -m pytest -v

## Airflow

The DAG files are available in airflow/dags/.

Airflow is installed in a separate Python environment.

## License

Add an appropriate project license after reviewing
the dataset's license and the terms for any dependencies
or material you redistribute.
# BRZL-olist-DE-pipeline
