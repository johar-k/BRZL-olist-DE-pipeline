
from datetime import timedelta
from pathlib import Path

import pendulum

from airflow.sdk import DAG
from airflow.providers.standard.operators.bash import BashOperator


# Existing PySpark project
PROJECT_DIR = Path(
    "/home/xanax/Documents/data-engineering/spark-1"
)

PYTHON = str(PROJECT_DIR / ".venv/bin/python")


# Helper to create a Spark project task
def spark_task(task_id, script):

    return BashOperator(
        task_id=task_id,

        bash_command=f"{PYTHON} src/{script}",

        cwd=str(PROJECT_DIR),

        env={
            "SPARK_LOCAL_IP": "127.0.0.1",
            "PYTHONUNBUFFERED": "1"
        },

        append_env=True
    )


with DAG(
    dag_id="olist_end_to_end_pipeline",

    description="Complete Olist Bronze to Gold data engineering pipeline",

    start_date=pendulum.datetime(
        2025, 1, 1, tz="UTC"
    ),

    schedule=None,

    catchup=False,

    max_active_runs=1,

    default_args={
        "owner": "data-engineering",
        "retries": 0
    },

    tags=["olist", "pyspark", "bronze", "silver", "gold"]
) as dag:

    # PHASE 1: BRONZE

    bronze = spark_task(
        "bronze_ingestion",
        "bronze_ingestion.py"
    )

    # PHASE 2: SILVER CLEANING

    silver_orders = spark_task(
        "silver_orders",
        "silver_orders.py"
    )

    silver_customers = spark_task(
        "silver_customers",
        "silver_customers.py"
    )

    silver_products = spark_task(
        "silver_products",
        "silver_products.py"
    )

    silver_sellers = spark_task(
        "silver_sellers",
        "silver_sellers.py"
    )

    silver_items = spark_task(
        "silver_order_items",
        "silver_order_items.py"
    )

    silver_payments = spark_task(
        "silver_payments",
        "silver_payments.py"
    )

    silver_reviews = spark_task(
        "silver_reviews",
        "silver_reviews.py"
    )

    silver_categories = spark_task(
        "silver_category_translation",
        "silver_category_translation.py"
    )

    silver_geolocation = spark_task(
        "silver_geolocation",
        "silver_geolocation.py"
    )

    silver_tasks = [
        silver_orders,
        silver_customers,
        silver_products,
        silver_sellers,
        silver_items,
        silver_payments,
        silver_reviews,
        silver_categories,
        silver_geolocation
    ]

    bronze >> silver_tasks

    # PHASE 3: SILVER INTEGRATION

    integrate_orders = spark_task(
        "integrate_orders",
        "integrate_orders.py"
    )

    integrate_items = spark_task(
        "integrate_items",
        "integrate_items.py"
    )

    # Orders integration dependencies
    [
        silver_orders,
        silver_customers,
        silver_payments,
        silver_reviews
    ] >> integrate_orders

    # Items integration dependencies
    [
        silver_orders,
        silver_items,
        silver_products,
        silver_sellers,
        silver_categories
    ] >> integrate_items

    validate_silver = spark_task(
        "validate_silver",
        "validate_silver.py"
    )

    silver_tasks >> validate_silver

    [
        integrate_orders,
        integrate_items
    ] >> validate_silver

    # PHASE 4: GOLD DIMENSIONS

    gold_dimensions = spark_task(
        "gold_dimensions",
        "gold_dimensions.py"
    )

    validate_silver >> gold_dimensions

    # PHASE 5: GOLD FACT TABLES

    fact_sales = spark_task(
        "gold_fact_sales",
        "gold_fact_sales.py"
    )

    fact_payments = spark_task(
        "gold_fact_payments",
        "gold_fact_payments.py"
    )

    fact_reviews = spark_task(
        "gold_fact_reviews",
        "gold_fact_reviews.py"
    )

    fact_delivery = spark_task(
        "gold_fact_delivery",
        "gold_fact_delivery.py"
    )

    gold_dimensions >> [
        fact_sales,
        fact_payments,
        fact_reviews,
        fact_delivery
    ]

    # PHASE 6: GOLD VALIDATION

    validate_gold = spark_task(
        "validate_gold",
        "validate_gold.py"
    )

    [
        fact_sales,
        fact_payments,
        fact_reviews,
        fact_delivery
    ] >> validate_gold

    # PHASE 7: BUSINESS ANALYTICS

    analytics_monthly = spark_task(
        "analytics_monthly_sales",
        "analytics_monthly_sales.py"
    )

    analytics_region = spark_task(
        "analytics_customer_region",
        "analytics_customer_region.py"
    )

    analytics_seller = spark_task(
        "analytics_seller_performance",
        "analytics_seller_performance.py"
    )

    analytics_payments = spark_task(
        "analytics_payment_transactions",
        "analytics_payment_transactions.py"
    )

    analytics_reviews = spark_task(
        "analytics_customer_reviews",
        "analytics_customer_reviews.py"
    )

    analytics_delivery = spark_task(
        "analytics_delivery_logistics",
        "analytics_delivery_logistics.py"
    )

    analytics_windows = spark_task(
        "analytics_window_functions",
        "analytics_window_functions.py"
    )

    analytics_tasks = [
        analytics_monthly,
        analytics_region,
        analytics_seller,
        analytics_payments,
        analytics_reviews,
        analytics_delivery,
        analytics_windows
    ]

    validate_gold >> analytics_tasks
