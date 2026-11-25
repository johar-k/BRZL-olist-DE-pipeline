
import pendulum

from airflow.sdk import dag, task


@dag(
    dag_id="olist_first_pipeline",
    description="My first Airflow ETL workflow",
    schedule=None,
    start_date=pendulum.datetime(
        2025, 1, 1, tz="UTC"
    ),
    catchup=False,
    tags=["olist", "data-engineering"]
)
def olist_first_pipeline():

    # Task 1: Extract
    @task
    def extract_orders():

        print("Extracting order data...")

        orders = {
            "ORD101": 250.00,
            "ORD102": 450.50,
            "ORD103": 120.00,
            "ORD104": 890.00
        }

        print(f"Extracted {len(orders)} orders")

        return orders

    # Task 2: Transform
    @task
    def transform_orders(orders):

        print("Transforming order data...")

        total_orders = len(orders)
        total_sales = sum(orders.values())

        result = {
            "total_orders": total_orders,
            "total_sales": total_sales
        }

        print(f"Transformation result: {result}")

        return result

    # Task 3: Load
    @task
    def load_results(result):

        print("Loading results...")

        print(
            f"Total orders: {result['total_orders']}"
        )

        print(
            f"Total sales: {result['total_sales']}"
        )

        print("Pipeline completed successfully!")

    # Define dependencies
    extracted_data = extract_orders()

    transformed_data = transform_orders(
        extracted_data
    )

    load_results(transformed_data)


olist_first_pipeline()
