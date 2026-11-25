
from pathlib import Path

from pyspark.sql import SparkSession
from pyspark.sql import functions as F

# 1. PROJECT PATHS

PROJECT_ROOT = Path(__file__).resolve().parent.parent

SILVER_DIR = PROJECT_ROOT / "data" / "silver"

ORDERS_PATH = SILVER_DIR / "olist_orders_dataset"
CUSTOMERS_PATH = SILVER_DIR / "olist_customers_dataset"
PAYMENTS_PATH = SILVER_DIR / "olist_order_payments_dataset"
REVIEWS_PATH = SILVER_DIR / "olist_order_reviews_dataset"

OUTPUT_PATH = SILVER_DIR / "olist_orders_enriched"

# 2. CREATE SPARK SESSION

spark = (
    SparkSession.builder
    .appName("OlistOrdersIntegration")
    .master("local[*]")
    .config("spark.driver.host", "127.0.0.1")
    .config("spark.driver.bindAddress", "127.0.0.1")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("ERROR")

print("\nOLIST ORDERS INTEGRATION")
print("=" * 65)

# 3. READ SILVER DATASETS

orders = spark.read.parquet(str(ORDERS_PATH))
customers = spark.read.parquet(str(CUSTOMERS_PATH))
payments = spark.read.parquet(str(PAYMENTS_PATH))
reviews = spark.read.parquet(str(REVIEWS_PATH))

orders_count = orders.count()

print(f"Orders: {orders_count}")
print(f"Customers: {customers.count()}")
print(f"Payments: {payments.count()}")
print(f"Reviews: {reviews.count()}")

# 4. AGGREGATE PAYMENTS BY ORDER

payment_summary = (
    payments
    .groupBy("order_id")
    .agg(
        F.sum("payment_value").alias("total_payment_value"),

        F.count("*").alias("payment_count"),

        F.countDistinct("payment_type").alias(
            "payment_type_count"
        ),

        F.collect_set("payment_type").alias(
            "payment_methods"
        )
    )
)

# 5. AGGREGATE REVIEWS BY ORDER

review_summary = (
    reviews
    .groupBy("order_id")
    .agg(
        F.count("*").alias("review_count"),

        F.avg("review_score").alias(
            "average_review_score"
        ),

        F.min("review_score").alias(
            "minimum_review_score"
        ),

        F.max("review_score").alias(
            "maximum_review_score"
        )
    )
)

# 6. JOIN ORDERS WITH CUSTOMERS

orders_customers = (
    orders.alias("o")
    .join(
        customers.alias("c"),
        F.col("o.customer_id") == F.col("c.customer_id"),
        "left"
    )
    .select(
        "o.*",

        F.col("c.customer_unique_id"),
        F.col("c.customer_zip_code_prefix"),
        F.col("c.customer_city"),
        F.col("c.customer_state")
    )
)

# 7. JOIN PAYMENT SUMMARY

orders_with_payments = orders_customers.join(
    payment_summary,
    on="order_id",
    how="left"
)

# 8. JOIN REVIEW SUMMARY

enriched_orders = orders_with_payments.join(
    review_summary,
    on="order_id",
    how="left"
)

# 9. DATA QUALITY CHECKS

print("\nINTEGRATION QUALITY REPORT")
print("-" * 65)

final_count = enriched_orders.count()

distinct_order_count = enriched_orders.select(
    "order_id"
).distinct().count()

missing_customer_count = enriched_orders.filter(
    F.col("customer_unique_id").isNull()
).count()

missing_payment_count = enriched_orders.filter(
    F.col("payment_count").isNull()
).count()

missing_review_count = enriched_orders.filter(
    F.col("review_count").isNull()
).count()

print(f"Original order records: {orders_count}")
print(f"Integrated records: {final_count}")
print(f"Distinct order IDs: {distinct_order_count}")
print(f"Orders missing customer match: {missing_customer_count}")
print(f"Orders without payment records: {missing_payment_count}")
print(f"Orders without review records: {missing_review_count}")

if final_count != orders_count:
    raise ValueError(
        "Integration changed the number of order records!"
    )

if distinct_order_count != final_count:
    raise ValueError(
        "Duplicate order IDs detected in integrated dataset!"
    )

# 10. INSPECT INTEGRATED DATA

print("\nINTEGRATED DATA SAMPLE")

enriched_orders.select(
    "order_id",
    "customer_unique_id",
    "order_status",
    "customer_city",
    "customer_state",
    "total_payment_value",
    "payment_methods",
    "average_review_score"
).show(10, truncate=False)

print("\nFINAL SCHEMA")

enriched_orders.printSchema()

# 11. SAVE INTEGRATED SILVER DATA

(
    enriched_orders.write
    .mode("overwrite")
    .parquet(str(OUTPUT_PATH))
)

print("\nINTEGRATION COMPLETED SUCCESSFULLY!")
print(f"Output: {OUTPUT_PATH}")

spark.stop()
