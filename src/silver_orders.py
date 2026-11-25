
from pathlib import Path

from pyspark.sql import SparkSession
from pyspark.sql import functions as F

# 1. PROJECT PATHS

PROJECT_ROOT = Path(__file__).resolve().parent.parent

BRONZE_PATH = (
    PROJECT_ROOT
    / "data"
    / "bronze"
    / "olist_orders_dataset"
)

SILVER_PATH = (
    PROJECT_ROOT
    / "data"
    / "silver"
    / "olist_orders_dataset"
)

QUARANTINE_PATH = (
    PROJECT_ROOT
    / "data"
    / "silver"
    / "quarantine"
    / "orders"
)

# 2. CREATE SPARK SESSION

spark = (
    SparkSession.builder
    .appName("OlistSilverOrders")
    .master("local[*]")
    .config("spark.driver.host", "127.0.0.1")
    .config("spark.driver.bindAddress", "127.0.0.1")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("ERROR")

# 3. READ BRONZE DATA

orders = spark.read.parquet(str(BRONZE_PATH))

print("\nORDERS DATA CLEANING")
print("=" * 60)

original_count = orders.count()

print(f"Original records: {original_count}")

# 4. CLEAN STRING COLUMNS

string_columns = [
    "order_id",
    "customer_id",
    "order_status",
]

for column in string_columns:
    orders = orders.withColumn(
        column,
        F.when(
            F.trim(F.col(column)) == "",
            None
        ).otherwise(F.trim(F.col(column)))
    )

# Normalize order status
orders = orders.withColumn(
    "order_status",
    F.lower(F.col("order_status"))
)

# 5. CONVERT STRING DATES TO TIMESTAMPS

timestamp_columns = [
    "order_purchase_timestamp",
    "order_approved_at",
    "order_delivered_carrier_date",
    "order_delivered_customer_date",
    "order_estimated_delivery_date",
]

for column in timestamp_columns:
    orders = orders.withColumn(
        column,
        F.to_timestamp(
            F.col(column),
            "yyyy-MM-dd HH:mm:ss"
        )
    )

# 6. IDENTIFY INVALID REQUIRED RECORDS

invalid_orders = orders.filter(
    F.col("order_id").isNull()
    | F.col("customer_id").isNull()
)

valid_orders = orders.filter(
    F.col("order_id").isNotNull()
    & F.col("customer_id").isNotNull()
)

invalid_count = invalid_orders.count()

print(f"Records with missing required IDs: {invalid_count}")

# Save invalid records separately for investigation
if invalid_count > 0:
    (
        invalid_orders.write
        .mode("overwrite")
        .parquet(str(QUARANTINE_PATH))
    )

# 7. REMOVE DUPLICATE ORDERS

before_deduplication = valid_orders.count()

duplicate_count = (
    before_deduplication
    - valid_orders.select("order_id").distinct().count()
)

clean_orders = valid_orders.dropDuplicates(["order_id"])

after_deduplication = clean_orders.count()

print(f"Duplicate order records removed: {duplicate_count}")

# 8. DATA QUALITY REPORT

print("\nDATA QUALITY REPORT")
print("-" * 60)

print(f"Original records: {original_count}")
print(f"Invalid records: {invalid_count}")
print(f"Records before deduplication: {before_deduplication}")
print(f"Duplicate records removed: {duplicate_count}")
print(f"Final clean records: {after_deduplication}")

print("\nORDER STATUS DISTRIBUTION")

clean_orders.groupBy("order_status").count().orderBy(
    F.desc("count")
).show(truncate=False)

print("\nFINAL SCHEMA")
clean_orders.printSchema()

# 9. SAVE SILVER DATA

(
    clean_orders.write
    .mode("overwrite")
    .parquet(str(SILVER_PATH))
)

print("\nSilver orders dataset saved successfully!")
print(f"Output location: {SILVER_PATH}")

spark.stop()

print("\nORDERS CLEANING COMPLETED")
