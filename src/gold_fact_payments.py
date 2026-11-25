
from pathlib import Path

from pyspark.sql import SparkSession
from pyspark.sql import functions as F


# 1. PROJECT PATHS

BASE_DIR = Path(__file__).resolve().parent.parent

SILVER_DIR = BASE_DIR / "data" / "silver"
GOLD_DIR = BASE_DIR / "data" / "gold"

PAYMENTS_PATH = SILVER_DIR / "olist_order_payments_dataset"
ORDERS_PATH = SILVER_DIR / "olist_orders_enriched"

CUSTOMER_DIM_PATH = GOLD_DIR / "dim_customer"
DATE_DIM_PATH = GOLD_DIR / "dim_date"

OUTPUT_PATH = GOLD_DIR / "fact_payments"


# 2. CREATE SPARK SESSION

spark = (
    SparkSession.builder
    .appName("OlistGoldFactPayments")
    .master("local[*]")
    .config("spark.driver.host", "127.0.0.1")
    .config("spark.driver.bindAddress", "127.0.0.1")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("WARN")


# 3. READ SOURCE DATA AND DIMENSIONS

print("\nReading Silver data and Gold dimensions...")

payments = spark.read.parquet(str(PAYMENTS_PATH))
orders = spark.read.parquet(str(ORDERS_PATH))

customers = spark.read.parquet(str(CUSTOMER_DIM_PATH))
dates = spark.read.parquet(str(DATE_DIM_PATH))

source_count = payments.count()

print(f"Source payment records: {source_count}")


# 4. ATTACH CUSTOMER AND ORDER DATE

print("\nJoining payments with order information...")

# orders_enriched contains one row per order.
payment_data = payments.join(
    orders.select(
        "order_id",
        "customer_id",
        "order_purchase_timestamp"
    ),
    on="order_id",
    how="left"
)


# 5. MAP CUSTOMER SURROGATE KEY

print("\nMapping customer keys...")

payment_data = payment_data.join(
    customers.select("customer_id", "customer_key"),
    on="customer_id",
    how="left"
)


# 6. MAP DATE SURROGATE KEY

payment_data = payment_data.withColumn(
    "_purchase_date",
    F.to_date("order_purchase_timestamp")
)

payment_data = payment_data.join(
    dates.select("full_date", "date_key"),
    payment_data["_purchase_date"] == dates["full_date"],
    how="left"
).drop("full_date")


# 7. CHECK DIMENSION MATCHES

print("\nChecking dimension key matches...")

missing_customer_keys = payment_data.filter(
    F.col("customer_key").isNull()
).count()

missing_date_keys = payment_data.filter(
    F.col("date_key").isNull()
).count()

missing_orders = payment_data.filter(
    F.col("customer_id").isNull()
).count()

print(f"Missing customer keys: {missing_customer_keys}")
print(f"Missing date keys: {missing_date_keys}")
print(f"Payments without matching orders: {missing_orders}")

if (
    missing_customer_keys > 0
    or missing_date_keys > 0
    or missing_orders > 0
):
    raise ValueError(
        "Payment records contain missing order or dimension references."
    )


# 8. BUILD FACT PAYMENTS

print("\nBuilding fact_payments...")

fact_payments = payment_data.select(
    "order_id",
    "payment_sequential",

    "customer_key",
    "date_key",

    "payment_type",
    "payment_installments",

    F.col("payment_value")
        .cast("decimal(12,2)")
        .alias("payment_value")
)


# 9. DATA QUALITY CHECKS

print("\nRunning fact_payments quality checks...")

fact_count = fact_payments.count()

duplicate_payments = (
    fact_payments
    .groupBy("order_id", "payment_sequential")
    .count()
    .filter(F.col("count") > 1)
    .count()
)

null_measures = fact_payments.filter(
    F.col("payment_value").isNull()
).count()

negative_payments = fact_payments.filter(
    F.col("payment_value") < 0
).count()

null_keys = fact_payments.filter(
    F.col("customer_key").isNull()
    | F.col("date_key").isNull()
).count()

print(f"Source payment count: {source_count}")
print(f"Fact payment count: {fact_count}")
print(f"Duplicate payment records: {duplicate_payments}")
print(f"Missing payment values: {null_measures}")
print(f"Negative payment values: {negative_payments}")
print(f"Missing dimension keys: {null_keys}")

if source_count != fact_count:
    raise ValueError("Payment record count changed during transformation!")

if duplicate_payments > 0:
    raise ValueError("Duplicate payment records found!")

if null_measures > 0:
    raise ValueError("Missing payment values found!")

if negative_payments > 0:
    raise ValueError("Negative payment values found!")

if null_keys > 0:
    raise ValueError("Missing dimension keys found!")


# 10. WRITE GOLD FACT TABLE

print("\nWriting fact_payments to Gold...")

(
    fact_payments
    .write
    .mode("overwrite")
    .parquet(str(OUTPUT_PATH))
)


# 11. DISPLAY RESULTS

print("\nFact payments sample:")

fact_payments.show(10, truncate=False)

print("\nFact payments schema:")

fact_payments.printSchema()

print(f"\nTotal fact_payments records: {fact_count}")

print("\nGold fact_payments created successfully!")


# 12. STOP SPARK

spark.stop()
