
from pathlib import Path

from pyspark.sql import SparkSession
from pyspark.sql import functions as F


# 1. PATHS

BASE_DIR = Path(__file__).resolve().parent.parent

SILVER_DIR = BASE_DIR / "data/silver"
GOLD_DIR = BASE_DIR / "data/gold"

ORDERS_PATH = SILVER_DIR / "olist_orders_enriched"
CUSTOMER_DIM_PATH = GOLD_DIR / "dim_customer"
DATE_DIM_PATH = GOLD_DIR / "dim_date"

OUTPUT_PATH = GOLD_DIR / "fact_delivery"


# 2. SPARK SESSION

spark = (
    SparkSession.builder
    .appName("OlistGoldFactDelivery")
    .master("local[*]")
    .config("spark.driver.host", "127.0.0.1")
    .config("spark.driver.bindAddress", "127.0.0.1")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("WARN")


# 3. READ DATA

print("\nReading Silver orders and Gold dimensions...")

orders = spark.read.parquet(str(ORDERS_PATH))
customers = spark.read.parquet(str(CUSTOMER_DIM_PATH))
dates = spark.read.parquet(str(DATE_DIM_PATH))

source_count = orders.count()

print("Source orders:", source_count)


# 4. JOIN CUSTOMER DIMENSION

print("\nMapping customer keys...")

delivery = orders.join(
    customers.select("customer_id", "customer_key"),
    on="customer_id",
    how="left"
)


# 5. PREPARE DATES

delivery = (
    delivery
    .withColumn(
        "_purchase_date",
        F.to_date("order_purchase_timestamp")
    )
    .withColumn(
        "_delivered_date",
        F.to_date("order_delivered_customer_date")
    )
    .withColumn(
        "_estimated_date",
        F.to_date("order_estimated_delivery_date")
    )
)


# 6. MAP PURCHASE DATE KEY

delivery = delivery.join(
    dates.select(
        F.col("full_date").alias("_purchase_lookup"),
        F.col("date_key").alias("purchase_date_key")
    ),
    delivery["_purchase_date"] == F.col("_purchase_lookup"),
    "left"
).drop("_purchase_lookup")


# 7. MAP ACTUAL DELIVERY DATE KEY

delivery = delivery.join(
    dates.select(
        F.col("full_date").alias("_delivered_lookup"),
        F.col("date_key").alias("delivered_date_key")
    ),
    delivery["_delivered_date"] == F.col("_delivered_lookup"),
    "left"
).drop("_delivered_lookup")


# 8. MAP ESTIMATED DELIVERY DATE KEY

delivery = delivery.join(
    dates.select(
        F.col("full_date").alias("_estimated_lookup"),
        F.col("date_key").alias("estimated_delivery_date_key")
    ),
    delivery["_estimated_date"] == F.col("_estimated_lookup"),
    "left"
).drop("_estimated_lookup")


# 9. VALIDATE DIMENSION KEYS

print("\nChecking dimension keys...")

missing_customer = delivery.filter(
    F.col("customer_key").isNull()
).count()

missing_purchase_date = delivery.filter(
    F.col("purchase_date_key").isNull()
).count()

missing_delivered_date = delivery.filter(
    F.col("order_delivered_customer_date").isNotNull()
    & F.col("delivered_date_key").isNull()
).count()

missing_estimated_date = delivery.filter(
    F.col("order_estimated_delivery_date").isNotNull()
    & F.col("estimated_delivery_date_key").isNull()
).count()

print("Missing customer keys:", missing_customer)
print("Missing purchase date keys:", missing_purchase_date)
print("Missing delivered date keys:", missing_delivered_date)
print("Missing estimated date keys:", missing_estimated_date)

if any([
    missing_customer,
    missing_purchase_date,
    missing_delivered_date,
    missing_estimated_date
]):
    raise ValueError("Delivery dimension mapping failed!")


# 10. CALCULATE DELIVERY METRICS

print("\nCalculating delivery metrics...")

delivery = (
    delivery
    .withColumn(
        "delivery_days",
        F.datediff("_delivered_date", "_purchase_date")
    )
    .withColumn(
        "estimated_delivery_days",
        F.datediff("_estimated_date", "_purchase_date")
    )
    .withColumn(
        "delivery_delay_days",
        F.datediff("_delivered_date", "_estimated_date")
    )
    .withColumn(
        "is_delivered",
        F.when(
            F.col("_delivered_date").isNotNull(), 1
        ).otherwise(0)
    )
    .withColumn(
        "is_late",
        F.when(
            F.col("_delivered_date").isNotNull()
            & F.col("_estimated_date").isNotNull()
            & (
                F.col("_delivered_date")
                > F.col("_estimated_date")
            ),
            1
        ).otherwise(0)
    )
)


# 11. BUILD FACT DELIVERY

fact_delivery = delivery.select(
    "order_id",
    "customer_key",
    "purchase_date_key",
    "delivered_date_key",
    "estimated_delivery_date_key",
    "order_status",
    "delivery_days",
    "estimated_delivery_days",
    "delivery_delay_days",
    "is_delivered",
    "is_late"
)


# 12. QUALITY CHECKS

print("\nRunning quality checks...")

fact_count = fact_delivery.count()

duplicate_orders = (
    fact_delivery
    .groupBy("order_id")
    .count()
    .filter(F.col("count") > 1)
    .count()
)

null_keys = fact_delivery.filter(
    F.col("order_id").isNull()
    | F.col("customer_key").isNull()
    | F.col("purchase_date_key").isNull()
).count()

invalid_delivery_days = fact_delivery.filter(
    F.col("delivery_days") < 0
).count()

print("Source orders:", source_count)
print("Fact delivery records:", fact_count)
print("Duplicate orders:", duplicate_orders)
print("Missing required keys:", null_keys)
print("Invalid delivery durations:", invalid_delivery_days)

if source_count != fact_count:
    raise ValueError("Source and fact counts do not match!")

if duplicate_orders > 0:
    raise ValueError("Duplicate orders found!")

if null_keys > 0:
    raise ValueError("Missing required keys found!")

if invalid_delivery_days > 0:
    raise ValueError("Invalid delivery duration found!")


# 13. WRITE GOLD TABLE

print("\nWriting fact_delivery...")

(
    fact_delivery
    .write
    .mode("overwrite")
    .parquet(str(OUTPUT_PATH))
)


# 14. DISPLAY RESULTS

print("\nFact delivery sample:")

fact_delivery.show(10, truncate=False)

print("\nFact delivery schema:")

fact_delivery.printSchema()

print("\nTotal records:", fact_count)

print("\nGold fact_delivery created successfully!")

spark.stop()
