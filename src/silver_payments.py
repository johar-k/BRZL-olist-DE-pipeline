
from pathlib import Path

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window

# 1. PROJECT PATHS

PROJECT_ROOT = Path(__file__).resolve().parent.parent

BRONZE_PATH = (
    PROJECT_ROOT
    / "data"
    / "bronze"
    / "olist_order_payments_dataset"
)

SILVER_PATH = (
    PROJECT_ROOT
    / "data"
    / "silver"
    / "olist_order_payments_dataset"
)

QUARANTINE_PATH = (
    PROJECT_ROOT
    / "data"
    / "silver"
    / "quarantine"
    / "payments"
)

ORDERS_PATH = (
    PROJECT_ROOT
    / "data"
    / "silver"
    / "olist_orders_dataset"
)

# 2. CREATE SPARK SESSION

spark = (
    SparkSession.builder
    .appName("OlistSilverPayments")
    .master("local[*]")
    .config("spark.driver.host", "127.0.0.1")
    .config("spark.driver.bindAddress", "127.0.0.1")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("ERROR")

# 3. READ BRONZE DATA

payments = spark.read.parquet(str(BRONZE_PATH))

original_count = payments.count()

print("\nPAYMENTS DATA CLEANING")
print("=" * 60)

print(f"Original records: {original_count}")

# 4. CLEAN STRING COLUMNS

string_columns = [
    "order_id",
    "payment_type",
]

for column in string_columns:

    payments = payments.withColumn(
        column,
        F.when(
            F.trim(F.col(column)) == "",
            None
        ).otherwise(F.trim(F.col(column)))
    )

# Normalize payment type
payments = payments.withColumn(
    "payment_type",
    F.lower(F.col("payment_type"))
)

# 5. CONVERT DATA TYPES

payments = (
    payments
    .withColumn(
        "payment_sequential",
        F.expr("try_cast(payment_sequential AS INT)")
    )
    .withColumn(
        "payment_installments",
        F.expr("try_cast(payment_installments AS INT)")
    )
    .withColumn(
        "payment_value",
        F.expr("try_cast(payment_value AS DECIMAL(12,2))")
    )
)

# 6. VALIDATE REQUIRED FIELDS

invalid_required = payments.filter(
    F.col("order_id").isNull()
    | F.col("payment_sequential").isNull()
    | F.col("payment_type").isNull()
    | F.col("payment_installments").isNull()
    | F.col("payment_value").isNull()
)

valid_payments = payments.filter(
    F.col("order_id").isNotNull()
    & F.col("payment_sequential").isNotNull()
    & F.col("payment_type").isNotNull()
    & F.col("payment_installments").isNotNull()
    & F.col("payment_value").isNotNull()
)

invalid_required_count = invalid_required.count()

# 7. VALIDATE PAYMENT VALUES

invalid_values = valid_payments.filter(
    (F.col("payment_value") < 0)
    | (F.col("payment_sequential") <= 0)
    | (F.col("payment_installments") < 0)
)

valid_values = valid_payments.filter(
    (F.col("payment_value") >= 0)
    & (F.col("payment_sequential") > 0)
    & (F.col("payment_installments") >= 0)
)

invalid_value_count = invalid_values.count()

# 8. DETECT DUPLICATE COMPOSITE KEYS

window_spec = Window.partitionBy(
    "order_id",
    "payment_sequential"
)

payments_with_counts = valid_values.withColumn(
    "_key_count",
    F.count("*").over(window_spec)
)

duplicate_payments = payments_with_counts.filter(
    F.col("_key_count") > 1
).drop("_key_count")

unique_payments = payments_with_counts.filter(
    F.col("_key_count") == 1
).drop("_key_count")

duplicate_count = duplicate_payments.count()

# 9. VALIDATE ORDER REFERENCES

orders = (
    spark.read.parquet(str(ORDERS_PATH))
    .select("order_id")
    .distinct()
    .withColumn("_order_exists", F.lit(True))
)

checked_payments = unique_payments.join(
    orders,
    on="order_id",
    how="left"
)

orphan_payments = checked_payments.filter(
    F.col("_order_exists").isNull()
)

clean_payments = checked_payments.filter(
    F.col("_order_exists").isNotNull()
)

orphan_count = orphan_payments.count()

# 10. REMOVE TEMPORARY COLUMNS

clean_payments = clean_payments.drop("_order_exists")
orphan_payments = orphan_payments.drop("_order_exists")

# 11. QUARANTINE INVALID RECORDS

quarantined = (
    invalid_required
    .unionByName(invalid_values)
    .unionByName(duplicate_payments)
    .unionByName(orphan_payments)
)

quarantine_count = quarantined.count()

if quarantine_count > 0:

    (
        quarantined.write
        .mode("overwrite")
        .parquet(str(QUARANTINE_PATH))
    )

# 12. DATA QUALITY REPORT

final_count = clean_payments.count()

print("\nPAYMENTS DATA QUALITY REPORT")
print("-" * 60)

print(f"Original records: {original_count}")
print(f"Invalid required records: {invalid_required_count}")
print(f"Invalid payment records: {invalid_value_count}")
print(f"Duplicate composite-key records: {duplicate_count}")
print(f"Orphan payment records: {orphan_count}")
print(f"Total quarantined records: {quarantine_count}")
print(f"Final clean records: {final_count}")

print("\nPAYMENT TYPE DISTRIBUTION")

clean_payments.groupBy("payment_type").count().orderBy(
    F.desc("count")
).show(truncate=False)

print("\nPAYMENT VALUE SUMMARY")

clean_payments.select(
    F.sum("payment_value").alias("total_payment_value"),
    F.avg("payment_value").alias("average_payment_value"),
    F.min("payment_value").alias("minimum_payment_value"),
    F.max("payment_value").alias("maximum_payment_value")
).show(truncate=False)

print("\nFINAL SCHEMA")
clean_payments.printSchema()

# 13. SAVE SILVER DATA

(
    clean_payments.write
    .mode("overwrite")
    .parquet(str(SILVER_PATH))
)

print("\nSilver payments dataset saved successfully!")
print(f"Output: {SILVER_PATH}")

spark.stop()

print("\nPAYMENTS CLEANING COMPLETED")

