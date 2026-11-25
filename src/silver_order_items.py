
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
    / "olist_order_items_dataset"
)

SILVER_PATH = (
    PROJECT_ROOT
    / "data"
    / "silver"
    / "olist_order_items_dataset"
)

QUARANTINE_PATH = (
    PROJECT_ROOT
    / "data"
    / "silver"
    / "quarantine"
    / "order_items"
)

ORDERS_PATH = (
    PROJECT_ROOT
    / "data"
    / "silver"
    / "olist_orders_dataset"
)

PRODUCTS_PATH = (
    PROJECT_ROOT
    / "data"
    / "silver"
    / "olist_products_dataset"
)

SELLERS_PATH = (
    PROJECT_ROOT
    / "data"
    / "silver"
    / "olist_sellers_dataset"
)

# 2. CREATE SPARK SESSION

spark = (
    SparkSession.builder
    .appName("OlistSilverOrderItems")
    .master("local[*]")
    .config("spark.driver.host", "127.0.0.1")
    .config("spark.driver.bindAddress", "127.0.0.1")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("ERROR")

# 3. READ BRONZE DATA

items = spark.read.parquet(str(BRONZE_PATH))

original_count = items.count()

print("\nORDER ITEMS DATA CLEANING")
print("=" * 65)
print(f"Original records: {original_count}")

# 4. CLEAN STRING COLUMNS

string_columns = [
    "order_id",
    "product_id",
    "seller_id",
]

for column in string_columns:

    items = items.withColumn(
        column,
        F.when(
            F.trim(F.col(column)) == "",
            None
        ).otherwise(F.trim(F.col(column)))
    )

# 5. CONVERT DATA TYPES

items = (
    items
    .withColumn(
        "order_item_id",
        F.expr("try_cast(order_item_id AS INT)")
    )
    .withColumn(
        "price",
        F.expr("try_cast(price AS DECIMAL(12,2))")
    )
    .withColumn(
        "freight_value",
        F.expr("try_cast(freight_value AS DECIMAL(12,2))")
    )
    .withColumn(
        "shipping_limit_date",
        F.to_timestamp(
            "shipping_limit_date",
            "yyyy-MM-dd HH:mm:ss"
        )
    )
)

# 6. VALIDATE REQUIRED FIELDS

invalid_required = items.filter(
    F.col("order_id").isNull()
    | F.col("order_item_id").isNull()
    | F.col("product_id").isNull()
    | F.col("seller_id").isNull()
)

valid_required = items.filter(
    F.col("order_id").isNotNull()
    & F.col("order_item_id").isNotNull()
    & F.col("product_id").isNotNull()
    & F.col("seller_id").isNotNull()
)

invalid_required_count = invalid_required.count()

# 7. VALIDATE PRICE AND FREIGHT

invalid_amounts = valid_required.filter(
    F.col("price").isNull()
    | F.col("freight_value").isNull()
    | (F.col("price") < 0)
    | (F.col("freight_value") < 0)
)

valid_amounts = valid_required.filter(
    F.col("price").isNotNull()
    & F.col("freight_value").isNotNull()
    & (F.col("price") >= 0)
    & (F.col("freight_value") >= 0)
)

invalid_amount_count = invalid_amounts.count()

# 8. DETECT DUPLICATE COMPOSITE KEYS

window_spec = Window.partitionBy(
    "order_id",
    "order_item_id"
)

items_with_counts = valid_amounts.withColumn(
    "_key_count",
    F.count("*").over(window_spec)
)

duplicate_items = items_with_counts.filter(
    F.col("_key_count") > 1
).drop("_key_count")

unique_items = items_with_counts.filter(
    F.col("_key_count") == 1
).drop("_key_count")

duplicate_count = duplicate_items.count()

# 9. VALIDATE FOREIGN KEY RELATIONSHIPS

orders = (
    spark.read.parquet(str(ORDERS_PATH))
    .select("order_id")
    .distinct()
    .withColumn("_order_exists", F.lit(True))
)

products = (
    spark.read.parquet(str(PRODUCTS_PATH))
    .select("product_id")
    .distinct()
    .withColumn("_product_exists", F.lit(True))
)

sellers = (
    spark.read.parquet(str(SELLERS_PATH))
    .select("seller_id")
    .distinct()
    .withColumn("_seller_exists", F.lit(True))
)

checked_items = (
    unique_items
    .join(orders, on="order_id", how="left")
    .join(products, on="product_id", how="left")
    .join(sellers, on="seller_id", how="left")
)

orphan_items = checked_items.filter(
    F.col("_order_exists").isNull()
    | F.col("_product_exists").isNull()
    | F.col("_seller_exists").isNull()
)

clean_items = checked_items.filter(
    F.col("_order_exists").isNotNull()
    & F.col("_product_exists").isNotNull()
    & F.col("_seller_exists").isNotNull()
)

orphan_count = orphan_items.count()

# 10. REMOVE TEMPORARY VALIDATION COLUMNS

validation_columns = [
    "_order_exists",
    "_product_exists",
    "_seller_exists",
]

clean_items = clean_items.drop(*validation_columns)

orphan_items = orphan_items.drop(*validation_columns)

# 11. QUARANTINE INVALID RECORDS

quarantined = (
    invalid_required
    .unionByName(invalid_amounts)
    .unionByName(duplicate_items)
    .unionByName(orphan_items)
)

quarantine_count = quarantined.count()

if quarantine_count > 0:

    (
        quarantined.write
        .mode("overwrite")
        .parquet(str(QUARANTINE_PATH))
    )

# 12. DATA QUALITY REPORT

final_count = clean_items.count()

print("\nORDER ITEMS DATA QUALITY REPORT")
print("-" * 65)

print(f"Original records: {original_count}")
print(f"Invalid required fields: {invalid_required_count}")
print(f"Invalid price/freight records: {invalid_amount_count}")
print(f"Duplicate composite-key records: {duplicate_count}")
print(f"Orphan records: {orphan_count}")
print(f"Total quarantined records: {quarantine_count}")
print(f"Final clean records: {final_count}")

print("\nFINAL SCHEMA")
clean_items.printSchema()

print("\nTOP SELLERS BY ITEM COUNT")

clean_items.groupBy("seller_id").count().orderBy(
    F.desc("count")
).show(10, truncate=False)

# 13. SAVE SILVER DATA
(
    clean_items.write
    .mode("overwrite")
    .parquet(str(SILVER_PATH))
)

print("\nSilver order items dataset saved successfully!")
print(f"Output: {SILVER_PATH}")

spark.stop()

print("\nORDER ITEMS CLEANING COMPLETED")
