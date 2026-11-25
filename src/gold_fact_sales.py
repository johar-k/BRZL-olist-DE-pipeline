
from pathlib import Path

from pyspark.sql import SparkSession
from pyspark.sql import functions as F


# 1. PROJECT PATHS

BASE_DIR = Path(__file__).resolve().parent.parent

SILVER_DIR = BASE_DIR / "data" / "silver"
GOLD_DIR = BASE_DIR / "data" / "gold"

ITEMS_PATH = SILVER_DIR / "olist_items_enriched"

CUSTOMER_DIM_PATH = GOLD_DIR / "dim_customer"
PRODUCT_DIM_PATH = GOLD_DIR / "dim_product"
SELLER_DIM_PATH = GOLD_DIR / "dim_seller"
DATE_DIM_PATH = GOLD_DIR / "dim_date"

OUTPUT_PATH = GOLD_DIR / "fact_sales"


# 2. CREATE SPARK SESSION

spark = (
    SparkSession.builder
    .appName("OlistGoldFactSales")
    .master("local[*]")
    .config("spark.driver.host", "127.0.0.1")
    .config("spark.driver.bindAddress", "127.0.0.1")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("WARN")


# 3. READ SILVER DATA AND GOLD DIMENSIONS

print("\nReading Silver data and Gold dimensions...")

items = spark.read.parquet(str(ITEMS_PATH))

customers = spark.read.parquet(str(CUSTOMER_DIM_PATH))
products = spark.read.parquet(str(PRODUCT_DIM_PATH))
sellers = spark.read.parquet(str(SELLER_DIM_PATH))
dates = spark.read.parquet(str(DATE_DIM_PATH))

source_count = items.count()

print(f"Silver order item records: {source_count}")


# 4. JOIN DIMENSION SURROGATE KEYS

print("\nMapping dimension keys...")

# Customer dimension
sales = items.join(
    customers.select("customer_id", "customer_key"),
    on="customer_id",
    how="left"
)

# Product dimension
sales = sales.join(
    products.select("product_id", "product_key"),
    on="product_id",
    how="left"
)

# Seller dimension
sales = sales.join(
    sellers.select("seller_id", "seller_key"),
    on="seller_id",
    how="left"
)

# Date dimension
sales = sales.withColumn(
    "_purchase_date",
    F.to_date("order_purchase_timestamp")
)

sales = sales.join(
    dates.select("full_date", "date_key"),
    sales["_purchase_date"] == dates["full_date"],
    how="left"
).drop("full_date")


# 5. CHECK DIMENSION KEY MATCHES

print("\nChecking dimension key matches...")

missing_keys = sales.filter(
    F.col("customer_key").isNull()
    | F.col("product_key").isNull()
    | F.col("seller_key").isNull()
    | F.col("date_key").isNull()
).count()

print(f"Records with missing dimension keys: {missing_keys}")

if missing_keys > 0:
    raise ValueError(
        "Some order items could not find their dimension keys. "
        "Check the Silver records and Gold dimensions before continuing."
    )


# 6. BUILD FACT SALES

print("\nBuilding fact_sales...")

fact_sales = sales.select(
    "order_id",
    "order_item_id",

    "customer_key",
    "product_key",
    "seller_key",
    "date_key",

    F.lit(1).cast("integer").alias("item_quantity"),

    F.col("price").cast("decimal(12,2)").alias("product_price"),

    F.col("freight_value")
        .cast("decimal(12,2)")
        .alias("freight_value"),

    (
        F.col("price") + F.col("freight_value")
    ).cast("decimal(12,2)").alias("gross_item_value"),

    "order_status"
)


# 7. DATA QUALITY CHECKS

print("\nRunning fact_sales quality checks...")

fact_count = fact_sales.count()

duplicate_items = (
    fact_sales
    .groupBy("order_id", "order_item_id")
    .count()
    .filter(F.col("count") > 1)
    .count()
)

null_keys = fact_sales.filter(
    F.col("customer_key").isNull()
    | F.col("product_key").isNull()
    | F.col("seller_key").isNull()
    | F.col("date_key").isNull()
).count()

null_measures = fact_sales.filter(
    F.col("product_price").isNull()
    | F.col("freight_value").isNull()
).count()

print(f"Source item count: {source_count}")
print(f"Fact sales count: {fact_count}")
print(f"Duplicate order items: {duplicate_items}")
print(f"Missing dimension keys: {null_keys}")
print(f"Missing price/freight values: {null_measures}")

if source_count != fact_count:
    raise ValueError("Fact sales row count does not match the source!")

if duplicate_items > 0:
    raise ValueError("Duplicate order items found!")

if null_keys > 0:
    raise ValueError("Missing dimension keys found!")

if null_measures > 0:
    raise ValueError("Missing sales measures found!")


# 8. WRITE GOLD FACT TABLE

print("\nWriting fact_sales to Gold...")

(
    fact_sales
    .write
    .mode("overwrite")
    .parquet(str(OUTPUT_PATH))
)


# 9. DISPLAY RESULTS

print("\nFact sales sample:")

fact_sales.show(10, truncate=False)

print("\nFact sales schema:")

fact_sales.printSchema()

print(f"\nTotal fact_sales records: {fact_count}")

print("\nGold fact_sales created successfully!")


# 10. STOP SPARK

spark.stop()
