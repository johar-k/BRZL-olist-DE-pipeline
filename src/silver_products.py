
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
    / "olist_products_dataset"
)

SILVER_PATH = (
    PROJECT_ROOT
    / "data"
    / "silver"
    / "olist_products_dataset"
)

QUARANTINE_PATH = (
    PROJECT_ROOT
    / "data"
    / "silver"
    / "quarantine"
    / "products"
)

# 2. CREATE SPARK SESSION

spark = (
    SparkSession.builder
    .appName("OlistSilverProducts")
    .master("local[*]")
    .config("spark.driver.host", "127.0.0.1")
    .config("spark.driver.bindAddress", "127.0.0.1")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("ERROR")

# 3. READ BRONZE DATA

products = spark.read.parquet(str(BRONZE_PATH))

original_count = products.count()

print("\nPRODUCTS DATA CLEANING")
print("=" * 60)
print(f"Original records: {original_count}")

# 4. CLEAN STRING COLUMNS

string_columns = [
    "product_id",
    "product_category_name",
]

for column in string_columns:

    products = products.withColumn(
        column,
        F.when(
            F.trim(F.col(column)) == "",
            None
        ).otherwise(F.trim(F.col(column)))
    )

# Normalize category names
products = products.withColumn(
    "product_category_name",
    F.lower(F.col("product_category_name"))
)

# 5. CONVERT NUMERIC COLUMNS

numeric_columns = [
    "product_name_lenght",
    "product_description_lenght",
    "product_photos_qty",
    "product_weight_g",
    "product_length_cm",
    "product_height_cm",
    "product_width_cm",
]

for column in numeric_columns:

    products = products.withColumn(
        column,
        F.expr(f"try_cast(`{column}` AS INT)")
    )

# 6. IDENTIFY MISSING PRODUCT IDS

invalid_ids = products.filter(
    F.col("product_id").isNull()
)

valid_products = products.filter(
    F.col("product_id").isNotNull()
)

invalid_id_count = invalid_ids.count()

# 7. IDENTIFY DUPLICATE PRODUCT IDS

window_spec = Window.partitionBy("product_id")

products_with_counts = valid_products.withColumn(
    "_product_id_count",
    F.count("*").over(window_spec)
)

duplicate_products = products_with_counts.filter(
    F.col("_product_id_count") > 1
).drop("_product_id_count")

unique_products = products_with_counts.filter(
    F.col("_product_id_count") == 1
).drop("_product_id_count")

duplicate_count = duplicate_products.count()

# 8. VALIDATE PRODUCT MEASUREMENTS

measurement_columns = [
    "product_name_lenght",
    "product_description_lenght",
    "product_photos_qty",
    "product_weight_g",
    "product_length_cm",
    "product_height_cm",
    "product_width_cm",
]

invalid_condition = None

for column in measurement_columns:

    condition = F.col(column) < 0

    if invalid_condition is None:
        invalid_condition = condition
    else:
        invalid_condition = invalid_condition | condition

invalid_measurements = unique_products.filter(
    invalid_condition
)

valid_measurements = unique_products.filter(
    ~invalid_condition
    | invalid_condition.isNull()
)

# Separate records with invalid measurements
invalid_measurement_count = invalid_measurements.count()

# 9. CREATE FINAL CLEAN DATASET

clean_products = valid_measurements

# 10. QUARANTINE INVALID RECORDS

quarantined = (
    invalid_ids
    .unionByName(duplicate_products)
    .unionByName(invalid_measurements)
)

quarantine_count = quarantined.count()

if quarantine_count > 0:

    (
        quarantined.write
        .mode("overwrite")
        .parquet(str(QUARANTINE_PATH))
    )

# 11. DATA QUALITY REPORT

final_count = clean_products.count()

print("\nPRODUCT DATA QUALITY REPORT")
print("-" * 60)

print(f"Original records: {original_count}")
print(f"Missing product IDs: {invalid_id_count}")
print(f"Duplicate product records: {duplicate_count}")
print(f"Invalid measurement records: {invalid_measurement_count}")
print(f"Total quarantined records: {quarantine_count}")
print(f"Final clean records: {final_count}")

print("\nPRODUCT CATEGORY DISTRIBUTION")

clean_products.groupBy("product_category_name").count().orderBy(
    F.desc("count")
).show(20, truncate=False)

print("\nFINAL SCHEMA")
clean_products.printSchema()

# 12. SAVE SILVER DATA

(
    clean_products.write
    .mode("overwrite")
    .parquet(str(SILVER_PATH))
)

print("\nSilver products dataset saved successfully!")
print(f"Output: {SILVER_PATH}")

spark.stop()

print("\nPRODUCTS CLEANING COMPLETED")
