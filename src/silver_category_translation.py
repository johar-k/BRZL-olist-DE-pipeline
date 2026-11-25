
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
    / "product_category_name_translation"
)

SILVER_PATH = (
    PROJECT_ROOT
    / "data"
    / "silver"
    / "product_category_name_translation"
)

QUARANTINE_PATH = (
    PROJECT_ROOT
    / "data"
    / "silver"
    / "quarantine"
    / "category_translation"
)

# 2. CREATE SPARK SESSION

spark = (
    SparkSession.builder
    .appName("OlistSilverCategoryTranslation")
    .master("local[*]")
    .config("spark.driver.host", "127.0.0.1")
    .config("spark.driver.bindAddress", "127.0.0.1")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("ERROR")

# 3. READ BRONZE DATA

categories = spark.read.parquet(str(BRONZE_PATH))

original_count = categories.count()

print("\nCATEGORY TRANSLATION CLEANING")
print("=" * 60)

print(f"Original records: {original_count}")

# 4. CLEAN STRING COLUMNS

category_columns = [
    "product_category_name",
    "product_category_name_english",
]

for column in category_columns:

    categories = categories.withColumn(
        column,
        F.when(
            F.trim(F.col(column)) == "",
            None
        ).otherwise(F.trim(F.col(column)))
    )

# Normalize category names
categories = (
    categories
    .withColumn(
        "product_category_name",
        F.lower(F.col("product_category_name"))
    )
    .withColumn(
        "product_category_name_english",
        F.lower(F.col("product_category_name_english"))
    )
)

# 5. IDENTIFY INVALID RECORDS

invalid_categories = categories.filter(
    F.col("product_category_name").isNull()
    | F.col("product_category_name_english").isNull()
)

valid_categories = categories.filter(
    F.col("product_category_name").isNotNull()
    & F.col("product_category_name_english").isNotNull()
)

invalid_count = invalid_categories.count()

# 6. DETECT DUPLICATE PORTUGUESE CATEGORY KEYS

window_spec = Window.partitionBy(
    "product_category_name"
)

categories_with_counts = valid_categories.withColumn(
    "_category_count",
    F.count("*").over(window_spec)
)

duplicate_categories = categories_with_counts.filter(
    F.col("_category_count") > 1
).drop("_category_count")

unique_categories = categories_with_counts.filter(
    F.col("_category_count") == 1
).drop("_category_count")

duplicate_count = duplicate_categories.count()

# 7. DATA QUALITY REPORT
final_count = unique_categories.count()

print("\nCATEGORY TRANSLATION QUALITY REPORT")
print("-" * 60)

print(f"Original records: {original_count}")
print(f"Invalid records: {invalid_count}")
print(f"Duplicate category records: {duplicate_count}")
print(f"Final clean records: {final_count}")

print("\nCATEGORY TRANSLATIONS")

unique_categories.show(20, truncate=False)

# 8. QUARANTINE INVALID RECORDS

quarantined = (
    invalid_categories
    .unionByName(duplicate_categories)
)

quarantine_count = quarantined.count()

if quarantine_count > 0:

    (
        quarantined.write
        .mode("overwrite")
        .parquet(str(QUARANTINE_PATH))
    )

# 9. SAVE SILVER DATA

(
    unique_categories.write
    .mode("overwrite")
    .parquet(str(SILVER_PATH))
)

print("\nSilver category translation dataset saved!")
print(f"Output: {SILVER_PATH}")

spark.stop()

print("\nCATEGORY TRANSLATION CLEANING COMPLETED")
