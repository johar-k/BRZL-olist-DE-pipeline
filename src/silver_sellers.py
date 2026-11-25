
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
    / "olist_sellers_dataset"
)

SILVER_PATH = (
    PROJECT_ROOT
    / "data"
    / "silver"
    / "olist_sellers_dataset"
)

QUARANTINE_PATH = (
    PROJECT_ROOT
    / "data"
    / "silver"
    / "quarantine"
    / "sellers"
)

# 2. CREATE SPARK SESSION

spark = (
    SparkSession.builder
    .appName("OlistSilverSellers")
    .master("local[*]")
    .config("spark.driver.host", "127.0.0.1")
    .config("spark.driver.bindAddress", "127.0.0.1")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("ERROR")

# 3. READ BRONZE DATA

sellers = spark.read.parquet(str(BRONZE_PATH))

original_count = sellers.count()

print("\nSELLERS DATA CLEANING")
print("=" * 60)

print(f"Original records: {original_count}")

# 4. CLEAN STRING COLUMNS

string_columns = [
    "seller_id",
    "seller_zip_code_prefix",
    "seller_city",
    "seller_state",
]

for column in string_columns:

    sellers = sellers.withColumn(
        column,
        F.when(
            F.trim(F.col(column)) == "",
            None
        ).otherwise(F.trim(F.col(column)))
    )

# Normalize city and state
sellers = (
    sellers
    .withColumn("seller_city", F.lower(F.col("seller_city")))
    .withColumn("seller_state", F.upper(F.col("seller_state")))
)

# 5. VALIDATE SELLER IDENTIFIERS

invalid_sellers = sellers.filter(
    F.col("seller_id").isNull()
)

valid_sellers = sellers.filter(
    F.col("seller_id").isNotNull()
)

invalid_count = invalid_sellers.count()

# 6. DETECT DUPLICATE SELLER IDS

window_spec = Window.partitionBy("seller_id")

sellers_with_counts = valid_sellers.withColumn(
    "_seller_count",
    F.count("*").over(window_spec)
)

duplicate_sellers = sellers_with_counts.filter(
    F.col("_seller_count") > 1
).drop("_seller_count")

unique_sellers = sellers_with_counts.filter(
    F.col("_seller_count") == 1
).drop("_seller_count")

duplicate_count = duplicate_sellers.count()

# 7. VALIDATE ZIP CODE PREFIX

unique_sellers = unique_sellers.withColumn(
    "_zip_valid",
    F.col("seller_zip_code_prefix").rlike(r"^[0-9]{5}$")
)

invalid_zip_count = unique_sellers.filter(
    F.col("seller_zip_code_prefix").isNotNull()
    & (~F.col("_zip_valid"))
).count()

missing_zip_count = unique_sellers.filter(
    F.col("seller_zip_code_prefix").isNull()
).count()

# Preserve ZIP values for investigation.
clean_sellers = unique_sellers.drop("_zip_valid")

# 8. QUARANTINE INVALID RECORDS

quarantined = (
    invalid_sellers
    .unionByName(duplicate_sellers)
)

quarantine_count = quarantined.count()

if quarantine_count > 0:

    (
        quarantined.write
        .mode("overwrite")
        .parquet(str(QUARANTINE_PATH))
    )

# 9. DATA QUALITY REPORT

final_count = clean_sellers.count()

print("\nSELLER DATA QUALITY REPORT")
print("-" * 60)

print(f"Original records: {original_count}")
print(f"Missing seller IDs: {invalid_count}")
print(f"Duplicate seller records: {duplicate_count}")
print(f"Quarantined records: {quarantine_count}")
print(f"Invalid ZIP prefixes: {invalid_zip_count}")
print(f"Missing ZIP prefixes: {missing_zip_count}")
print(f"Final clean records: {final_count}")

print("\nSELLERS BY STATE")

clean_sellers.groupBy("seller_state").count().orderBy(
    F.desc("count")
).show(truncate=False)

print("\nFINAL SCHEMA")
clean_sellers.printSchema()

# 10. SAVE SILVER DATA

(
    clean_sellers.write
    .mode("overwrite")
    .parquet(str(SILVER_PATH))
)

print("\nSilver sellers dataset saved successfully!")
print(f"Output: {SILVER_PATH}")

spark.stop()

print("\nSELLERS CLEANING COMPLETED")
