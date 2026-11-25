
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
    / "olist_customers_dataset"
)

SILVER_PATH = (
    PROJECT_ROOT
    / "data"
    / "silver"
    / "olist_customers_dataset"
)

QUARANTINE_PATH = (
    PROJECT_ROOT
    / "data"
    / "silver"
    / "quarantine"
    / "customers"
)

# 2. CREATE SPARK SESSION

spark = (
    SparkSession.builder
    .appName("OlistSilverCustomers")
    .master("local[*]")
    .config("spark.driver.host", "127.0.0.1")
    .config("spark.driver.bindAddress", "127.0.0.1")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("ERROR")

# 3. READ BRONZE DATA

customers = spark.read.parquet(str(BRONZE_PATH))

original_count = customers.count()

print("\nCUSTOMERS DATA CLEANING")
print("=" * 60)
print(f"Original records: {original_count}")

# 4. CLEAN STRING COLUMNS

string_columns = [
    "customer_id",
    "customer_unique_id",
    "customer_zip_code_prefix",
    "customer_city",
    "customer_state",
]

for column in string_columns:

    customers = customers.withColumn(
        column,
        F.when(
            F.trim(F.col(column)) == "",
            None
        ).otherwise(F.trim(F.col(column)))
    )

# Normalize state codes
customers = customers.withColumn(
    "customer_state",
    F.upper(F.col("customer_state"))
)

# 5. IDENTIFY MISSING REQUIRED IDENTIFIERS

invalid_ids = customers.filter(
    F.col("customer_id").isNull()
    | F.col("customer_unique_id").isNull()
)

valid_ids = customers.filter(
    F.col("customer_id").isNotNull()
    & F.col("customer_unique_id").isNotNull()
)

invalid_count = invalid_ids.count()

print(f"Records with missing required IDs: {invalid_count}")

# 6. DETECT DUPLICATE PRIMARY KEYS

window_spec = Window.partitionBy("customer_id")

customers_with_counts = valid_ids.withColumn(
    "_customer_id_count",
    F.count("*").over(window_spec)
)

duplicate_customers = customers_with_counts.filter(
    F.col("_customer_id_count") > 1
).drop("_customer_id_count")

duplicate_count = duplicate_customers.count()

# Keep only records with unique customer_id
unique_customers = customers_with_counts.filter(
    F.col("_customer_id_count") == 1
).drop("_customer_id_count")

print(f"Records involved in duplicate customer IDs: {duplicate_count}")

# 7. ZIP CODE QUALITY CHECK

clean_customers = unique_customers.withColumn(
    "_zip_valid",
    F.col("customer_zip_code_prefix").rlike(r"^[0-9]{5}$")
)

invalid_zip_count = clean_customers.filter(
    F.col("customer_zip_code_prefix").isNotNull()
    & (~F.col("_zip_valid"))
).count()

missing_zip_count = clean_customers.filter(
    F.col("customer_zip_code_prefix").isNull()
).count()

print(f"Invalid ZIP prefixes: {invalid_zip_count}")
print(f"Missing ZIP prefixes: {missing_zip_count}")

# Keep ZIP values unchanged for investigation.
# Remove the temporary validation column.
clean_customers = clean_customers.drop("_zip_valid")

# 8. SAVE QUARANTINED RECORDS

quarantined = (
    invalid_ids
    .unionByName(duplicate_customers)
)

quarantine_count = quarantined.count()

if quarantine_count > 0:
    (
        quarantined.write
        .mode("overwrite")
        .parquet(str(QUARANTINE_PATH))
    )

# 9. FINAL DATA QUALITY REPORT

final_count = clean_customers.count()

print("\nCUSTOMER DATA QUALITY REPORT")
print("-" * 60)

print(f"Original records: {original_count}")
print(f"Missing identifier records: {invalid_count}")
print(f"Duplicate-key records: {duplicate_count}")
print(f"Quarantined records: {quarantine_count}")
print(f"Invalid ZIP prefixes: {invalid_zip_count}")
print(f"Missing ZIP prefixes: {missing_zip_count}")
print(f"Final clean records: {final_count}")

print("\nCUSTOMER STATE DISTRIBUTION")

clean_customers.groupBy("customer_state").count().orderBy(
    F.desc("count")
).show(truncate=False)

print("\nFINAL SCHEMA")
clean_customers.printSchema()

# 10. SAVE SILVER DATA

(
    clean_customers.write
    .mode("overwrite")
    .parquet(str(SILVER_PATH))
)

print("\nSilver customers dataset saved successfully!")
print(f"Output: {SILVER_PATH}")

spark.stop()

print("\nCUSTOMERS CLEANING COMPLETED")
