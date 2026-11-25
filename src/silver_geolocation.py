
from pathlib import Path

from pyspark.sql import SparkSession
from pyspark.sql import functions as F

# 1. PROJECT PATHS

PROJECT_ROOT = Path(__file__).resolve().parent.parent

BRONZE_PATH = (
    PROJECT_ROOT
    / "data"
    / "bronze"
    / "olist_geolocation_dataset"
)

SILVER_PATH = (
    PROJECT_ROOT
    / "data"
    / "silver"
    / "olist_geolocation_dataset"
)

QUARANTINE_PATH = (
    PROJECT_ROOT
    / "data"
    / "silver"
    / "quarantine"
    / "geolocation"
)

# 2. CREATE SPARK SESSION

spark = (
    SparkSession.builder
    .appName("OlistSilverGeolocation")
    .master("local[*]")
    .config("spark.driver.host", "127.0.0.1")
    .config("spark.driver.bindAddress", "127.0.0.1")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("ERROR")

# 3. READ BRONZE DATA

geo = spark.read.parquet(str(BRONZE_PATH))

original_count = geo.count()

print("\nGEOLOCATION DATA CLEANING")
print("=" * 65)
print(f"Original records: {original_count}")

# 4. CLEAN STRING COLUMNS

string_columns = [
    "geolocation_zip_code_prefix",
    "geolocation_city",
    "geolocation_state",
]

for column in string_columns:

    geo = geo.withColumn(
        column,
        F.when(
            F.trim(F.col(column)) == "",
            None
        ).otherwise(F.trim(F.col(column)))
    )

geo = (
    geo
    .withColumn(
        "geolocation_city",
        F.lower(F.col("geolocation_city"))
    )
    .withColumn(
        "geolocation_state",
        F.upper(F.col("geolocation_state"))
    )
)

# 5. CONVERT COORDINATES

geo = (
    geo
    .withColumn(
        "geolocation_lat",
        F.expr("try_cast(geolocation_lat AS DOUBLE)")
    )
    .withColumn(
        "geolocation_lng",
        F.expr("try_cast(geolocation_lng AS DOUBLE)")
    )
)

# 6. VALIDATE ZIP PREFIX

invalid_zip = geo.filter(
    F.col("geolocation_zip_code_prefix").isNull()
    | (~F.col("geolocation_zip_code_prefix").rlike(r"^[0-9]{5}$"))
)

valid_zip = geo.filter(
    F.col("geolocation_zip_code_prefix").rlike(r"^[0-9]{5}$")
)

invalid_zip_count = invalid_zip.count()

# 7. VALIDATE COORDINATES

invalid_coordinates = valid_zip.filter(
    F.col("geolocation_lat").isNull()
    | F.col("geolocation_lng").isNull()
    | (F.col("geolocation_lat") < -90)
    | (F.col("geolocation_lat") > 90)
    | (F.col("geolocation_lng") < -180)
    | (F.col("geolocation_lng") > 180)
)

valid_coordinates = valid_zip.filter(
    F.col("geolocation_lat").isNotNull()
    & F.col("geolocation_lng").isNotNull()
    & F.col("geolocation_lat").between(-90, 90)
    & F.col("geolocation_lng").between(-180, 180)
)

invalid_coordinate_count = invalid_coordinates.count()

# 8. REMOVE EXACT DUPLICATES

before_deduplication = valid_coordinates.count()

unique_geo = valid_coordinates.dropDuplicates([
    "geolocation_zip_code_prefix",
    "geolocation_lat",
    "geolocation_lng",
    "geolocation_city",
    "geolocation_state",
])

after_deduplication = unique_geo.count()

duplicate_count = before_deduplication - after_deduplication

# 9. AGGREGATE BY ZIP PREFIX

silver_geo = (
    unique_geo
    .groupBy("geolocation_zip_code_prefix")
    .agg(
        F.avg("geolocation_lat").alias("latitude"),
        F.avg("geolocation_lng").alias("longitude"),
        F.first("geolocation_city", ignorenulls=True).alias("city"),
        F.first("geolocation_state", ignorenulls=True).alias("state"),
        F.count("*").alias("coordinate_records")
    )
)
# 10. DATA QUALITY REPORT

final_count = silver_geo.count()

print("\nGEOLOCATION DATA QUALITY REPORT")
print("-" * 65)

print(f"Original records: {original_count}")
print(f"Invalid ZIP records: {invalid_zip_count}")
print(f"Invalid coordinate records: {invalid_coordinate_count}")
print(f"Exact duplicate records removed: {duplicate_count}")
print(f"Records after cleaning: {after_deduplication}")
print(f"Unique ZIP prefixes: {final_count}")

print("\nSAMPLE GEOLOCATION RECORDS")

silver_geo.show(10, truncate=False)

print("\nFINAL SCHEMA")
silver_geo.printSchema()

# 11. QUARANTINE INVALID RECORDS

quarantined = invalid_zip.unionByName(invalid_coordinates)

quarantine_count = quarantined.count()

if quarantine_count > 0:

    (
        quarantined.write
        .mode("overwrite")
        .parquet(str(QUARANTINE_PATH))
    )

# 12. SAVE SILVER DATA

(
    silver_geo.write
    .mode("overwrite")
    .parquet(str(SILVER_PATH))
)

print("\nSilver geolocation dataset saved successfully!")
print(f"Output: {SILVER_PATH}")

spark.stop()

print("\nGEOLOCATION CLEANING COMPLETED")
