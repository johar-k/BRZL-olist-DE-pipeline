
from pathlib import Path

from pyspark.sql import SparkSession
from pyspark.sql import functions as F

# 1. PROJECT CONFIGURATION

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw" / "olist"


# 2. CREATE SPARK SESSION

spark = (
    SparkSession.builder
    .appName("OlistDataDiscovery")
    .master("local[*]")
    .config("spark.driver.host", "127.0.0.1")
    .config("spark.driver.bindAddress", "127.0.0.1")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("ERROR")


# 3. FIND CSV FILES

csv_files = sorted(RAW_DATA_DIR.glob("*.csv"))

if not csv_files:
    raise FileNotFoundError(
        f"No CSV files found in {RAW_DATA_DIR}"
    )

print("\n" + "=" * 65)
print("OLIST DATA DISCOVERY REPORT")
print("=" * 65)

print(f"Total CSV files found: {len(csv_files)}")

# 4. PROFILE EACH DATASET

for file_path in csv_files:

    print("\n" + "=" * 65)
    print(f"DATASET: {file_path.name}")
    print("=" * 65)

    # Read CSV as strings for initial discovery.
    df = (
        spark.read
        .option("header", True)
        .option("inferSchema", False)
        .option("encoding", "UTF-8")
        .option("mode", "PERMISSIVE")
        .csv(str(file_path))
    )

    # Display columns
    print("\nCOLUMNS:")
    for column in df.columns:
        print(f" - {column}")

    # Display schema
    print("\nSCHEMA:")
    df.printSchema()

    # Calculate missing and empty values.
    profile_expressions = [
        F.count("*").alias("total_records")
    ]

    for index, column in enumerate(df.columns):

        missing_condition = (
            F.col(column).isNull()
            | (F.trim(F.col(column)) == "")
        )

        profile_expressions.append(
            F.sum(
                F.when(missing_condition, 1).otherwise(0)
            ).alias(f"missing_{index}")
        )

    # One aggregate operation per dataset.
    profile = df.agg(*profile_expressions).first()

    total_records = profile["total_records"]

    print("\nDATASET STATISTICS:")
    print(f"Total records: {total_records}")
    print(f"Total columns: {len(df.columns)}")
    print(f"File size: {file_path.stat().st_size / (1024 * 1024):.2f} MB")

    print("\nMISSING / EMPTY VALUES:")

    for index, column in enumerate(df.columns):

        missing_count = profile[f"missing_{index}"]

        percentage = (
            (missing_count / total_records) * 100
            if total_records else 0
        )

        print(
            f"{column}: {missing_count} "
            f"({percentage:.2f}%)"
        )

# 5. STOP SPARK


spark.stop()

print("\n" + "=" * 65)
print("DATA DISCOVERY COMPLETED SUCCESSFULLY")
print("=" * 65)
