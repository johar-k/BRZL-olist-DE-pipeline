
from pathlib import Path
from datetime import datetime, timezone

from pyspark.sql import SparkSession
from pyspark.sql import functions as F

# 1. PROJECT PATHS

PROJECT_ROOT = Path(__file__).resolve().parent.parent

RAW_DIR = PROJECT_ROOT / "data" / "raw" / "olist"
BRONZE_DIR = PROJECT_ROOT / "data" / "bronze"

# 2. CREATE SPARK SESSION

spark = (
    SparkSession.builder
    .appName("OlistBronzeIngestion")
    .master("local[*]")
    .config("spark.driver.host", "127.0.0.1")
    .config("spark.driver.bindAddress", "127.0.0.1")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("ERROR")

# 3. INGEST CSV FILES INTO BRONZE

csv_files = sorted(RAW_DIR.glob("*.csv"))

if not csv_files:
    raise FileNotFoundError(f"No CSV files found in {RAW_DIR}")

print("\nSTARTING BRONZE INGESTION")
print("=" * 60)

ingestion_time = datetime.now(timezone.utc).isoformat()

for file_path in csv_files:

    table_name = file_path.stem
    output_path = BRONZE_DIR / table_name

    print(f"\nProcessing: {table_name}")

    # Read source CSV
    df = (
        spark.read
        .option("header", True)
        .option("inferSchema", False)
        .option("encoding", "UTF-8")
        .option("mode", "PERMISSIVE")
        .csv(str(file_path))
    )

    # Add ingestion metadata
    bronze_df = (
        df
        .withColumn("_source_file", F.lit(file_path.name))
        .withColumn("_ingested_at", F.lit(ingestion_time))
    )

    # Save as Parquet
    (
        bronze_df.write
        .mode("overwrite")
        .parquet(str(output_path))
    )

    # Report
    row_count = bronze_df.count()

    print(f"Source file: {file_path.name}")
    print(f"Records ingested: {row_count}")
    print(f"Columns: {len(bronze_df.columns)}")
    print(f"Output: {output_path}")
    print("Status: SUCCESS")

print("\n" + "=" * 60)
print("BRONZE INGESTION COMPLETED")
print("=" * 60)

spark.stop()
