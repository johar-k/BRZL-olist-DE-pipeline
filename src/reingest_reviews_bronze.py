
import csv
from pathlib import Path

from pyspark.sql import SparkSession
from pyspark.sql import functions as F


# 1. PATHS

BASE_DIR = Path(__file__).resolve().parent.parent

RAW_FILE = (
    BASE_DIR
    / "data/raw/olist/olist_order_reviews_dataset.csv"
)

STAGING_DIR = BASE_DIR / "data/bronze/_staging"
CLEAN_CSV = STAGING_DIR / "olist_order_reviews_dataset.csv"

BRONZE_OUTPUT = (
    BASE_DIR / "data/bronze/olist_order_reviews_dataset"
)


# 2. PARSE ORIGINAL CSV WITH PYTHON

print("\nReading original reviews CSV...")

STAGING_DIR.mkdir(parents=True, exist_ok=True)

record_count = 0

with open(
    RAW_FILE,
    mode="r",
    encoding="utf-8-sig",
    newline=""
) as source:

    reader = csv.DictReader(
        source,
        delimiter=",",
        quotechar='"',
        doublequote=True,
        strict=True
    )

    headers = reader.fieldnames

    if not headers:
        raise ValueError("CSV headers could not be read!")

    print("CSV columns:", headers)

    with open(
        CLEAN_CSV,
        mode="w",
        encoding="utf-8",
        newline=""
    ) as destination:

        writer = csv.DictWriter(
            destination,
            fieldnames=headers,
            delimiter=",",
            quotechar='"',
            quoting=csv.QUOTE_MINIMAL,
            lineterminator="\n"
        )

        writer.writeheader()

        for row in reader:

            # Detect rows with extra columns.
            if None in row:
                raise ValueError(
                    f"Malformed CSV record near line {reader.line_num}"
                )

            # Normalize multiline text fields.
            cleaned_row = {}

            for column, value in row.items():

                if value is None:
                    cleaned_row[column] = ""
                else:
                    cleaned_row[column] = (
                        value
                        .replace("\r\n", " ")
                        .replace("\n", " ")
                        .replace("\r", " ")
                    )

            writer.writerow(cleaned_row)

            record_count += 1

print(f"Successfully parsed records: {record_count}")


# 3. CREATE SPARK SESSION

spark = (
    SparkSession.builder
    .appName("ReingestReviewsBronze")
    .master("local[*]")
    .config("spark.driver.host", "127.0.0.1")
    .config("spark.driver.bindAddress", "127.0.0.1")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("WARN")


# 4. READ CLEANED CSV

print("\nReading normalized CSV with Spark...")

reviews = (
    spark.read
    .option("header", True)
    .option("inferSchema", False)
    .option("mode", "FAILFAST")
    .option("quote", '"')
    .option("escape", '"')
    .csv(str(CLEAN_CSV))
)

spark_count = reviews.count()

print(f"Spark records: {spark_count}")

if spark_count != record_count:
    raise ValueError(
        "Python and Spark record counts do not match!"
    )


# 5. VALIDATE COLUMN ALIGNMENT

print("\nChecking review data...")

reviews.select(
    "review_id",
    "order_id",
    "review_score",
    "review_creation_date",
    "review_answer_timestamp"
).show(10, truncate=False)

invalid_review_ids = reviews.filter(
    F.col("review_id").isNull()
    | (F.length(F.trim("review_id")) != 32)
).count()

invalid_order_ids = reviews.filter(
    F.col("order_id").isNull()
    | (F.length(F.trim("order_id")) != 32)
).count()

print(f"Invalid review IDs: {invalid_review_ids}")
print(f"Invalid order IDs: {invalid_order_ids}")

if invalid_review_ids > 0 or invalid_order_ids > 0:
    raise ValueError("CSV column alignment validation failed!")


# 6. WRITE CORRECTED BRONZE DATA

print("\nWriting corrected Bronze reviews...")

reviews = (
    reviews
    .withColumn(
        "_source_file",
        F.lit("olist_order_reviews_dataset.csv")
    )
    .withColumn(
        "_ingested_at",
        F.current_timestamp().cast("string")
    )
)

(
    reviews.write
    .mode("overwrite")
    .parquet(str(BRONZE_OUTPUT))
)

print("\nCorrected Bronze reviews created successfully!")

spark.stop()
