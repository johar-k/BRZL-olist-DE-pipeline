
from pathlib import Path

from pyspark.sql import SparkSession
from pyspark.sql import functions as F

BASE_DIR = Path(__file__).resolve().parent.parent

spark = (
    SparkSession.builder
    .appName("InspectNullReviewDates")
    .master("local[*]")
    .config("spark.driver.host", "127.0.0.1")
    .config("spark.driver.bindAddress", "127.0.0.1")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("WARN")

silver_path = BASE_DIR / "data/silver/olist_order_reviews_dataset"
bronze_path = BASE_DIR / "data/bronze/olist_order_reviews_dataset"

silver = spark.read.parquet(str(silver_path))
bronze = spark.read.parquet(str(bronze_path))

print("\nSILVER REVIEW SCHEMA")
silver.printSchema()

print("\nSILVER NULL REVIEW DATE COUNT")

silver.filter(
    F.col("review_creation_date").isNull()
).select(
    "review_id",
    "order_id",
    "review_creation_date",
    "review_answer_timestamp"
).show(15, truncate=False)

print("\nBRONZE NULL REVIEW DATE COUNT")

bronze.filter(
    F.col("review_creation_date").isNull()
    | (F.trim(F.col("review_creation_date")) == "")
).select(
    "review_id",
    "order_id",
    "review_creation_date",
    "review_answer_timestamp"
).show(15, truncate=False)

print("\nBRONZE AND SILVER COUNTS")

print("Bronze records:", bronze.count())
print("Silver records:", silver.count())

print(
    "Silver null review dates:",
    silver.filter(
        F.col("review_creation_date").isNull()
    ).count()
)

spark.stop()
