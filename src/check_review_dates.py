
from pathlib import Path

from pyspark.sql import SparkSession
from pyspark.sql import functions as F

BASE_DIR = Path(__file__).resolve().parent.parent

spark = (
    SparkSession.builder
    .appName("CheckReviewDates")
    .master("local[*]")
    .config("spark.driver.host", "127.0.0.1")
    .config("spark.driver.bindAddress", "127.0.0.1")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("WARN")

reviews = spark.read.parquet(
    str(BASE_DIR / "data/silver/olist_order_reviews_dataset")
)

dates = spark.read.parquet(
    str(BASE_DIR / "data/gold/dim_date")
)

# Convert review timestamps into dates
review_dates = reviews.withColumn(
    "review_date",
    F.to_date("review_creation_date")
)

# Show review date statistics
print("\nREVIEW DATE STATISTICS")

review_dates.select(
    F.min("review_date").alias("minimum_review_date"),
    F.max("review_date").alias("maximum_review_date"),
    F.count("*").alias("total_reviews"),
    F.sum(
        F.when(F.col("review_date").isNull(), 1).otherwise(0)
    ).alias("null_review_dates")
).show(truncate=False)


# Show date dimension range
print("\nDATE DIMENSION RANGE")

dates.select(
    F.min("full_date").alias("minimum_dimension_date"),
    F.max("full_date").alias("maximum_dimension_date"),
    F.count("*").alias("total_dimension_dates")
).show(truncate=False)


# Find review dates missing from dim_date
valid_review_dates = (
    review_dates
    .filter(F.col("review_date").isNotNull())
    .select("review_date")
    .distinct()
)

missing_dates = valid_review_dates.join(
    dates.select(
        F.col("full_date").alias("review_date")
    ),
    on="review_date",
    how="left_anti"
)

print("\nMISSING DISTINCT REVIEW DATES")

print("Missing distinct dates:", missing_dates.count())

missing_dates.orderBy("review_date").show(
    30,
    truncate=False
)

spark.stop()
