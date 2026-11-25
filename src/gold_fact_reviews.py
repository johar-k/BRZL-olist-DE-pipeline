
from pathlib import Path

from pyspark.sql import SparkSession
from pyspark.sql import functions as F


# 1. PROJECT PATHS

BASE_DIR = Path(__file__).resolve().parent.parent

SILVER_DIR = BASE_DIR / "data" / "silver"
GOLD_DIR = BASE_DIR / "data" / "gold"

REVIEWS_PATH = SILVER_DIR / "olist_order_reviews_dataset"
ORDERS_PATH = SILVER_DIR / "olist_orders_enriched"

CUSTOMER_DIM_PATH = GOLD_DIR / "dim_customer"
DATE_DIM_PATH = GOLD_DIR / "dim_date"

OUTPUT_PATH = GOLD_DIR / "fact_reviews"


# 2. CREATE SPARK SESSION

spark = (
    SparkSession.builder
    .appName("OlistGoldFactReviews")
    .master("local[*]")
    .config("spark.driver.host", "127.0.0.1")
    .config("spark.driver.bindAddress", "127.0.0.1")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("WARN")


# 3. READ SOURCE DATA AND DIMENSIONS

print("\nReading Silver data and Gold dimensions...")

reviews = spark.read.parquet(str(REVIEWS_PATH))
orders = spark.read.parquet(str(ORDERS_PATH))

customers = spark.read.parquet(str(CUSTOMER_DIM_PATH))
dates = spark.read.parquet(str(DATE_DIM_PATH))

source_count = reviews.count()

print(f"Source review records: {source_count}")


# 4. ATTACH ORDER AND CUSTOMER INFORMATION

print("\nJoining reviews with order information...")

review_data = reviews.join(
    orders.select(
        "order_id",
        "customer_id"
    ),
    on="order_id",
    how="left"
)

# 5. MAP CUSTOMER SURROGATE KEY

print("\nMapping customer keys...")

review_data = review_data.join(
    customers.select(
        "customer_id",
        "customer_key"
    ),
    on="customer_id",
    how="left"
)


# 6. MAP REVIEW CREATION DATE

print("\nMapping review dates...")

review_data = review_data.withColumn(
    "_review_date",
    F.to_date("review_creation_date")
)

review_data = review_data.join(
    dates.select(
        "full_date",
        "date_key"
    ),
    review_data["_review_date"] == dates["full_date"],
    how="left"
).drop("full_date")


# 7. CHECK DIMENSION MATCHES

print("\nChecking dimension key matches...")

missing_orders = review_data.filter(
    F.col("customer_id").isNull()
).count()

missing_customer_keys = review_data.filter(
    F.col("customer_key").isNull()
).count()

missing_date_keys = review_data.filter(
    F.col("date_key").isNull()
).count()

print(f"Reviews without matching orders: {missing_orders}")
print(f"Missing customer keys: {missing_customer_keys}")
print(f"Missing review date keys: {missing_date_keys}")

if (
    missing_orders > 0
    or missing_customer_keys > 0
    or missing_date_keys > 0
):
    raise ValueError(
        "Some reviews could not be mapped to their order, "
        "customer, or review date."
    )


# 8. BUILD FACT REVIEWS

print("\nBuilding fact_reviews...")

fact_reviews = review_data.select(
    "review_id",
    "order_id",

    "customer_key",
    "date_key",

    F.col("review_score")
        .cast("integer")
        .alias("review_score"),

    "review_comment_title",
    "review_comment_message",

    "review_answer_timestamp"
)


# 9. DATA QUALITY CHECKS

print("\nRunning fact_reviews quality checks...")

fact_count = fact_reviews.count()

duplicate_reviews = (
    fact_reviews
    .groupBy("review_id")
    .count()
    .filter(F.col("count") > 1)
    .count()
)

invalid_scores = fact_reviews.filter(
    F.col("review_score").isNull()
    | (F.col("review_score") < 1)
    | (F.col("review_score") > 5)
).count()

null_keys = fact_reviews.filter(
    F.col("customer_key").isNull()
    | F.col("date_key").isNull()
).count()

print(f"Source review count: {source_count}")
print(f"Fact review count: {fact_count}")
print(f"Duplicate review IDs: {duplicate_reviews}")
print(f"Invalid review scores: {invalid_scores}")
print(f"Missing dimension keys: {null_keys}")

if source_count != fact_count:
    raise ValueError("Review record count changed during transformation!")

if duplicate_reviews > 0:
    raise ValueError("Duplicate review IDs found!")

if invalid_scores > 0:
    raise ValueError("Invalid review scores found!")

if null_keys > 0:
    raise ValueError("Missing review dimension keys found!")


# 10. WRITE GOLD FACT TABLE

print("\nWriting fact_reviews to Gold...")

(
    fact_reviews
    .write
    .mode("overwrite")
    .parquet(str(OUTPUT_PATH))
)


# 11. DISPLAY RESULTS

print("\nFact reviews sample:")

fact_reviews.show(10, truncate=False)

print("\nFact reviews schema:")

fact_reviews.printSchema()

print(f"\nTotal fact_reviews records: {fact_count}")

print("\nGold fact_reviews created successfully!")


# 12. STOP SPARK

spark.stop()
