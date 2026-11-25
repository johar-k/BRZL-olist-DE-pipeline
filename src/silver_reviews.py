
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
    / "olist_order_reviews_dataset"
)

SILVER_PATH = (
    PROJECT_ROOT
    / "data"
    / "silver"
    / "olist_order_reviews_dataset"
)

QUARANTINE_PATH = (
    PROJECT_ROOT
    / "data"
    / "silver"
    / "quarantine"
    / "reviews"
)

ORDERS_PATH = (
    PROJECT_ROOT
    / "data"
    / "silver"
    / "olist_orders_dataset"
)

# 2. CREATE SPARK SESSION

spark = (
    SparkSession.builder
    .appName("OlistSilverReviews")
    .master("local[*]")
    .config("spark.driver.host", "127.0.0.1")
    .config("spark.driver.bindAddress", "127.0.0.1")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("ERROR")

# 3. READ BRONZE DATA

reviews = spark.read.parquet(str(BRONZE_PATH))

original_count = reviews.count()

print("\nREVIEWS DATA CLEANING")
print("=" * 60)

print(f"Original records: {original_count}")

# 4. CLEAN STRING COLUMNS

string_columns = [
    "review_id",
    "order_id",
    "review_comment_title",
    "review_comment_message",
]

for column in string_columns:

    reviews = reviews.withColumn(
        column,
        F.when(
            F.trim(F.col(column)) == "",
            None
        ).otherwise(F.trim(F.col(column)))
    )

# 5. CONVERT DATA TYPES

reviews = (
    reviews
    .withColumn(
        "review_score",
        F.expr("try_cast(review_score AS INT)")
    )
    .withColumn(
        "review_creation_date",
        F.expr(
            "try_to_timestamp(review_creation_date, "
            "'yyyy-MM-dd HH:mm:ss')"
        )
    )
    .withColumn(
        "review_answer_timestamp",
        F.expr(
            "try_to_timestamp(review_answer_timestamp, "
            "'yyyy-MM-dd HH:mm:ss')"
        )
    )
)

# 6. VALIDATE REQUIRED FIELDS

invalid_required = reviews.filter(
    F.col("review_id").isNull()
    | F.col("order_id").isNull()
    | F.col("review_score").isNull()
)

valid_reviews = reviews.filter(
    F.col("review_id").isNotNull()
    & F.col("order_id").isNotNull()
    & F.col("review_score").isNotNull()
)

invalid_required_count = invalid_required.count()

# 7. VALIDATE REVIEW SCORES

invalid_scores = valid_reviews.filter(
    (F.col("review_score") < 1)
    | (F.col("review_score") > 5)
)

valid_scores = valid_reviews.filter(
    (F.col("review_score") >= 1)
    & (F.col("review_score") <= 5)
)

invalid_score_count = invalid_scores.count()

# 8. DETECT DUPLICATE REVIEW IDS

window_spec = Window.partitionBy("review_id")

reviews_with_counts = valid_scores.withColumn(
    "_review_count",
    F.count("*").over(window_spec)
)

duplicate_reviews = reviews_with_counts.filter(
    F.col("_review_count") > 1
).drop("_review_count")

unique_reviews = reviews_with_counts.filter(
    F.col("_review_count") == 1
).drop("_review_count")

duplicate_count = duplicate_reviews.count()

# 9. VALIDATE ORDER REFERENCES

orders = (
    spark.read.parquet(str(ORDERS_PATH))
    .select("order_id")
    .distinct()
    .withColumn("_order_exists", F.lit(True))
)

checked_reviews = unique_reviews.join(
    orders,
    on="order_id",
    how="left"
)

orphan_reviews = checked_reviews.filter(
    F.col("_order_exists").isNull()
)

clean_reviews = checked_reviews.filter(
    F.col("_order_exists").isNotNull()
)

orphan_count = orphan_reviews.count()

# 10. REMOVE TEMPORARY COLUMNS

clean_reviews = clean_reviews.drop("_order_exists")
orphan_reviews = orphan_reviews.drop("_order_exists")

# 11. QUARANTINE INVALID RECORDS

quarantined = (
    invalid_required
    .unionByName(invalid_scores)
    .unionByName(duplicate_reviews)
    .unionByName(orphan_reviews)
)

quarantine_count = quarantined.count()

if quarantine_count > 0:

    (
        quarantined.write
        .mode("overwrite")
        .parquet(str(QUARANTINE_PATH))
    )

# 12. DATA QUALITY REPORT

final_count = clean_reviews.count()

print("\nREVIEWS DATA QUALITY REPORT")
print("-" * 60)

print(f"Original records: {original_count}")
print(f"Invalid required records: {invalid_required_count}")
print(f"Invalid review scores: {invalid_score_count}")
print(f"Duplicate review IDs: {duplicate_count}")
print(f"Orphan review records: {orphan_count}")
print(f"Total quarantined records: {quarantine_count}")
print(f"Final clean records: {final_count}")

print("\nREVIEW SCORE DISTRIBUTION")

clean_reviews.groupBy("review_score").count().orderBy(
    "review_score"
).show(truncate=False)

print("\nAVERAGE REVIEW SCORE")

clean_reviews.select(
    F.avg("review_score").alias("average_review_score")
).show()

print("\nFINAL SCHEMA")
clean_reviews.printSchema()

# 13. SAVE SILVER DATA

(
    clean_reviews.write
    .mode("overwrite")
    .parquet(str(SILVER_PATH))
)

print("\nSilver reviews dataset saved successfully!")
print(f"Output: {SILVER_PATH}")

spark.stop()

print("\nREVIEWS CLEANING COMPLETED")
