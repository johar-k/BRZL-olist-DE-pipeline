
from pyspark.sql import SparkSession

from pyspark.sql.functions import (
    col,
    count,
    countDistinct,
    avg,
    round,
    when,
    desc,
    length
)

# 1. Spark Session

spark = (
    SparkSession.builder
    .appName("Olist-Customer-Review-Analytics")
    .master("local[*]")
    .config("spark.driver.host", "127.0.0.1")
    .config("spark.driver.bindAddress", "127.0.0.1")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("ERROR")

BASE = "data/gold"
OUTPUT = "output/analytics"

# 2. Read Gold Tables

reviews = spark.read.parquet(
    f"{BASE}/fact_reviews"
)

dates = spark.read.parquet(
    f"{BASE}/dim_date"
)

# 3. Overall Customer Satisfaction

overall_satisfaction = (
    reviews
    .agg(
        count("*").alias("total_reviews"),

        countDistinct("order_id").alias("reviewed_orders"),

        round(avg("review_score"), 2).alias("average_review_score"),

        count(
            when(col("review_score") >= 4, 1)
        ).alias("positive_reviews"),

        count(
            when(col("review_score") == 3, 1)
        ).alias("neutral_reviews"),

        count(
            when(col("review_score") <= 2, 1)
        ).alias("negative_reviews")
    )
    .withColumn(
        "positive_review_percentage",
        round(
            col("positive_reviews") * 100 / col("total_reviews"),
            2
        )
    )
    .withColumn(
        "negative_review_percentage",
        round(
            col("negative_reviews") * 100 / col("total_reviews"),
            2
        )
    )
)

print("\n========== OVERALL CUSTOMER SATISFACTION ==========")
overall_satisfaction.show(truncate=False)

# 4. Review Score Distribution

score_distribution = (
    reviews
    .groupBy("review_score")
    .agg(
        count("*").alias("total_reviews"),

        countDistinct("order_id").alias("reviewed_orders")
    )
    .orderBy("review_score")
)

print("\n========== REVIEW SCORE DISTRIBUTION ==========")
score_distribution.show(truncate=False)

# 5. Positive, Neutral and Negative Reviews

review_categories = (
    reviews
    .withColumn(
        "review_category",
        when(col("review_score") >= 4, "Positive")
        .when(col("review_score") == 3, "Neutral")
        .otherwise("Negative")
    )
    .groupBy("review_category")
    .agg(
        count("*").alias("total_reviews"),

        round(avg("review_score"), 2).alias("average_score")
    )
    .orderBy(desc("total_reviews"))
)

print("\n========== REVIEW CATEGORY ANALYSIS ==========")
review_categories.show(truncate=False)

# 6. Monthly Review Trends

monthly_reviews = (
    reviews
    .join(
        dates.select(
            "date_key",
            "year",
            "month",
            "month_name"
        ),
        on="date_key",
        how="inner"
    )
    .groupBy(
        "year",
        "month",
        "month_name"
    )
    .agg(
        count("*").alias("total_reviews"),

        round(avg("review_score"), 2).alias("average_review_score"),

        count(
            when(col("review_score") <= 2, 1)
        ).alias("negative_reviews"),

        count(
            when(col("review_score") >= 4, 1)
        ).alias("positive_reviews")
    )
    .orderBy("year", "month")
)

print("\n========== MONTHLY REVIEW TRENDS ==========")
monthly_reviews.show(100, truncate=False)

# 7. Seller Response Analytics

review_responses = (
    reviews
    .agg(
        count("*").alias("total_reviews"),

        count(
            when(col("review_answer_timestamp").isNotNull(), 1)
        ).alias("reviews_with_response"),

        count(
            when(col("review_answer_timestamp").isNull(), 1)
        ).alias("reviews_without_response")
    )
    .withColumn(
        "response_percentage",
        round(
            col("reviews_with_response") * 100 / col("total_reviews"),
            2
        )
    )
)

print("\n========== REVIEW RESPONSE ANALYTICS ==========")
review_responses.show(truncate=False)

# 8. Low-Rating Reviews

low_rating_reviews = (
    reviews
    .filter(col("review_score") <= 2)
    .select(
        "review_id",
        "order_id",
        "review_score",
        "review_comment_title",
        "review_comment_message",
        "review_answer_timestamp"
    )
    .orderBy("review_score")
)

print("\n========== LOW-RATING REVIEWS ==========")
low_rating_reviews.show(30, truncate=False)

# 9. Review Comment Analytics

comment_analytics = (
    reviews
    .agg(
        count(
            when(
                col("review_comment_message").isNotNull()
                & (length(col("review_comment_message")) > 0),
                1
            )
        ).alias("reviews_with_comments"),

        count(
            when(
                col("review_comment_message").isNull()
                | (length(col("review_comment_message")) == 0),
                1
            )
        ).alias("reviews_without_comments")
    )
)

print("\n========== REVIEW COMMENT ANALYTICS ==========")
comment_analytics.show(truncate=False)

# 10. Save Analytics Results

overall_satisfaction.write.mode("overwrite").parquet(
    f"{OUTPUT}/overall_satisfaction"
)

score_distribution.write.mode("overwrite").parquet(
    f"{OUTPUT}/review_score_distribution"
)

review_categories.write.mode("overwrite").parquet(
    f"{OUTPUT}/review_categories"
)

monthly_reviews.write.mode("overwrite").parquet(
    f"{OUTPUT}/monthly_reviews"
)

review_responses.write.mode("overwrite").parquet(
    f"{OUTPUT}/review_responses"
)

low_rating_reviews.write.mode("overwrite").parquet(
    f"{OUTPUT}/low_rating_reviews"
)

comment_analytics.write.mode("overwrite").parquet(
    f"{OUTPUT}/review_comment_analytics"
)

print("\nCustomer review analytics completed successfully!")

spark.stop()
