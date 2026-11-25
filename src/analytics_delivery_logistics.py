
from pyspark.sql import SparkSession

from pyspark.sql.functions import (
    col,
    count,
    countDistinct,
    avg,
    round,
    sum,
    when,
    desc,
    try_divide
)

# 1. Create Spark Session

spark = (
    SparkSession.builder
    .appName("Olist-Delivery-Logistics-Analytics")
    .master("local[*]")
    .config("spark.driver.host", "127.0.0.1")
    .config("spark.driver.bindAddress", "127.0.0.1")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("ERROR")

BASE = "data/gold"
OUTPUT = "output/analytics"

# 2. Read Gold Tables

delivery = spark.read.parquet(
    f"{BASE}/fact_delivery"
)

dates = spark.read.parquet(
    f"{BASE}/dim_date"
)

customers = spark.read.parquet(
    f"{BASE}/dim_customer"
)

print("\n========== DELIVERY TABLE SCHEMA ==========")
delivery.printSchema()

# 3. Overall Delivery Performance

overall_delivery = (
    delivery
    .agg(
        count("*").alias("total_orders"),

        count(
            when(col("is_delivered") == 1, 1)
        ).alias("delivered_orders"),

        count(
            when(col("is_delivered") == 0, 1)
        ).alias("undelivered_orders"),

        count(
            when(col("is_late") == 1, 1)
        ).alias("late_orders"),

        round(
            avg(
                when(
                    col("is_delivered") == 1,
                    col("delivery_days")
                )
            ),
            2
        ).alias("average_delivery_days"),

        round(
            avg(
                when(
                    col("is_delivered") == 1,
                    col("delivery_delay_days")
                )
            ),
            2
        ).alias("average_delivery_delay_days")
    )
    .withColumn(
        "late_delivery_percentage",
        round(
            try_divide(
                col("late_orders") * 100,
                col("delivered_orders")
            ),
            2
        )
    )
)

print("\n========== OVERALL DELIVERY PERFORMANCE ==========")
overall_delivery.show(truncate=False)

# 4. Delivery Status Analysis

delivery_status = (
    delivery
    .withColumn(
        "delivery_category",
        when(col("is_delivered") == 0, "Undelivered")
        .when(col("is_late") == 1, "Late")
        .otherwise("On Time or Early")
    )
    .groupBy("delivery_category")
    .agg(
        count("*").alias("total_orders")
    )
    .orderBy(desc("total_orders"))
)

print("\n========== DELIVERY STATUS ANALYSIS ==========")
delivery_status.show(truncate=False)

# 5. Delivery Duration Analysis

delivery_duration = (
    delivery
    .filter(col("is_delivered") == 1)
    .agg(
        round(avg("delivery_days"), 2).alias("average_delivery_days"),

        round(
            avg("estimated_delivery_days"), 2
        ).alias("average_estimated_days"),

        round(
            avg("delivery_delay_days"), 2
        ).alias("average_delay_days"),

        round(
            avg("delivery_days") - avg("estimated_delivery_days"),
            2
        ).alias("average_actual_vs_estimated_days")
    )
)

print("\n========== DELIVERY DURATION ANALYSIS ==========")
delivery_duration.show(truncate=False)

# 6. Delivery Performance by Customer State

delivery_by_state = (
    delivery
    .join(
        customers.select(
            "customer_key",
            "customer_city",
            "customer_state"
        ),
        on="customer_key",
        how="left"
    )
    .groupBy("customer_state")
    .agg(
        countDistinct("order_id").alias("total_orders"),

        count(
            when(col("is_delivered") == 1, 1)
        ).alias("delivered_orders"),

        count(
            when(col("is_late") == 1, 1)
        ).alias("late_orders"),

        round(
            avg(
                when(
                    col("is_delivered") == 1,
                    col("delivery_days")
                )
            ),
            2
        ).alias("average_delivery_days"),

        round(
            avg(
                when(
                    col("is_delivered") == 1,
                    col("delivery_delay_days")
                )
            ),
            2
        ).alias("average_delay_days")
    )
    .withColumn(
        "late_delivery_percentage",
        round(
            try_divide(
                col("late_orders") * 100,
                col("delivered_orders")
            ),
            2
        )
    )
    .orderBy(desc("late_delivery_percentage"))
)

print("\n========== DELIVERY PERFORMANCE BY STATE ==========")
delivery_by_state.show(30, truncate=False)

# 7. Monthly Delivery Trends
# Use purchase date as the reporting month.

monthly_delivery = (
    delivery
    .join(
        dates.select(
            col("date_key").alias("purchase_date_key"),
            "year",
            "month",
            "month_name"
        ),
        on="purchase_date_key",
        how="left"
    )
    .groupBy(
        "year",
        "month",
        "month_name"
    )
    .agg(
        count("*").alias("total_orders"),

        count(
            when(col("is_delivered") == 1, 1)
        ).alias("delivered_orders"),

        count(
            when(col("is_late") == 1, 1)
        ).alias("late_orders"),

        round(
            avg(
                when(
                    col("is_delivered") == 1,
                    col("delivery_days")
                )
            ),
            2
        ).alias("average_delivery_days")
    )
    .withColumn(
        "late_delivery_percentage",
        round(
            try_divide(
                col("late_orders") * 100,
                col("delivered_orders")
            ),
            2
        )
    )
    .orderBy("year", "month")
)

print("\n========== MONTHLY DELIVERY TRENDS ==========")
monthly_delivery.show(100, truncate=False)

# 8. Orders With Longest Delivery Delays

longest_delays = (
    delivery
    .filter(
        (col("is_delivered") == 1)
        & (col("is_late") == 1)
    )
    .select(
        "order_id",
        "delivery_days",
        "estimated_delivery_days",
        "delivery_delay_days"
    )
    .orderBy(desc("delivery_delay_days"))
)

print("\n========== TOP 30 LONGEST DELIVERY DELAYS ==========")
longest_delays.show(30, truncate=False)

# 9. Save Analytics Results

overall_delivery.write.mode("overwrite").parquet(
    f"{OUTPUT}/overall_delivery"
)

delivery_status.write.mode("overwrite").parquet(
    f"{OUTPUT}/delivery_status"
)

delivery_duration.write.mode("overwrite").parquet(
    f"{OUTPUT}/delivery_duration"
)

delivery_by_state.write.mode("overwrite").parquet(
    f"{OUTPUT}/delivery_by_state"
)

monthly_delivery.write.mode("overwrite").parquet(
    f"{OUTPUT}/monthly_delivery"
)

longest_delays.write.mode("overwrite").parquet(
    f"{OUTPUT}/longest_delivery_delays"
)

print("\nDelivery and logistics analytics completed successfully!")

spark.stop()
