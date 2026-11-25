
from pyspark.sql import SparkSession

from pyspark.sql.functions import (
    col,
    count,
    sum,
    avg,
    countDistinct,
    round,
    row_number,
    rank,
    dense_rank,
    lag,
    lead,
    when
)

from pyspark.sql.window import Window


# 1. Create Spark Session

spark = (
    SparkSession.builder
    .appName("Olist-Advanced-Window-Analytics")
    .master("local[*]")
    .config("spark.driver.host", "127.0.0.1")
    .config("spark.driver.bindAddress", "127.0.0.1")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("ERROR")

BASE = "data/gold"
OUTPUT = "output/analytics"


# 2. Read Gold Tables

sales = spark.read.parquet(
    f"{BASE}/fact_sales"
)

dates = spark.read.parquet(
    f"{BASE}/dim_date"
)

products = spark.read.parquet(
    f"{BASE}/dim_product"
)

customers = spark.read.parquet(
    f"{BASE}/dim_customer"
)


# 3. Monthly Sales Data

monthly_sales = (
    sales
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
        round(sum("gross_item_value"), 2).alias("monthly_sales"),

        countDistinct("order_id").alias("total_orders")
    )
    .withColumn(
        "month_index",
        col("year") * 12 + col("month")
    )
)


# 4. Running Total of Monthly Sales

running_window = (
    Window
    .orderBy("year", "month")
    .rowsBetween(Window.unboundedPreceding, Window.currentRow)
)

monthly_running_sales = (
    monthly_sales
    .withColumn(
        "running_total_sales",
        round(sum("monthly_sales").over(running_window), 2)
    )
)

print("\n========== MONTHLY RUNNING SALES ==========")

monthly_running_sales.select(
    "year",
    "month_name",
    "monthly_sales",
    "running_total_sales"
).show(100, truncate=False)


# 5. Month-over-Month Sales Analysis

monthly_order_window = Window.orderBy("year", "month")

monthly_growth = (
    monthly_sales
    .withColumn(
        "previous_month_sales",
        lag("monthly_sales", 1).over(monthly_order_window)
    )
    .withColumn(
        "next_month_sales",
        lead("monthly_sales", 1).over(monthly_order_window)
    )
    .withColumn(
        "sales_difference",
        round(
            col("monthly_sales") - col("previous_month_sales"),
            2
        )
    )
    .withColumn(
        "sales_growth_percentage",
        round(
            (
                col("monthly_sales") - col("previous_month_sales")
            ) * 100 / when(
                col("previous_month_sales") != 0,
                col("previous_month_sales")
            ),
            2
        )
    )
)

print("\n========== MONTH-OVER-MONTH SALES GROWTH ==========")

monthly_growth.select(
    "year",
    "month_name",
    "monthly_sales",
    "previous_month_sales",
    "sales_difference",
    "sales_growth_percentage"
).show(100, truncate=False)


# 6. Three-Month Moving Average

moving_window = (
    Window
    .orderBy("year", "month")
    .rowsBetween(-2, Window.currentRow)
)

monthly_moving_average = (
    monthly_sales
    .withColumn(
        "three_month_moving_average",
        round(avg("monthly_sales").over(moving_window), 2)
    )
)

print("\n========== THREE-MONTH MOVING AVERAGE ==========")

monthly_moving_average.select(
    "year",
    "month_name",
    "monthly_sales",
    "three_month_moving_average"
).show(100, truncate=False)


# 7. Product Sales Ranking

product_sales = (
    sales
    .groupBy("product_key")
    .agg(
        round(sum("gross_item_value"), 2).alias("total_sales"),

        countDistinct("order_id").alias("total_orders"),

        count("*").alias("total_items")
    )
)

product_window = Window.orderBy(
    col("total_sales").desc()
)

product_ranking = (
    product_sales
    .withColumn(
        "row_number",
        row_number().over(product_window)
    )
    .withColumn(
        "rank",
        rank().over(product_window)
    )
    .withColumn(
        "dense_rank",
        dense_rank().over(product_window)
    )
    .join(
        products.select(
            "product_key",
            "product_category_name_english"
        ),
        on="product_key",
        how="left"
    )
    .orderBy("rank")
)

print("\n========== TOP 20 PRODUCTS BY SALES ==========")

product_ranking.select(
    "product_key",
    "product_category_name_english",
    "total_sales",
    "total_orders",
    "row_number",
    "rank",
    "dense_rank"
).show(20, truncate=False)


# 8. Customer Spending Ranking

customer_spending = (
    sales
    .join(
        customers.select(
            "customer_key",
            "customer_unique_id",
            "customer_city",
            "customer_state"
        ),
        on="customer_key",
        how="inner"
    )
    .groupBy(
        "customer_unique_id",
        "customer_city",
        "customer_state"
    )
    .agg(
        round(sum("gross_item_value"), 2).alias("total_spending"),

        countDistinct("order_id").alias("total_orders")
    )
)

customer_window = Window.orderBy(
    col("total_spending").desc()
)

customer_ranking = (
    customer_spending
    .withColumn(
        "customer_rank",
        dense_rank().over(customer_window)
    )
    .orderBy("customer_rank")
)

print("\n========== TOP 20 CUSTOMERS BY SPENDING ==========")

customer_ranking.show(20, truncate=False)


# 9. Save Results

monthly_running_sales.write.mode("overwrite").parquet(
    f"{OUTPUT}/monthly_running_sales"
)

monthly_growth.write.mode("overwrite").parquet(
    f"{OUTPUT}/monthly_sales_growth"
)

monthly_moving_average.write.mode("overwrite").parquet(
    f"{OUTPUT}/monthly_moving_average"
)

product_ranking.write.mode("overwrite").parquet(
    f"{OUTPUT}/product_sales_ranking"
)

customer_ranking.write.mode("overwrite").parquet(
    f"{OUTPUT}/customer_spending_ranking"
)


print("\nAdvanced window analytics completed successfully!")

spark.stop()
