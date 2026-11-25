
from pyspark.sql import SparkSession

from pyspark.sql.functions import (
    col,
    countDistinct,
    count,
    sum,
    avg,
    round,
    when,
    desc,
    try_divide
)


spark = (
    SparkSession.builder
    .appName("Olist-Seller-Performance-Analytics")
    .master("local[*]")
    .config("spark.driver.host", "127.0.0.1")
    .config("spark.driver.bindAddress", "127.0.0.1")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("ERROR")

BASE = "data/gold"

# 1. Read Gold tables

sales = spark.read.parquet(f"{BASE}/fact_sales")
sellers = spark.read.parquet(f"{BASE}/dim_seller")
delivery = spark.read.parquet(f"{BASE}/fact_delivery")

# 2. Join sales with seller dimension

sales_with_sellers = (
    sales
    .join(
        sellers,
        on="seller_key",
        how="inner"
    )
)

# 3. Seller-level sales performance

seller_performance = (
    sales_with_sellers
    .groupBy(
        "seller_key",
        "seller_id",
        "seller_city",
        "seller_state"
    )
    .agg(
        countDistinct("order_id").alias("total_orders"),
        count("*").alias("total_items_sold"),
        round(sum("product_price"), 2).alias("total_product_sales"),
        round(sum("freight_value"), 2).alias("total_freight"),
        round(sum("gross_item_value"), 2).alias("total_sales"),
        round(avg("product_price"), 2).alias("average_item_price")
    )
    .withColumn(
        "average_sales_per_order",
        round(col("total_sales") / col("total_orders"), 2)
    )
)

# 4. Top 20 sellers by total sales

top_sellers = (
    seller_performance
    .orderBy(desc("total_sales"))
    .limit(20)
)

print("\n========== TOP 20 SELLERS BY SALES ==========")
top_sellers.show(20, truncate=False)

# 5. Seller performance by order status

seller_order_status = (
    sales_with_sellers
    .groupBy(
        "seller_id",
        "seller_city",
        "seller_state",
        "order_status"
    )
    .agg(
        countDistinct("order_id").alias("total_orders"),
        round(sum("gross_item_value"), 2).alias("total_sales")
    )
    .orderBy(desc("total_sales"))
)

print("\n========== SELLER PERFORMANCE BY ORDER STATUS ==========")
seller_order_status.show(30, truncate=False)

# 6. Delivery performance

# First, reduce item-level sales to one row per seller/order.
seller_orders = (
    sales_with_sellers
    .groupBy("seller_id", "order_id")
    .agg(
        count("*").alias("items_in_order"),
        round(sum("gross_item_value"), 2).alias("seller_order_sales")
    )
)

# Join one seller/order record to one order-level delivery record.
seller_delivery = (
    seller_orders
    .join(
        delivery.select(
            "order_id",
            "delivery_days",
            "delivery_delay_days",
            "is_delivered",
            "is_late"
        ),
        on="order_id",
        how="left"
    )
)


seller_delivery_performance = (
    seller_delivery
    .groupBy("seller_id")
    .agg(
        countDistinct("order_id").alias("total_orders"),

        count(
            when(col("is_delivered") == 1, 1)
        ).alias("delivered_orders"),

        count(
            when(col("is_late") == 1, 1)
        ).alias("late_orders"),

        round(avg("delivery_days"), 2).alias("average_delivery_days"),

        round(avg("delivery_delay_days"), 2).alias("average_delivery_delay_days")
    )
    .withColumn(
        "late_delivery_rate",
        round(
            try_divide(
                col("late_orders") * 100,
                col("delivered_orders")
            ),
            2
        )
    )
    .orderBy(desc("late_delivery_rate"))
)

print("\n========== SELLER DELIVERY PERFORMANCE ==========")
seller_delivery_performance.show(30, truncate=False)



# 7. Join seller performance with delivery metrics

complete_seller_performance = (
    seller_performance
    .join(
        seller_delivery_performance.drop("total_orders"),
        on="seller_id",
        how="left"
    )
)

print("\n========== COMPLETE SELLER PERFORMANCE ==========")
complete_seller_performance.show(20, truncate=False)

# 8. Save analytics outputs

output = "output/analytics"

seller_performance.write.mode("overwrite").parquet(
    f"{output}/seller_performance"
)

top_sellers.write.mode("overwrite").parquet(
    f"{output}/top_sellers"
)

seller_order_status.write.mode("overwrite").parquet(
    f"{output}/seller_order_status"
)

seller_delivery_performance.write.mode("overwrite").parquet(
    f"{output}/seller_delivery_performance"
)

complete_seller_performance.write.mode("overwrite").parquet(
    f"{output}/complete_seller_performance"
)

print("\nSeller analytics completed successfully!")

spark.stop()
