
from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    count,
    countDistinct
)
from pyspark.sql.functions import broadcast

spark = (
    SparkSession.builder
    .appName("Stage7_Performance_Tuning")
    .master("local[*]")
    .config("spark.driver.host", "127.0.0.1")
    .config("spark.driver.bindAddress", "127.0.0.1")
    .config("spark.sql.adaptive.enabled", "true")
    .config("spark.sql.adaptive.skewJoin.enabled", "true")
    .config("spark.sql.shuffle.partitions", "8")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("ERROR")

# STEP 1: Read only required columns
orders = (
    spark.read.parquet(
        "data/silver/olist_orders_dataset"
    )
    .select(
        "order_id",
        "customer_id",
        "order_status"
    )
)

customers = (
    spark.read.parquet(
        "data/silver/olist_customers_dataset"
    )
    .select(
        "customer_id",
        "customer_unique_id",
        "customer_state"
    )
)

# STEP 2: Filter early
delivered_orders = orders.filter(
    orders.order_status == "delivered"
)

# STEP 3: Broadcast join
joined_data = delivered_orders.join(
    broadcast(customers),
    on="customer_id",
    how="inner"
)

# STEP 4: Cache reusable data
joined_data.cache()

# Materialize cache
print("\n========== MATERIALIZE CACHE ==========")
print("Joined records:", joined_data.count())

# STEP 5: Customer analytics
print("\n========== CUSTOMER ANALYTICS ==========")

customer_stats = (
    joined_data
    .groupBy("customer_unique_id")
    .agg(
        countDistinct("order_id").alias("total_orders")
    )
    .orderBy("total_orders", ascending=False)
)

customer_stats.show(10)

# STEP 6: State analytics
print("\n========== STATE ANALYTICS ==========")

state_stats = (
    joined_data
    .groupBy("customer_state")
    .agg(
        countDistinct("order_id").alias("total_orders"),
        countDistinct("customer_unique_id").alias(
            "total_customers"
        )
    )
    .orderBy("total_orders", ascending=False)
)

state_stats.show()

# STEP 7: Inspect execution plan
print("\n========== CUSTOMER EXECUTION PLAN ==========")

customer_stats.explain(mode="formatted")

# STEP 8: Cache status
print("\n========== CACHE STATUS ==========")
print("Is cached:", joined_data.is_cached)
print("Storage:", joined_data.storageLevel)

# STEP 9: Release cache
joined_data.unpersist()

print("\n========== CACHE RELEASED ==========")
print("Is cached:", joined_data.is_cached)

spark.stop()
