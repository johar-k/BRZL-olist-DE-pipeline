
from pyspark.sql import SparkSession
from pyspark.sql.functions import count, avg

spark = (
    SparkSession.builder
    .appName("Stage7_Caching_Persistence")
    .master("local[*]")
    .config("spark.driver.host", "127.0.0.1")
    .config("spark.driver.bindAddress", "127.0.0.1")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("ERROR")

# 1. Read orders
orders = spark.read.parquet(
    "data/silver/olist_orders_dataset"
)

# 2. Create a reusable DataFrame
delivered_orders = orders.filter(
    orders.order_status == "delivered"
).select(
    "order_id",
    "customer_id",
    "order_purchase_timestamp",
    "order_status"
)

# 3. Cache the DataFrame
delivered_orders.cache()

# 4. First action - materializes the cache
print("\n========== FIRST ACTION ==========")

total_orders = delivered_orders.count()

print("Delivered orders:", total_orders)

# 5. Second action - reuses cached data
print("\n========== SECOND ACTION ==========")

customer_counts = (
    delivered_orders
    .groupBy("customer_id")
    .agg(count("*").alias("total_orders"))
)

customer_counts.show(10)

# 6. Third action - reuses cached data
print("\n========== THIRD ACTION ==========")

average_orders = (
    delivered_orders
    .groupBy("order_status")
    .agg(count("*").alias("total_orders"))
)

average_orders.show()

# 7. Inspect storage information
print("\n========== STORAGE INFORMATION ==========")

print("Is cached:", delivered_orders.is_cached)

print(
    "Storage level:",
    delivered_orders.storageLevel
)

# 8. Release cached data
delivered_orders.unpersist()

print("\n========== AFTER UNPERSIST ==========")

print("Is cached:", delivered_orders.is_cached)

spark.stop()
