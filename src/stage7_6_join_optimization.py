
from pyspark.sql import SparkSession
from pyspark.sql.functions import broadcast

spark = (
    SparkSession.builder
    .appName("Stage7_Join_Optimization")
    .master("local[*]")
    .config("spark.driver.host", "127.0.0.1")
    .config("spark.driver.bindAddress", "127.0.0.1")
    .config("spark.sql.adaptive.enabled", "true")
    .config("spark.sql.adaptive.skewJoin.enabled", "true")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("ERROR")

# 1. Read Silver datasets
orders = spark.read.parquet(
    "data/silver/olist_orders_dataset"
)

customers = spark.read.parquet(
    "data/silver/olist_customers_dataset"
)

# 2. Select only necessary columns
orders_small = orders.select(
    "order_id",
    "customer_id",
    "order_status"
)

customers_small = customers.select(
    "customer_id",
    "customer_state"
)

print("\n========== DATASET COUNTS ==========")
print("Orders:", orders_small.count())
print("Customers:", customers_small.count())

# 3. Normal join - Spark chooses the strategy
normal_join = orders_small.join(
    customers_small,
    on="customer_id",
    how="inner"
)

print("\n========== NORMAL JOIN PLAN ==========")
normal_join.explain(mode="formatted")

# 4. Broadcast Hash Join
broadcast_join = orders_small.join(
    broadcast(customers_small),
    on="customer_id",
    how="inner"
)

print("\n========== BROADCAST JOIN PLAN ==========")
broadcast_join.explain(mode="formatted")

# 5. Sort-Merge Join hint
sort_merge_join = orders_small.hint("merge").join(
    customers_small.hint("merge"),
    on="customer_id",
    how="inner"
)

print("\n========== SORT-MERGE JOIN PLAN ==========")
sort_merge_join.explain(mode="formatted")

# 6. Shuffle Hash Join hint
shuffle_hash_join = orders_small.hint(
    "shuffle_hash"
).join(
    customers_small.hint("shuffle_hash"),
    on="customer_id",
    how="inner"
)

print("\n========== SHUFFLE HASH JOIN PLAN ==========")
shuffle_hash_join.explain(mode="formatted")

# 7. Execute one join and verify its result
print("\n========== BROADCAST JOIN RESULT ==========")
broadcast_join.show(10, truncate=False)

print("\n========== AQE CONFIGURATION ==========")
print(
    "AQE enabled:",
    spark.conf.get("spark.sql.adaptive.enabled")
)

print(
    "Skew join enabled:",
    spark.conf.get("spark.sql.adaptive.skewJoin.enabled")
)

spark.stop()
