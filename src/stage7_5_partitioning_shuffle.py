
from pyspark.sql import SparkSession
from pyspark.sql.functions import count

spark = (
    SparkSession.builder
    .appName("Stage7_Partitioning_Shuffle")
    .master("local[*]")
    .config("spark.driver.host", "127.0.0.1")
    .config("spark.driver.bindAddress", "127.0.0.1")
    .config("spark.sql.shuffle.partitions", "8")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("ERROR")

# 1. Read Olist orders
orders = spark.read.parquet(
    "data/silver/olist_orders_dataset"
)

print("\n========== ORIGINAL PARTITIONS ==========")
print(orders.rdd.getNumPartitions())

# 2. Repartition
orders_repartitioned = orders.repartition(8)

print("\n========== AFTER REPARTITION ==========")
print(orders_repartitioned.rdd.getNumPartitions())

# 3. Coalesce
orders_coalesced = orders_repartitioned.coalesce(2)

print("\n========== AFTER COALESCE ==========")
print(orders_coalesced.rdd.getNumPartitions())

# 4. Repartition by a column
orders_by_status = orders.repartition(
    8,
    "order_status"
)

print("\n========== REPARTITION BY STATUS ==========")
print(orders_by_status.rdd.getNumPartitions())

# 5. Group by - may cause shuffle
status_count = (
    orders
    .groupBy("order_status")
    .agg(count("*").alias("total_orders"))
)

print("\n========== GROUP BY RESULT ==========")
status_count.show()

# 6. Inspect execution plan
print("\n========== GROUP BY EXECUTION PLAN ==========")
status_count.explain(mode="formatted")

# 7. Compare partition counts
print("\n========== FINAL PARTITION COMPARISON ==========")

print("Original:", orders.rdd.getNumPartitions())
print("Repartitioned:", orders_repartitioned.rdd.getNumPartitions())
print("Coalesced:", orders_coalesced.rdd.getNumPartitions())
print("By status:", orders_by_status.rdd.getNumPartitions())

spark.stop()
