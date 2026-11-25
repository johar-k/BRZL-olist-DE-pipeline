
from pyspark.sql import SparkSession
from pyspark.sql.functions import count

spark = (
    SparkSession.builder
    .appName("Stage7_Catalyst_AQE")
    .master("local[*]")
    .config("spark.driver.host", "127.0.0.1")
    .config("spark.driver.bindAddress", "127.0.0.1")
    .config("spark.sql.adaptive.enabled", "true")
    .config("spark.sql.shuffle.partitions", "8")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("ERROR")

# Read Olist orders
orders = spark.read.parquet(
    "data/silver/olist_orders_dataset"
)

# Build a query
result = (
    orders
    .filter(orders.order_status == "delivered")
    .groupBy("customer_id")
    .agg(count("*").alias("total_orders"))
)

# Check AQE configuration
print("\n========== AQE CONFIGURATION ==========")

print(
    "AQE enabled:",
    spark.conf.get("spark.sql.adaptive.enabled")
)

print(
    "Shuffle partitions:",
    spark.conf.get("spark.sql.shuffle.partitions")
)

# View the optimized execution plan
print("\n========== EXECUTION PLAN BEFORE ACTION ==========")

result.explain(mode="formatted")

# Trigger execution
print("\n========== EXECUTION RESULT ==========")

result.show(10, truncate=False)

# Inspect the executed plan after the action
print("\n========== EXECUTED PLAN ==========")

print(
    result._jdf.queryExecution().executedPlan().toString()
)

spark.stop()
