
from pyspark.sql import SparkSession
from pyspark.sql.functions import count

spark = (
    SparkSession.builder
    .appName("Stage7_Spark_Architecture")
    .master("local[*]")
    .config("spark.driver.host", "127.0.0.1")
    .config("spark.driver.bindAddress", "127.0.0.1")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("ERROR")

# Read Olist orders
orders = spark.read.parquet(
    "data/silver/olist_orders_dataset"
)

# Transformation: builds a computation plan
result = (
    orders
    .groupBy("order_status")
    .agg(count("*").alias("total_orders"))
)

print("\n========== BEFORE ACTION ==========")
print("The DataFrame transformation has been defined.")

# Action: triggers execution
print("\n========== AFTER ACTION ==========")
result.show()

# Inspect application information
print("\n========== SPARK APPLICATION ==========")
print("Application ID:", spark.sparkContext.applicationId)
print("Application name:", spark.sparkContext.appName)
print("Master:", spark.sparkContext.master)

print("\n========== EXECUTORS ==========")
print(spark.sparkContext._jsc.sc().getExecutorMemoryStatus().keys())

# Execution plan
print("\n========== EXECUTION PLAN ==========")
result.explain(mode="formatted")

input("Press Enter to stop Spark and close the application...")

spark.stop()
