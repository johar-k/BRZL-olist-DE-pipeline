
from pyspark.sql import SparkSession

spark = (
    SparkSession.builder
    .appName("Stage7_RDD_DataFrame")
    .master("local[*]")
    .config("spark.driver.host", "127.0.0.1")
    .config("spark.driver.bindAddress", "127.0.0.1")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("ERROR")

# PART 1: Read Olist orders as a DataFrame
orders_df = spark.read.parquet(
    "data/silver/olist_orders_dataset"
)

print("\n========== DATAFRAME ==========")
orders_df.printSchema()
print("DataFrame count:", orders_df.count())

# PART 2: Convert DataFrame to RDD
orders_rdd = orders_df.rdd

print("\n========== RDD ==========")
print("RDD count:", orders_rdd.count())

# Display one record
print("RDD sample:", orders_rdd.first())

# PART 3: RDD transformation
delivered_rdd = orders_rdd.filter(
    lambda row: row["order_status"] == "delivered"
)

print("\nDelivered orders using RDD:", delivered_rdd.count())

# PART 4: DataFrame transformation
delivered_df = orders_df.filter(
    orders_df.order_status == "delivered"
)

print("Delivered orders using DataFrame:", delivered_df.count())

# PART 5: Compare schemas
print("\n========== DATAFRAME SCHEMA ==========")
orders_df.printSchema()

print("\n========== RDD SAMPLE ==========")
print(orders_rdd.first())

spark.stop()
