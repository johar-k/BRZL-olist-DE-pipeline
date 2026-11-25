
from pyspark.sql import SparkSession
from pyspark.sql.types import (
    StructType,
    StructField,
    StringType,
    DoubleType,
    TimestampType
)

# 1. Create Spark session
spark = (
    SparkSession.builder
    .appName("Stage8_4_StructuredStreaming")
    .master("local[*]")
    .config("spark.driver.host", "127.0.0.1")
    .config("spark.driver.bindAddress", "127.0.0.1")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("WARN")

# 2. Define the incoming event schema
# Streaming sources should have an explicit schema.
order_schema = StructType([
    StructField("order_id", StringType(), True),
    StructField("customer_id", StringType(), True),
    StructField("amount", DoubleType(), True),
    StructField("status", StringType(), True),
    StructField("event_time", TimestampType(), True)
])

# 3. Define source and checkpoint locations
input_path = "data/streaming/input"
checkpoint_path = "data/streaming/checkpoint"

# 4. Read incoming JSON files as a stream
orders_stream = (
    spark.readStream
    .format("json")
    .schema(order_schema)
    .load(input_path)
)

# 5. Transform incoming records
processed_orders = (
    orders_stream
    .filter("order_id IS NOT NULL")
    .filter("amount >= 0")
    .select(
        "order_id",
        "customer_id",
        "amount",
        "status",
        "event_time"
    )
)

# 6. Write streaming output to console
query = (
    processed_orders.writeStream
    .format("console")
    .outputMode("append")
    .option("truncate", False)
    .option("checkpointLocation", checkpoint_path)
    .trigger(processingTime="5 seconds")
    .start()
)

print("Streaming order processor started...")
print("Waiting for incoming order files...")

# 7. Keep the streaming query running
query.awaitTermination()
