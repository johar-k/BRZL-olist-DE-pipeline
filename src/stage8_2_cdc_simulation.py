
from pyspark.sql import SparkSession
from pyspark.sql.functions import col

spark = (
    SparkSession.builder
    .appName("Stage8_CDC_Simulation")
    .master("local[*]")
    .config("spark.driver.host", "127.0.0.1")
    .config("spark.driver.bindAddress", "127.0.0.1")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("ERROR")

# STEP 1: EXISTING TARGET DATA

target_data = [
    (101, "C01", "Pending"),
    (102, "C02", "Shipped"),
    (103, "C03", "Delivered"),
]

target = spark.createDataFrame(
    target_data,
    ["order_id", "customer_id", "status"]
)

print("\n========== ORIGINAL TARGET ==========")
target.show()

# STEP 2: INCOMING CDC EVENTS

change_data = [
    (101, "C01", "Shipped", "UPDATE"),
    (104, "C04", "Pending", "INSERT"),
    (103, "C03", None, "DELETE"),
]

changes = spark.createDataFrame(
    change_data,
    ["order_id", "customer_id", "status", "operation"]
)

print("\n========== CDC EVENTS ==========")
changes.show()

# STEP 3: EXTRACT OPERATIONS

inserts = changes.filter(
    col("operation") == "INSERT"
)

updates = changes.filter(
    col("operation") == "UPDATE"
)

deletes = changes.filter(
    col("operation") == "DELETE"
)

# STEP 4: REMOVE DELETED RECORDS

remaining = target.join(
    deletes.select("order_id"),
    on="order_id",
    how="left_anti"
)

# STEP 5: APPLY UPDATES

updated_target = (
    remaining.alias("t")
    .join(
        updates.select(
            "order_id",
            "status"
        ).alias("u"),
        on="order_id",
        how="left"
    )
    .select(
        col("t.order_id"),
        col("t.customer_id"),
        col("u.status").alias("updated_status"),
        col("t.status").alias("old_status")
    )
    .select(
        "order_id",
        "customer_id",
        col("updated_status").alias("status")
    )
)

# STEP 6: APPLY INSERTS

final_target = updated_target.unionByName(
    inserts.select(
        "order_id",
        "customer_id",
        "status"
    )
)

# STEP 7: DISPLAY FINAL DATA

print("\n========== FINAL TARGET AFTER CDC ==========")

final_target.orderBy("order_id").show()

spark.stop()
