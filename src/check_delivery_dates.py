
from pathlib import Path

from pyspark.sql import SparkSession
from pyspark.sql import functions as F

BASE_DIR = Path(__file__).resolve().parent.parent

spark = (
    SparkSession.builder
    .appName("CheckDeliveryDates")
    .master("local[*]")
    .config("spark.driver.host", "127.0.0.1")
    .config("spark.driver.bindAddress", "127.0.0.1")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("WARN")

orders = spark.read.parquet(
    str(BASE_DIR / "data/silver/olist_orders_enriched")
)

dates = spark.read.parquet(
    str(BASE_DIR / "data/gold/dim_date")
)

# Extract estimated delivery dates
estimated_dates = (
    orders
    .filter(F.col("order_estimated_delivery_date").isNotNull())
    .select(
        F.to_date("order_estimated_delivery_date")
        .alias("estimated_date")
    )
    .distinct()
)

# Find dates not present in dim_date
missing = estimated_dates.join(
    dates.select(
        F.col("full_date").alias("estimated_date")
    ),
    on="estimated_date",
    how="left_anti"
)

print("\nMISSING ESTIMATED DELIVERY DATES:")

missing.orderBy("estimated_date").show(50, truncate=False)

print("Number of missing dates:", missing.count())

print("\nCURRENT DATE DIMENSION RANGE:")

dates.select(
    F.min("full_date").alias("minimum_date"),
    F.max("full_date").alias("maximum_date")
).show()

print("\nESTIMATED DELIVERY DATE RANGE:")

estimated_dates.select(
    F.min("estimated_date").alias("minimum_date"),
    F.max("estimated_date").alias("maximum_date")
).show()

spark.stop()
