
from pathlib import Path

from pyspark.sql import SparkSession
from pyspark.sql import functions as F


# 1. PROJECT PATHS

BASE_DIR = Path(__file__).resolve().parent.parent

GOLD_DIR = BASE_DIR / "data" / "gold"
OUTPUT_DIR = BASE_DIR / "output" / "analytics"

FACT_SALES_PATH = GOLD_DIR / "fact_sales"
CUSTOMER_DIM_PATH = GOLD_DIR / "dim_customer"


# 2. CREATE SPARK SESSION

spark = (
    SparkSession.builder
    .appName("CustomerRegionAnalytics")
    .master("local[*]")
    .config("spark.driver.host", "127.0.0.1")
    .config("spark.driver.bindAddress", "127.0.0.1")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("WARN")


# 3. READ GOLD TABLES

print("\nREADING GOLD TABLES")

sales = spark.read.parquet(str(FACT_SALES_PATH))
customers = spark.read.parquet(str(CUSTOMER_DIM_PATH))

print("Sales records:", sales.count())
print("Customer dimension records:", customers.count())


# 4. JOIN SALES WITH CUSTOMER DIMENSION

print("\nJOINING CUSTOMER DIMENSION")

customer_sales = sales.join(
    customers.select(
        "customer_key",
        "customer_unique_id",
        "customer_city",
        "customer_state"
    ),
    on="customer_key",
    how="left"
)


# 5. CUSTOMER SPENDING ANALYSIS
# Grain: One row per actual customer

print("\nCALCULATING CUSTOMER SPENDING")

customer_summary = (
    customer_sales
    .groupBy(
        "customer_unique_id",
        "customer_city",
        "customer_state"
    )
    .agg(
        F.countDistinct("order_id").alias("total_orders"),

        F.count("*").alias("total_items"),

        F.sum("product_price").alias("total_product_spending"),

        F.sum("freight_value").alias("total_freight"),

        F.sum("gross_item_value").alias("total_spending")
    )
    .withColumn(
        "average_order_value",
        F.col("total_spending") / F.col("total_orders")
    )
    .withColumn(
        "customer_type",
        F.when(
            F.col("total_orders") > 1,
            "Repeat Customer"
        ).otherwise("One-time Customer")
    )
)


# 6. TOP 20 CUSTOMERS

print("\nTOP 20 CUSTOMERS BY SPENDING")

top_customers = (
    customer_summary
    .orderBy(F.desc("total_spending"))
)

top_customers.show(20, truncate=False)


# 7. REPEAT CUSTOMER ANALYSIS

print("\nCUSTOMER TYPE ANALYSIS")

customer_types = (
    customer_summary
    .groupBy("customer_type")
    .agg(
        F.count("*").alias("total_customers"),

        F.sum("total_orders").alias("total_orders"),

        F.sum("total_spending").alias("total_spending")
    )
)

customer_types.show(truncate=False)


# 8. STATE-WISE SALES ANALYSIS

print("\nSTATE-WISE SALES")

state_sales = (
    customer_sales
    .groupBy("customer_state")
    .agg(
        F.countDistinct("customer_unique_id")
        .alias("unique_customers"),

        F.countDistinct("order_id")
        .alias("total_orders"),

        F.count("*")
        .alias("total_items"),

        F.sum("gross_item_value")
        .alias("total_sales")
    )
    .withColumn(
        "average_order_value",
        F.col("total_sales") / F.col("total_orders")
    )
    .orderBy(F.desc("total_sales"))
)

state_sales.show(30, truncate=False)


# 9. CITY-WISE SALES ANALYSIS

print("\nTOP 20 CITIES BY SALES")

city_sales = (
    customer_sales
    .groupBy(
        "customer_state",
        "customer_city"
    )
    .agg(
        F.countDistinct("customer_unique_id")
        .alias("unique_customers"),

        F.countDistinct("order_id")
        .alias("total_orders"),

        F.sum("gross_item_value")
        .alias("total_sales")
    )
    .orderBy(F.desc("total_sales"))
)

city_sales.show(20, truncate=False)


# 10. SAVE ANALYTICS OUTPUT

print("\nSAVING ANALYTICS RESULTS")

(
    customer_summary.write
    .mode("overwrite")
    .parquet(str(OUTPUT_DIR / "customer_summary"))
)

(
    customer_types.write
    .mode("overwrite")
    .parquet(str(OUTPUT_DIR / "customer_types"))
)

(
    state_sales.write
    .mode("overwrite")
    .parquet(str(OUTPUT_DIR / "state_sales"))
)

(
    city_sales.write
    .mode("overwrite")
    .parquet(str(OUTPUT_DIR / "city_sales"))
)


# 11. FINAL SUMMARY

print("\nANALYTICS SUMMARY")

print("Unique customers:", customer_summary.count())
print("Customer types:", customer_types.count())
print("States:", state_sales.count())
print("Cities:", city_sales.count())

print("\nCUSTOMER AND REGIONAL ANALYTICS COMPLETED!")

spark.stop()
