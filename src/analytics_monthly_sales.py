
from pathlib import Path

from pyspark.sql import SparkSession
from pyspark.sql import functions as F


# 1. PROJECT PATHS

BASE_DIR = Path(__file__).resolve().parent.parent

GOLD_DIR = BASE_DIR / "data" / "gold"
OUTPUT_DIR = BASE_DIR / "output" / "analytics"

FACT_SALES_PATH = GOLD_DIR / "fact_sales"
DATE_DIM_PATH = GOLD_DIR / "dim_date"
PRODUCT_DIM_PATH = GOLD_DIR / "dim_product"


# 2. CREATE SPARK SESSION

spark = (
    SparkSession.builder
    .appName("MonthlySalesAnalytics")
    .master("local[*]")
    .config("spark.driver.host", "127.0.0.1")
    .config("spark.driver.bindAddress", "127.0.0.1")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("WARN")


# 3. READ GOLD TABLES

print("\nREADING GOLD TABLES")

sales = spark.read.parquet(str(FACT_SALES_PATH))
dates = spark.read.parquet(str(DATE_DIM_PATH))
products = spark.read.parquet(str(PRODUCT_DIM_PATH))

print("Sales records:", sales.count())


# 4. JOIN FACT AND DIMENSION TABLES

print("\nJOINING FACT AND DIMENSION TABLES")

sales_data = (
    sales
    .join(
        dates.select(
            "date_key",
            "year",
            "month",
            "month_name"
        ),
        on="date_key",
        how="left"
    )
    .join(
        products.select(
            "product_key",
            "product_category_name_english"
        ),
        on="product_key",
        how="left"
    )
)


# 5. MONTHLY SALES ANALYSIS

print("\nCALCULATING MONTHLY SALES")

monthly_sales = (
    sales_data
    .groupBy(
        "year",
        "month",
        "month_name"
    )
    .agg(
        F.count("*").alias("total_items"),

        F.countDistinct("order_id").alias("total_orders"),

        F.sum("item_quantity").alias("total_quantity"),

        F.sum("product_price").alias("total_product_sales"),

        F.sum("freight_value").alias("total_freight"),

        F.sum("gross_item_value").alias("total_sales")
    )
    .withColumn(
        "average_order_value",
        F.col("total_sales") / F.col("total_orders")
    )
    .orderBy(
        "year",
        "month"
    )
)


# 6. CATEGORY SALES ANALYSIS

print("\nCALCULATING CATEGORY SALES")

category_sales = (
    sales_data
    .groupBy(
        "product_category_name_english"
    )
    .agg(
        F.count("*").alias("total_items"),

        F.countDistinct("order_id").alias("total_orders"),

        F.sum("product_price").alias("total_product_sales"),

        F.sum("freight_value").alias("total_freight"),

        F.sum("gross_item_value").alias("total_sales")
    )
    .orderBy(
        F.desc("total_sales")
    )
)


# 7. FIND BEST SALES MONTH

print("\nBEST SALES MONTH")

best_month = (
    monthly_sales
    .orderBy(F.desc("total_sales"))
)

best_month.show(1, truncate=False)


# 8. DISPLAY RESULTS

print("\n========== MONTHLY SALES ==========")

monthly_sales.show(50, truncate=False)

print("\n========== TOP 20 PRODUCT CATEGORIES ==========")

category_sales.show(20, truncate=False)


# 9. SAVE ANALYTICS OUTPUT

print("\nSAVING ANALYTICS OUTPUT")

(
    monthly_sales.write
    .mode("overwrite")
    .parquet(str(OUTPUT_DIR / "monthly_sales"))
)

(
    category_sales.write
    .mode("overwrite")
    .parquet(str(OUTPUT_DIR / "category_sales"))
)


# 10. FINAL SUMMARY

print("\nANALYTICS SUMMARY")

print("Monthly records:", monthly_sales.count())
print("Category records:", category_sales.count())

print("\nMONTHLY SALES ANALYTICS COMPLETED!")

spark.stop()
