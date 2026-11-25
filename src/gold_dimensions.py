
from pathlib import Path

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window


# 1. PROJECT PATHS

PROJECT_ROOT = Path(__file__).resolve().parent.parent

SILVER_DIR = PROJECT_ROOT / "data" / "silver"
GOLD_DIR = PROJECT_ROOT / "data" / "gold"

CUSTOMERS_PATH = SILVER_DIR / "olist_customers_dataset"
PRODUCTS_PATH = SILVER_DIR / "olist_products_dataset"
SELLERS_PATH = SILVER_DIR / "olist_sellers_dataset"
TRANSLATIONS_PATH = SILVER_DIR / "product_category_name_translation"
ORDERS_PATH = SILVER_DIR / "olist_orders_enriched"
REVIEWS_PATH = SILVER_DIR / "olist_order_reviews_dataset"


# 2. CREATE SPARK SESSION

spark = (
    SparkSession.builder
    .appName("OlistGoldDimensions")
    .master("local[*]")
    .config("spark.driver.host", "127.0.0.1")
    .config("spark.driver.bindAddress", "127.0.0.1")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("ERROR")

print("\nGOLD DIMENSION TABLE GENERATION")
print("=" * 65)


# 3. HELPER: GENERATE SURROGATE KEYS

def add_surrogate_key(df, natural_key, surrogate_key):

    window_spec = Window.orderBy(natural_key)

    return df.withColumn(
        surrogate_key,
        F.row_number().over(window_spec)
    )


# 4. READ SILVER DATASETS

print("\nReading Silver datasets...")

customers = spark.read.parquet(str(CUSTOMERS_PATH))
products = spark.read.parquet(str(PRODUCTS_PATH))
sellers = spark.read.parquet(str(SELLERS_PATH))
translations = spark.read.parquet(str(TRANSLATIONS_PATH))
orders = spark.read.parquet(str(ORDERS_PATH))
reviews = spark.read.parquet(str(REVIEWS_PATH))


# 5. DIM_CUSTOMER
# Grain: One row per customer_id

print("\nBuilding dim_customer...")

dim_customer = customers.select(
    "customer_id",
    "customer_unique_id",
    "customer_zip_code_prefix",
    "customer_city",
    "customer_state"
)

dim_customer = add_surrogate_key(
    dim_customer,
    "customer_id",
    "customer_key"
)

dim_customer = dim_customer.select(
    "customer_key",
    "customer_id",
    "customer_unique_id",
    "customer_zip_code_prefix",
    "customer_city",
    "customer_state"
)


# 6. DIM_PRODUCT
# Grain: One row per product_id

print("\nBuilding dim_product...")

translation_data = translations.select(
    "product_category_name",
    "product_category_name_english"
)

dim_product = products.join(
    translation_data,
    on="product_category_name",
    how="left"
)

dim_product = dim_product.select(
    "product_id",
    "product_category_name",
    "product_category_name_english",
    "product_name_lenght",
    "product_description_lenght",
    "product_photos_qty",
    "product_weight_g",
    "product_length_cm",
    "product_height_cm",
    "product_width_cm"
)

dim_product = add_surrogate_key(
    dim_product,
    "product_id",
    "product_key"
)

dim_product = dim_product.select(
    "product_key",
    *[
        column
        for column in dim_product.columns
        if column != "product_key"
    ]
)


# 7. DIM_SELLER
# Grain: One row per seller_id

print("\nBuilding dim_seller...")

dim_seller = sellers.select(
    "seller_id",
    "seller_zip_code_prefix",
    "seller_city",
    "seller_state"
)

dim_seller = add_surrogate_key(
    dim_seller,
    "seller_id",
    "seller_key"
)

dim_seller = dim_seller.select(
    "seller_key",
    "seller_id",
    "seller_zip_code_prefix",
    "seller_city",
    "seller_state"
)


# 8. DIM_DATE
# Grain: One row per calendar date
#
# Covers:
# - Purchase dates
# - Actual delivery dates
# - Estimated delivery dates
# - Review creation dates

print("\nBuilding dim_date...")

# Collect purchase dates
purchase_dates = orders.select(
    F.to_date("order_purchase_timestamp").alias("business_date")
)

# Collect actual delivery dates
actual_delivery_dates = orders.select(
    F.to_date("order_delivered_customer_date").alias("business_date")
)

# Collect estimated delivery dates
estimated_delivery_dates = orders.select(
    F.to_date("order_estimated_delivery_date").alias("business_date")
)

# Collect review creation dates
review_dates = reviews.select(
    F.to_date("review_creation_date").alias("business_date")
)

# Combine all date sources
date_sources = (
    purchase_dates
    .unionByName(actual_delivery_dates)
    .unionByName(estimated_delivery_dates)
    .unionByName(review_dates)
)

# Remove NULL dates
date_sources = date_sources.filter(
    F.col("business_date").isNotNull()
)

# Find overall minimum and maximum dates
date_bounds = date_sources.agg(
    F.min("business_date").alias("start_date"),
    F.max("business_date").alias("end_date")
).first()

start_date = date_bounds["start_date"]
end_date = date_bounds["end_date"]

print("Minimum business date:", start_date)
print("Maximum business date:", end_date)

# Generate every calendar date in the range
dates = spark.createDataFrame(
    [(start_date, end_date)],
    ["start_date", "end_date"]
)

dim_date = dates.select(
    F.explode(
        F.sequence(
            F.col("start_date"),
            F.col("end_date"),
            F.expr("INTERVAL 1 DAY")
        )
    ).alias("full_date")
)

# Add calendar attributes
dim_date = dim_date.select(
    F.date_format("full_date", "yyyyMMdd")
    .cast("INT")
    .alias("date_key"),

    F.col("full_date"),

    F.dayofmonth("full_date").alias("day"),

    F.month("full_date").alias("month"),

    F.year("full_date").alias("year"),

    F.quarter("full_date").alias("quarter"),

    F.date_format("full_date", "MMMM").alias("month_name"),

    F.date_format("full_date", "EEEE").alias("day_name"),

    F.weekofyear("full_date").alias("week_of_year"),

    F.dayofweek("full_date").isin(1, 7).alias("is_weekend")
)


# 9. SAVE GOLD DIMENSIONS

dimensions = {
    "dim_customer": dim_customer,
    "dim_product": dim_product,
    "dim_seller": dim_seller,
    "dim_date": dim_date
}

print("\nSAVING DIMENSION TABLES")
print("-" * 65)

for table_name, df in dimensions.items():

    output_path = GOLD_DIR / table_name

    (
        df.write
        .mode("overwrite")
        .parquet(str(output_path))
    )

    print(f"{table_name}: {df.count()} records")
    print(f"Saved to: {output_path}")


# 10. DISPLAY SAMPLE DATA

print("\nDIM_CUSTOMER SAMPLE")
dim_customer.show(5, truncate=False)

print("\nDIM_PRODUCT SAMPLE")
dim_product.show(5, truncate=False)

print("\nDIM_SELLER SAMPLE")
dim_seller.show(5, truncate=False)

print("\nDIM_DATE SAMPLE")
dim_date.show(5, truncate=False)

print("\nDIM_DATE FINAL RANGE")

dim_date.select(
    F.min("full_date").alias("minimum_date"),
    F.max("full_date").alias("maximum_date"),
    F.count("*").alias("total_dates")
).show()


# 11. FINISH

spark.stop()

print("\nALL GOLD DIMENSIONS CREATED SUCCESSFULLY!")
