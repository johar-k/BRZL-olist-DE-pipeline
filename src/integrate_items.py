
from pathlib import Path

from pyspark.sql import SparkSession
from pyspark.sql import functions as F

# 1. PROJECT PATHS

PROJECT_ROOT = Path(__file__).resolve().parent.parent

SILVER_DIR = PROJECT_ROOT / "data" / "silver"

ITEMS_PATH = SILVER_DIR / "olist_order_items_dataset"
PRODUCTS_PATH = SILVER_DIR / "olist_products_dataset"
SELLERS_PATH = SILVER_DIR / "olist_sellers_dataset"

TRANSLATION_PATH = (
    SILVER_DIR / "product_category_name_translation"
)

ORDERS_PATH = SILVER_DIR / "olist_orders_dataset"

OUTPUT_PATH = SILVER_DIR / "olist_items_enriched"

# 2. CREATE SPARK SESSION

spark = (
    SparkSession.builder
    .appName("OlistItemsIntegration")
    .master("local[*]")
    .config("spark.driver.host", "127.0.0.1")
    .config("spark.driver.bindAddress", "127.0.0.1")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("ERROR")

print("\nPRODUCT AND SELLER INTEGRATION")
print("=" * 65)

# 3. READ SILVER DATASETS

items = spark.read.parquet(str(ITEMS_PATH))
products = spark.read.parquet(str(PRODUCTS_PATH))
sellers = spark.read.parquet(str(SELLERS_PATH))
translations = spark.read.parquet(str(TRANSLATION_PATH))
orders = spark.read.parquet(str(ORDERS_PATH))

original_count = items.count()

print(f"Order items: {original_count}")
print(f"Products: {products.count()}")
print(f"Sellers: {sellers.count()}")
print(f"Translations: {translations.count()}")

# 4. SELECT REQUIRED PRODUCT COLUMNS

product_data = products.select(
    "product_id",
    "product_category_name",
    "product_name_lenght",
    "product_description_lenght",
    "product_photos_qty",
    "product_weight_g",
    "product_length_cm",
    "product_height_cm",
    "product_width_cm"
)

# 5. SELECT REQUIRED SELLER COLUMNS

seller_data = sellers.select(
    "seller_id",
    "seller_zip_code_prefix",
    "seller_city",
    "seller_state"
)

# 6. SELECT TRANSLATION COLUMNS

translation_data = translations.select(
    "product_category_name",
    "product_category_name_english"
)

# 7. JOIN ORDER ITEMS WITH PRODUCTS

items_products = items.join(
    product_data,
    on="product_id",
    how="left"
)

# 8. JOIN WITH SELLERS

items_products_sellers = items_products.join(
    seller_data,
    on="seller_id",
    how="left"
)

# 9. JOIN CATEGORY TRANSLATIONS

items_enriched = items_products_sellers.join(
    translation_data,
    on="product_category_name",
    how="left"
)

# 10. JOIN ORDERS FOR ORDER STATUS

order_data = orders.select(
    "order_id",
    "order_status",
    "order_purchase_timestamp",
    "customer_id"
)

items_enriched = items_enriched.join(
    order_data,
    on="order_id",
    how="left"
)

# 11. DATA QUALITY CHECKS

final_count = items_enriched.count()

distinct_item_count = items_enriched.select(
    "order_id",
    "order_item_id"
).distinct().count()

missing_product_count = items_enriched.filter(
    F.col("product_category_name").isNull()
).count()

missing_seller_count = items_enriched.filter(
    F.col("seller_city").isNull()
).count()

missing_translation_count = items_enriched.filter(
    F.col("product_category_name_english").isNull()
    & F.col("product_category_name").isNotNull()
).count()

print("\nINTEGRATION QUALITY REPORT")
print("-" * 65)

print(f"Original order items: {original_count}")
print(f"Final integrated records: {final_count}")
print(f"Distinct order item keys: {distinct_item_count}")
print(f"Missing product category: {missing_product_count}")
print(f"Missing seller location: {missing_seller_count}")
print(f"Missing English translations: {missing_translation_count}")

if final_count != original_count:
    raise ValueError(
        "Unexpected row multiplication during integration!"
    )

if distinct_item_count != final_count:
    raise ValueError(
        "Duplicate order item keys found after integration!"
    )

# 12. INSPECT DATA

print("\nINTEGRATED ITEM SAMPLE")

items_enriched.select(
    "order_id",
    "order_item_id",
    "product_id",
    "product_category_name_english",
    "seller_id",
    "seller_city",
    "price",
    "freight_value",
    "order_status"
).show(10, truncate=False)

print("\nFINAL SCHEMA")

items_enriched.printSchema()

# 13. SAVE INTEGRATED DATASET

(
    items_enriched.write
    .mode("overwrite")
    .parquet(str(OUTPUT_PATH))
)

print("\nPRODUCT AND SELLER INTEGRATION COMPLETED!")
print(f"Output: {OUTPUT_PATH}")

spark.stop()

