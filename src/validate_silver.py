
from pathlib import Path

from pyspark.sql import SparkSession
from pyspark.sql import functions as F

# 1. PROJECT PATHS

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SILVER_DIR = PROJECT_ROOT / "data" / "silver"

PATHS = {
    "orders": SILVER_DIR / "olist_orders_dataset",
    "customers": SILVER_DIR / "olist_customers_dataset",
    "products": SILVER_DIR / "olist_products_dataset",
    "sellers": SILVER_DIR / "olist_sellers_dataset",
    "items": SILVER_DIR / "olist_order_items_dataset",
    "payments": SILVER_DIR / "olist_order_payments_dataset",
    "reviews": SILVER_DIR / "olist_order_reviews_dataset",
    "translations": SILVER_DIR / "product_category_name_translation",
    "geolocation": SILVER_DIR / "olist_geolocation_dataset",
    "orders_enriched": SILVER_DIR / "olist_orders_enriched",
    "items_enriched": SILVER_DIR / "olist_items_enriched",
}

# 2. CREATE SPARK SESSION

spark = (
    SparkSession.builder
    .appName("OlistSilverValidation")
    .master("local[*]")
    .config("spark.driver.host", "127.0.0.1")
    .config("spark.driver.bindAddress", "127.0.0.1")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("ERROR")

print("\nOLIST SILVER DATA VALIDATION")
print("=" * 70)

# 3. READ DATASETS

dfs = {
    name: spark.read.parquet(str(path))
    for name, path in PATHS.items()
}

orders = dfs["orders"]
customers = dfs["customers"]
products = dfs["products"]
sellers = dfs["sellers"]
items = dfs["items"]
payments = dfs["payments"]
reviews = dfs["reviews"]
translations = dfs["translations"]
geolocation = dfs["geolocation"]

orders_enriched = dfs["orders_enriched"]
items_enriched = dfs["items_enriched"]

# 4. VALIDATION HELPER

results = []

def record_check(name, actual, expected, passed):
    status = "PASS" if passed else "FAIL"

    results.append({
        "check": name,
        "actual": str(actual),
        "expected": str(expected),
        "status": status
    })

    print(
        f"{status:6} | {name:40} | "
        f"Actual: {actual} | Expected: {expected}"
    )


def count_duplicates(df, columns):
    return (
        df.groupBy(*columns)
        .count()
        .filter(F.col("count") > 1)
        .count()
    )


# 5. RECORD COUNT VALIDATION

print("\n1. RECORD COUNT VALIDATION")
print("-" * 70)

source_orders_count = orders.count()
integrated_orders_count = orders_enriched.count()

source_items_count = items.count()
integrated_items_count = items_enriched.count()

record_check(
    "Orders integration row count",
    integrated_orders_count,
    source_orders_count,
    integrated_orders_count == source_orders_count
)

record_check(
    "Order items integration row count",
    integrated_items_count,
    source_items_count,
    integrated_items_count == source_items_count
)

# 6. PRIMARY KEY VALIDATION

print("\n2. PRIMARY KEY VALIDATION")
print("-" * 70)

duplicate_orders = count_duplicates(
    orders_enriched,
    ["order_id"]
)

duplicate_items = count_duplicates(
    items_enriched,
    ["order_id", "order_item_id"]
)

record_check(
    "Duplicate order IDs",
    duplicate_orders,
    0,
    duplicate_orders == 0
)

record_check(
    "Duplicate order item keys",
    duplicate_items,
    0,
    duplicate_items == 0
)

# 7. NULL VALIDATION

print("\n3. NULL VALIDATION")
print("-" * 70)

missing_order_ids = orders_enriched.filter(
    F.col("order_id").isNull()
).count()

missing_item_ids = items_enriched.filter(
    F.col("product_id").isNull()
    | F.col("seller_id").isNull()
    | F.col("order_item_id").isNull()
).count()

record_check(
    "Missing order IDs",
    missing_order_ids,
    0,
    missing_order_ids == 0
)

record_check(
    "Missing required item identifiers",
    missing_item_ids,
    0,
    missing_item_ids == 0
)

# 8. FOREIGN KEY VALIDATION

print("\n4. FOREIGN KEY VALIDATION")
print("-" * 70)

order_ids = orders.select("order_id").distinct()
product_ids = products.select("product_id").distinct()
seller_ids = sellers.select("seller_id").distinct()

orphan_order_items = (
    items
    .join(order_ids, "order_id", "left_anti")
    .count()
)

orphan_products = (
    items
    .join(product_ids, "product_id", "left_anti")
    .count()
)

orphan_sellers = (
    items
    .join(seller_ids, "seller_id", "left_anti")
    .count()
)

record_check(
    "Order item references to orders",
    orphan_order_items,
    0,
    orphan_order_items == 0
)

record_check(
    "Order item references to products",
    orphan_products,
    0,
    orphan_products == 0
)

record_check(
    "Order item references to sellers",
    orphan_sellers,
    0,
    orphan_sellers == 0
)

# 9. CUSTOMER RELATIONSHIP VALIDATION

print("\n5. CUSTOMER RELATIONSHIP VALIDATION")
print("-" * 70)

missing_customers = (
    orders
    .join(
        customers.select("customer_id"),
        "customer_id",
        "left_anti"
    )
    .count()
)

record_check(
    "Orders without matching customers",
    missing_customers,
    0,
    missing_customers == 0
)

# 10. PAYMENT RECONCILIATION

print("\n6. PAYMENT RECONCILIATION")
print("-" * 70)

source_payment_total = payments.agg(
    F.sum("payment_value").alias("total")
).first()["total"]

integrated_payment_total = orders_enriched.agg(
    F.sum("total_payment_value").alias("total")
).first()["total"]

source_payment_total = source_payment_total or 0
integrated_payment_total = integrated_payment_total or 0

payment_difference = abs(
    float(source_payment_total)
    - float(integrated_payment_total)
)

record_check(
    "Total payment reconciliation",
    round(float(integrated_payment_total), 2),
    round(float(source_payment_total), 2),
    payment_difference < 0.01
)

# 11. ITEM VALUE SUMMARY

print("\n7. ITEM FINANCIAL SUMMARY")
print("-" * 70)

item_summary = items_enriched.agg(
    F.sum("price").alias("total_product_price"),
    F.sum("freight_value").alias("total_freight")
)

item_summary.show(truncate=False)

print(
    "Note: Product prices plus freight are not expected "
    "to equal payments exactly. Vouchers, discounts and "
    "other payment adjustments can cause differences."
)

# 12. DATASET INVENTORY

print("\n8. SILVER DATASET INVENTORY")
print("-" * 70)

for name, df in dfs.items():
    print(f"{name:20} | {df.count()} records")

# 13. FINAL VALIDATION REPORT

print("\nFINAL VALIDATION REPORT")
print("=" * 70)

passed_count = sum(
    1 for result in results
    if result["status"] == "PASS"
)

failed_count = sum(
    1 for result in results
    if result["status"] == "FAIL"
)

print(f"Total checks: {len(results)}")
print(f"Passed: {passed_count}")
print(f"Failed: {failed_count}")

if failed_count == 0:
    print("\nALL SILVER VALIDATION CHECKS PASSED!")
else:
    print("\nSOME VALIDATION CHECKS FAILED.")
    print("Investigate the failed checks before proceeding.")

# 14. STOP SPARK

spark.stop()
