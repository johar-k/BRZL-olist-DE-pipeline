
from pathlib import Path

from pyspark.sql import SparkSession
from pyspark.sql import functions as F


BASE_DIR = Path(__file__).resolve().parent.parent
GOLD_DIR = BASE_DIR / "data/gold"

spark = (
    SparkSession.builder
    .appName("ValidateGoldLayer")
    .master("local[*]")
    .config("spark.driver.host", "127.0.0.1")
    .config("spark.driver.bindAddress", "127.0.0.1")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("WARN")


# 1. TABLE DEFINITIONS

tables = {
    "dim_customer": ["customer_key"],
    "dim_product": ["product_key"],
    "dim_seller": ["seller_key"],
    "dim_date": ["date_key"],

    "fact_sales": ["order_id", "order_item_id"],
    "fact_payments": ["order_id", "payment_sequential"],
    "fact_reviews": ["review_id"],
    "fact_delivery": ["order_id"]
}


# 2. CHECK TABLE EXISTENCE AND RECORD COUNTS

print("\n========== GOLD TABLE VALIDATION ==========\n")

dataframes = {}
validation_results = []

for table_name, key_columns in tables.items():

    table_path = GOLD_DIR / table_name

    if not table_path.exists():
        raise FileNotFoundError(
            f"Gold table missing: {table_name}"
        )

    df = spark.read.parquet(str(table_path))

    dataframes[table_name] = df

    row_count = df.count()

    duplicate_count = (
        df.groupBy(*key_columns)
        .count()
        .filter(F.col("count") > 1)
        .count()
    )

    null_key_condition = None

    for column in key_columns:
        condition = F.col(column).isNull()

        if null_key_condition is None:
            null_key_condition = condition
        else:
            null_key_condition = (
                null_key_condition | condition
            )

    null_key_count = df.filter(
        null_key_condition
    ).count()

    print(f"TABLE: {table_name}")
    print(f"Records: {row_count}")
    print(f"Duplicate keys: {duplicate_count}")
    print(f"Null keys: {null_key_count}")
    print("------------------------------------------")

    validation_results.append({
        "table": table_name,
        "records": row_count,
        "duplicates": duplicate_count,
        "null_keys": null_key_count
    })

    if duplicate_count > 0 or null_key_count > 0:
        raise ValueError(
            f"Validation failed for {table_name}"
        )


# 3. FACT TABLE RELATIONSHIP CHECKS

print("\n========== FACT RELATIONSHIP CHECKS ==========\n")

checks = [
    ("fact_sales", "customer_key", "dim_customer", "customer_key"),
    ("fact_sales", "product_key", "dim_product", "product_key"),
    ("fact_sales", "seller_key", "dim_seller", "seller_key"),
    ("fact_sales", "date_key", "dim_date", "date_key"),

    ("fact_payments", "customer_key", "dim_customer", "customer_key"),
    ("fact_payments", "date_key", "dim_date", "date_key"),

    ("fact_reviews", "customer_key", "dim_customer", "customer_key"),
    ("fact_reviews", "date_key", "dim_date", "date_key"),

    ("fact_delivery", "customer_key", "dim_customer", "customer_key"),
    ("fact_delivery", "purchase_date_key", "dim_date", "date_key")
]

for fact_name, fact_column, dim_name, dim_column in checks:

    fact_df = dataframes[fact_name]
    dim_df = dataframes[dim_name]

    missing = (
        fact_df.select(fact_column)
        .distinct()
        .join(
            dim_df.select(dim_column).distinct(),
            fact_df[fact_column] == dim_df[dim_column],
            "left_anti"
        )
        .count()
    )

    print(
        f"{fact_name}.{fact_column} -> "
        f"{dim_name}.{dim_column}: {missing} unmatched"
    )

    if missing > 0:
        raise ValueError(
            f"Unmatched dimension keys in {fact_name}"
        )


# 4. SUMMARY

print("\n========== VALIDATION SUMMARY ==========\n")

for result in validation_results:
    print(
        f"{result['table']}: "
        f"{result['records']} records | "
        f"Duplicates: {result['duplicates']} | "
        f"Null keys: {result['null_keys']}"
    )

print("\nALL GOLD TABLE CHECKS PASSED!")

spark.stop()
