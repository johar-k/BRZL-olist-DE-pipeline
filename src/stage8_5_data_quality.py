
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

# 1. Create Spark session
spark = (
    SparkSession.builder
    .appName("Stage8_5_DataQuality")
    .master("local[*]")
    .config("spark.driver.host", "127.0.0.1")
    .config("spark.driver.bindAddress", "127.0.0.1")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("WARN")

# 2. Load existing datasets
orders = spark.read.parquet(
    "data/silver/olist_orders_dataset"
)

customers = spark.read.parquet(
    "data/silver/olist_customers_dataset"
)

payments = spark.read.parquet(
    "data/silver/olist_order_payments_dataset"
)

reviews = spark.read.parquet(
    "data/silver/olist_order_reviews_dataset"
)

# 3. Store quality results
quality_results = []


def record_check(check_name, failed_count, severity="ERROR"):
    status = "PASS" if failed_count == 0 else "FAIL"

    quality_results.append({
        "check_name": check_name,
        "failed_count": failed_count,
        "severity": severity,
        "status": status
    })

    print(
        f"{status} | {check_name} | "
        f"Failed records: {failed_count}"
    )


# 4. Completeness checks
record_check(
    "Orders: missing order_id",
    orders.filter(F.col("order_id").isNull()).count()
)

record_check(
    "Customers: missing customer_id",
    customers.filter(F.col("customer_id").isNull()).count()
)

# 5. Uniqueness checks
duplicate_orders = (
    orders.groupBy("order_id")
    .count()
    .filter(F.col("count") > 1)
    .count()
)

record_check(
    "Orders: duplicate order_id",
    duplicate_orders
)

duplicate_customers = (
    customers.groupBy("customer_id")
    .count()
    .filter(F.col("count") > 1)
    .count()
)

record_check(
    "Customers: duplicate customer_id",
    duplicate_customers
)

# 6. Validity checks
invalid_payments = payments.filter(
    F.col("payment_value").isNull() |
    (F.col("payment_value") < 0)
).count()

record_check(
    "Payments: null or negative payment_value",
    invalid_payments
)

invalid_reviews = reviews.filter(
    F.col("review_score").isNull() |
    (F.col("review_score") < 1) |
    (F.col("review_score") > 5)
).count()

record_check(
    "Reviews: invalid review_score",
    invalid_reviews
)

# 7. Referential integrity
invalid_customer_references = (
    orders.select("customer_id")
    .distinct()
    .join(
        customers.select("customer_id").distinct(),
        on="customer_id",
        how="left_anti"
    )
    .count()
)

record_check(
    "Orders: customer references missing",
    invalid_customer_references
)

invalid_payment_references = (
    payments.select("order_id")
    .distinct()
    .join(
        orders.select("order_id").distinct(),
        on="order_id",
        how="left_anti"
    )
    .count()
)

record_check(
    "Payments: order references missing",
    invalid_payment_references
)

# 8. Create quality report
report_schema = """
    check_name STRING,
    failed_count LONG,
    severity STRING,
    status STRING
"""

report = spark.createDataFrame(
    quality_results,
    schema=report_schema
)

print("\n========== DATA QUALITY REPORT ==========")

report.show(truncate=False)

# 9. Save report
report.write \
    .mode("overwrite") \
    .parquet("output/data_quality/report")

print("Data quality report saved successfully.")

# 10. Stop Spark
spark.stop()
