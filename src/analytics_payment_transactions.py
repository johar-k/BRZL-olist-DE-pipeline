
from pyspark.sql import SparkSession

from pyspark.sql.functions import (
    col,
    count,
    countDistinct,
    sum,
    avg,
    round,
    desc,
    when
)

# 1. Create Spark Session

spark = (
    SparkSession.builder
    .appName("Olist-Payment-Transaction-Analytics")
    .master("local[*]")
    .config("spark.driver.host", "127.0.0.1")
    .config("spark.driver.bindAddress", "127.0.0.1")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("ERROR")

BASE = "data/gold"
OUTPUT = "output/analytics"

# 2. Read Gold Tables

payments = spark.read.parquet(
    f"{BASE}/fact_payments"
)

dates = spark.read.parquet(
    f"{BASE}/dim_date"
)

print("\n========== PAYMENT TABLE SCHEMA ==========")
payments.printSchema()

# 3. Payment Method Analytics

payment_method_summary = (
    payments
    .groupBy("payment_type")
    .agg(
        count("*").alias("total_transactions"),

        countDistinct("order_id").alias("total_orders"),

        round(sum("payment_value"), 2).alias("total_payment_value"),

        round(avg("payment_value"), 2).alias("average_payment_value"),

        round(avg("payment_installments"), 2).alias("average_installments")
    )
    .orderBy(desc("total_payment_value"))
)

print("\n========== PAYMENT METHOD SUMMARY ==========")
payment_method_summary.show(truncate=False)

# 4. Most Common Payment Methods

most_used_payment_methods = (
    payments
    .groupBy("payment_type")
    .agg(
        count("*").alias("transaction_count")
    )
    .orderBy(desc("transaction_count"))
)

print("\n========== MOST USED PAYMENT METHODS ==========")
most_used_payment_methods.show(truncate=False)

# 5. Installment Analytics

installment_summary = (
    payments
    .groupBy("payment_installments")
    .agg(
        count("*").alias("total_transactions"),

        countDistinct("order_id").alias("total_orders"),

        round(sum("payment_value"), 2).alias("total_payment_value"),

        round(avg("payment_value"), 2).alias("average_payment_value")
    )
    .orderBy("payment_installments")
)

print("\n========== INSTALLMENT SUMMARY ==========")
installment_summary.show(truncate=False)

# 6. Installment Categories

installment_categories = (
    payments
    .withColumn(
        "installment_category",
        when(col("payment_installments") == 1, "Single Payment")
        .when(col("payment_installments") <= 3, "2-3 Installments")
        .when(col("payment_installments") <= 6, "4-6 Installments")
        .when(col("payment_installments") <= 12, "7-12 Installments")
        .otherwise("More than 12 Installments")
    )
    .groupBy("installment_category")
    .agg(
        count("*").alias("total_transactions"),

        countDistinct("order_id").alias("total_orders"),

        round(sum("payment_value"), 2).alias("total_payment_value")
    )
    .orderBy(desc("total_transactions"))
)

print("\n========== INSTALLMENT CATEGORY ANALYSIS ==========")
installment_categories.show(truncate=False)

# 7. Monthly Payment Analytics

monthly_payments = (
    payments
    .join(
        dates.select(
            "date_key",
            "year",
            "month",
            "month_name"
        ),
        on="date_key",
        how="inner"
    )
    .groupBy(
        "year",
        "month",
        "month_name"
    )
    .agg(
        count("*").alias("total_transactions"),

        countDistinct("order_id").alias("total_orders"),

        round(sum("payment_value"), 2).alias("total_payment_value"),

        round(avg("payment_value"), 2).alias("average_payment_value")
    )
    .orderBy("year", "month")
)

print("\n========== MONTHLY PAYMENT ANALYTICS ==========")
monthly_payments.show(100, truncate=False)

# 8. Multiple Payment Transactions Per Order

order_payment_summary = (
    payments
    .groupBy("order_id")
    .agg(
        count("*").alias("number_of_transactions"),

        round(sum("payment_value"), 2).alias("total_order_payment")
    )
)

multiple_payment_orders = (
    order_payment_summary
    .filter(col("number_of_transactions") > 1)
    .orderBy(desc("number_of_transactions"))
)

print("\n========== MULTIPLE PAYMENT ORDERS ==========")
multiple_payment_orders.show(30, truncate=False)

# 9. Overall Payment Statistics

overall_payment_summary = (
    payments
    .agg(
        count("*").alias("total_transactions"),

        countDistinct("order_id").alias("total_orders"),

        round(sum("payment_value"), 2).alias("total_payment_value"),

        round(avg("payment_value"), 2).alias("average_transaction_value"),

        round(avg("payment_installments"), 2).alias("average_installments")
    )
)

print("\n========== OVERALL PAYMENT STATISTICS ==========")
overall_payment_summary.show(truncate=False)

# 10. Save Analytics Results

payment_method_summary.write.mode("overwrite").parquet(
    f"{OUTPUT}/payment_method_summary"
)

most_used_payment_methods.write.mode("overwrite").parquet(
    f"{OUTPUT}/most_used_payment_methods"
)

installment_summary.write.mode("overwrite").parquet(
    f"{OUTPUT}/installment_summary"
)

installment_categories.write.mode("overwrite").parquet(
    f"{OUTPUT}/installment_categories"
)

monthly_payments.write.mode("overwrite").parquet(
    f"{OUTPUT}/monthly_payments"
)

multiple_payment_orders.write.mode("overwrite").parquet(
    f"{OUTPUT}/multiple_payment_orders"
)

overall_payment_summary.write.mode("overwrite").parquet(
    f"{OUTPUT}/overall_payment_summary"
)

print("\nPayment and transaction analytics completed successfully!")

spark.stop()
    