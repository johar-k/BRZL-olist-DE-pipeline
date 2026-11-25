
import pytest

from pyspark.sql import SparkSession
from pyspark.sql.types import (
    StructType,
    StructField,
    StringType,
    DoubleType
)

from src.transformations import clean_orders


@pytest.fixture(scope="session")
def spark():

    session = (
        SparkSession.builder
        .appName("PipelineTesting")
        .master("local[2]")
        .config("spark.driver.host", "127.0.0.1")
        .config("spark.driver.bindAddress", "127.0.0.1")
        .getOrCreate()
    )

    session.sparkContext.setLogLevel("ERROR")

    yield session

    session.stop()


@pytest.fixture
def sample_orders(spark):

    schema = StructType([
        StructField("order_id", StringType(), True),
        StructField("customer_id", StringType(), True),
        StructField("amount", DoubleType(), True)
    ])

    data = [
        ("ORD101", "CUS01", 250.0),
        ("ORD102", "CUS02", 450.0),
        ("ORD103", "CUS03", -100.0),
        (None, "CUS04", 300.0),
        ("ORD101", "CUS01", 250.0)
    ]

    return spark.createDataFrame(data, schema)


def test_remove_null_order_ids(sample_orders):

    result = clean_orders(sample_orders)

    assert result.filter(
        "order_id IS NULL"
    ).count() == 0


def test_remove_duplicate_orders(sample_orders):

    result = clean_orders(sample_orders)

    duplicate_count = (
        result.groupBy("order_id")
        .count()
        .filter("count > 1")
        .count()
    )

    assert duplicate_count == 0


def test_remove_negative_amounts(sample_orders):

    result = clean_orders(sample_orders)

    assert result.filter(
        "amount < 0"
    ).count() == 0


def test_expected_record_count(sample_orders):

    result = clean_orders(sample_orders)

    assert result.count() == 2


def test_cleaned_order_ids(sample_orders):

    result = clean_orders(sample_orders)

    actual_ids = {
        row["order_id"]
        for row in result.select("order_id").collect()
    }

    expected_ids = {"ORD101", "ORD102"}

    assert actual_ids == expected_ids
