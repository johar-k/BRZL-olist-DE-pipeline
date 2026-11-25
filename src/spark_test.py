
from pyspark.sql import SparkSession

# 1. Create Spark session
spark = (
    SparkSession.builder
    .appName("SparkInstallationTest")
    .master("local[*]")
    .getOrCreate()
)

print("SPARK STARTED SUCCESSFULLY")

# 2. Check Spark version
print("Spark version:", spark.version)

# 3. Create sample data
data = [
    (1, "Johar", 25000),
    (2, "Rahul", 30000),
    (3, "Kiran", 20000),
    (4, "Arun", 35000)
]

columns = ["id", "name", "salary"]

# 4. Create Spark DataFrame
df = spark.createDataFrame(data, columns)

# 5. Display data
df.show()

# 6. Perform a transformation
high_salary = df.filter(df.salary > 25000)

print("Employees earning above 25000:")
high_salary.show()

# 7. Stop Spark
spark.stop()

print("Spark test completed successfully!")
