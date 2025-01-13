from data_loader import load_data
from train_model import train_model
from pyspark.sql import SparkSession
from register_model import register_model

# Initialize Spark session
spark = SparkSession.builder.appName("IrisModel").getOrCreate()
# Step 1: Load the dataset
catalog_name = "cronos_unity_catalog"
schema_name = "default"
table_name = "iris_dataset"

processed_data = load_data(spark)
train_data, test_data = processed_data.randomSplit([0.8, 0.2], seed=42)

# Step 2: Train the model
model = train_model(train_data)

# Step 3: Register the model
register_model(model)
