import os
from dotenv import load_dotenv
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, VectorAssembler

# Load environment variables
load_dotenv()

def load_data(spark):
    """
    Load and preprocess data from a table in Unity Catalog.
    """
    catalog_name = os.getenv("CATALOG_NAME")
    schema_name = os.getenv("SCHEMA_NAME")
    table_name = os.getenv("TABLE_NAME")
    
    df = spark.read.table(f"{catalog_name}.{schema_name}.{table_name}")
    # df = spark.read.csv("iris_dataset.csv", header=True, inferSchema=True)

    feature_columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
    label_column = 'species'
    
    assembler = VectorAssembler(inputCols=feature_columns, outputCol='features')
    label_indexer = StringIndexer(inputCol=label_column, outputCol='label')
    
    pipeline = Pipeline(stages=[assembler, label_indexer])
    processed_data = pipeline.fit(df).transform(df)
    
    return processed_data
