import os
from dotenv import load_dotenv
import mlflow
import mlflow.spark

# Load environment variables
load_dotenv()

def register_model(model):
    """
    Register a trained model to MLflow Model Registry.
    """
    run_name = os.getenv("MLFLOW_RUN_NAME")
    registered_model_name = os.getenv("MLFLOW_REGISTERED_MODEL_NAME")
    
    with mlflow.start_run(run_name=run_name):
        # Log the model and register it
        mlflow.spark.log_model(
            spark_model=model,
            artifact_path="dt_model",
            registered_model_name=registered_model_name
        )
        print(f"Model '{registered_model_name}' successfully registered in MLflow.")
