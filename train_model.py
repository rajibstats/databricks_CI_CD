from pyspark.ml.classification import DecisionTreeClassifier

def train_model(train_data):
    """
    Train a Decision Tree Classifier.
    """
    dt = DecisionTreeClassifier(featuresCol='features', labelCol='label')
    model = dt.fit(train_data)
    return model