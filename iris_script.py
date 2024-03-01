from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler, StringIndexer
from pyspark.ml.classification import MultilayerPerceptronClassifier, DecisionTreeClassifier, RandomForestClassifier
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

# Set your variables
model_type = "neural_network"  # Options: "neural_network", "decision_tree", "random_forest"
metric_name = "accuracy"  # Options: "f1", "accuracy", "weightedRecall", "weightedPrecision"
split_ratio = 0.8
layers = [4, 5, 3]  # Specify layers for the neural network (input features, hidden layer, output classes)
num_trees = 40  # Specify the number of trees for the random forest

# Create a Spark session
spark = SparkSession.builder.appName("IrisModelEvaluation").getOrCreate()

# Load the Iris dataset
iris_df=spark.read.load("file:///home/u1/Iris1.csv",format="csv",sep=",",inferSchema="tru e", header="true")


# Assuming 'features' is a vector containing your input features and 'species' is the target variable
assembler = VectorAssembler(inputCols=iris_df.columns[:-1], outputCol="features")
data = assembler.transform(df)
indexer = StringIndexer(inputCol="species", outputCol="label")
index_model = indexer.fit(data)

data_indexed = index_model.transform(data)

# Split the data into training and testing sets
trainingData, testData = data_indexed.randomSplit([split_ratio, 1 - split_ratio],0.0)

# Create the specified ML model
if model_type == "neural_network":
    model = MultilayerPerceptronClassifier(layers=layers, seed=123, featuresCol="features", labelCol="label")
elif model_type == "decision_tree":
    model = DecisionTreeClassifier(featuresCol="features", labelCol="label")
elif model_type == "random_forest":
    model = RandomForestClassifier(featuresCol="features", labelCol="label", numTrees=num_trees)


# Fit the model
trained_model = model.fit(trainingData)

# Make predictions on the test set
predictions = trained_model.transform(testData)

# Evaluate the model using the specified metric
evaluator = MulticlassClassificationEvaluator(metricName=metric_name)
metric_value = evaluator.evaluate(predictions)
print(f"{model_type.capitalize()} Model - {metric_name.capitalize()}: {metric_value:.4f}")

# Stop the Spark session
spark.stop()
