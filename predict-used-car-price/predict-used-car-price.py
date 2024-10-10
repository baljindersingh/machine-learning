## ./spark-submit --driver-memory 8g --executor-memory 8g /Users/baljinder/Tools/Kaggle/predict_used_car_price.py

# Import necessary libraries
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, mean
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler
from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator

# Initialize Spark session
spark = SparkSession.builder \
    .appName("UsedCarPricePrediction") \
    .config("spark.driver.memory", "8g") \
    .config("spark.executor.memory", "8g") \
    .getOrCreate()
    
# Set the logging level
spark.sparkContext.setLogLevel("WARN")

# Load training and test data from CSV files
df = spark.read.csv("train.csv", header=True, inferSchema=True)

# Data Cleaning

# Handle missing values
# Fill missing numeric values with the mean (for model_year, milage, price)
numeric_columns = ["model_year", "milage", "price"]
for column in numeric_columns:
    mean_value = df.select(mean(col(column))).collect()[0][0]
    df = df.fillna({column: mean_value})

# Fill missing categorical values with a placeholder (e.g., "Unknown")
categorical_columns = ["brand", "model", "fuel_type", "engine", "transmission", "ext_col", "int_col", "accident", "clean_title"]
for column in categorical_columns:
    df = df.fillna({column: "Unknown"})

# Remove duplicate records
df = df.dropDuplicates()

# Split the data into 70% training and 30% test sets
train_df, test_df = df.randomSplit([0.8, 0.2], seed=42)

# Data Preprocessing

# Index and encode categorical columns
indexers = [StringIndexer(inputCol=column, outputCol=column + "_index").setHandleInvalid("keep") for column in categorical_columns]
encoders = [OneHotEncoder(inputCol=column + "_index", outputCol=column + "_encoded") for column in categorical_columns]

# Assemble all features into a single vector
# Include numeric columns and encoded categorical columns
feature_columns = ["model_year", "milage"] + [column + "_encoded" for column in categorical_columns]
assembler = VectorAssembler(inputCols=feature_columns, outputCol="features")

# Define the linear regression model
lr = LinearRegression(featuresCol="features", labelCol="price", predictionCol="prediction")

# Create a pipeline to chain the transformations and model
from pyspark.ml import Pipeline
pipeline = Pipeline(stages=indexers + encoders + [assembler, lr])

# Train the model on the cleaned training data
model = pipeline.fit(train_df)

# Make predictions on the cleaned test data
predictions = model.transform(test_df)

# Evaluate the model using Root Mean Squared Error (RMSE)
evaluator = RegressionEvaluator(labelCol="price", predictionCol="prediction", metricName="rmse")
rmse = evaluator.evaluate(predictions)
print(f"Root Mean Squared Error (RMSE) on test data: {rmse}")

# Now, load a new CSV file without the price column for prediction
new_data_df = spark.read.csv("test.csv", header=True, inferSchema=True)

# Fill missing categorical values with a placeholder (e.g., "Unknown")
for column in categorical_columns:
    new_data_df = new_data_df.fillna({column: "Unknown"})

# Apply the trained model to predict prices for the new dataset
predictions_new_data = model.transform(new_data_df)

# Show the predicted prices along with the original features
##predictions_new_data.select("id", "features", "prediction").show()

# Optionally, save the predicted results to a CSV file
predictions_new_data.select("id", "prediction").coalesce(1).write.csv("predicted_prices.csv", mode='overwrite', header=True)

# Stop the Spark session
spark.stop()