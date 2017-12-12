# Classify Images using Tensorflow on Spark
# Authors: Saurabh Deochake, Ridip De, Anish Grover

from sparkdl import readImages
from pyspark.sql.functions import lit
from pyspark.ml.classification import LogisticRegression
from pyspark.ml import Pipeline
from sparkdl import DeepImageFeaturizer
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

img_dir = "/home/sdeochake/Downloads/images/"

# Read Images from image directory
# Create dataframes for respective member pictures
anish_df = readImages(img_dir + "/anish").withColumn("label", lit(1))
ridip_df = readImages(img_dir + "/ridip").withColumn("label", lit(0))

# Get 60% of images for training and 40% for testing
anish_train, anish_test = anish_df.randomSplit([0.6, 0.4])
ridip_train, ridip_test = ridip_df.randomSplit([0.6, 0.4])

# get a dataframe for training the neurons
train_df = anish_train.unionAll(ridip_train)

# get a dataframe for testing the neurons
test_df = anish_test.unionAll(ridip_test)

# apply logistic regression on InceptionV3
featurizer = DeepImageFeaturizer(inputCol="image", outputCol="features", modelName="InceptionV3")
lr = LogisticRegression(maxIter=20, regParam=0.05, elasticNetParam=0.3, labelCol="label")
p = Pipeline(stages=[featurizer, lr])
p_model = p.fit(train_df)

# get predictions on testing dataframes
predictions = p_model.transform(test_df)
predictions.select("filePath", "prediction").show(truncate=False)

# show testing dataframe prediction and accuracy
df = p_model.transform(test_df)
df.show()

predictionAndLabels = df.select("prediction", "label")
evaluator = MulticlassClassificationEvaluator(metricName="accuracy")
print "Training set is at "+str(evaluator.evaluate(predictionAndLabels))+" accuracy"
