# The project is done as a part of Assignments in Scalable Machine Learning course 
# at the University of Sheffield

# Supervised classification of Higgs bosons


from pyspark.sql import SparkSession
import pandas as pd
import numpy as np
import time
from pyspark.sql.types import DoubleType, IntegerType
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import RandomForestClassifier, GBTClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml import Pipeline
import heapq

spark = SparkSession.builder.master("local[2]")\
	.config("spark.local.dir","/data/my_name")\
	.appName("COM6012_Assignment2").getOrCreate()

sc = spark.sparkContext
sc.setLogLevel("WARN")


print("\nNumber of cores: 20, Memory: 6G, time: 1:00:00,  Output: AS2_20_gbt.output")
      
df_s = spark.read.csv("/data/my_name/HIGGS.csv.gz")
df_s.cache()

schemaNames = df_s.columns
labels = schemaNames[0]
feature_names = schemaNames[1:]

column_names =['labels', 'lepton_pT', 'lepton_eta', 'lepton_phi', 
               'missing_energy_magnitude', 'missing_energy_phi', 
               'jet_1_pt', 'jet_1_eta', 'jet_1_phi', 'jet_1_b_tag', 
              'jet_2_pt', 'jet_2_eta', 'jet_2_phi', 'jet_2_b-tag', 
               'jet_3_pt', 'jet_3_eta', 'jet_3_phi', 'jet_3_b-tag', 
               'jet_4_pt', 'jet_4_eta', 'jet_4_phi', 'jet_4_b-tag',
               'm_jj', 'm_jjj', 'm_lv', 'm_jlv', 'm_bb', 'm_wbb', 'm_wwbb']

# Changing labels types
df_s = df_s.withColumn(labels, df_s[labels].cast(DoubleType()).cast(IntegerType()))

# Changing types of data in columns
for column in feature_names:
    df_s = df_s.withColumn(column, df_s[column].cast(DoubleType()))

print("Split Train/Test data...")    
(trainingData, testData) = df_s.randomSplit([0.7, 0.3], 123)
trainingData.cache()
testData.cache()


assembler = VectorAssembler(inputCols = feature_names, outputCol = 'features')
trainingData = assembler.transform(trainingData).select("features", labels)
testData = assembler.transform(testData).select("features", labels)

trainingData.cache(), testData.cache()

df_s.unpersist()

print("\nTraining the GBT model...\n")
iterations = 20
depth = 10
bins = 10
print("Number of iterations:", iterations, "Depth:", depth, "Bins:", bins)

gbt = GBTClassifier(featuresCol='features', labelCol=labels, predictionCol='prediction', 
                    maxDepth=depth, maxBins=bins, maxIter=iterations, seed=123)


model_gbt = gbt.fit(trainingData)
print("Training is completed!")

print("Evaluating predictions")
predictions = model_gbt.transform(testData).select(labels, 'prediction', 'rawPrediction').cache()

evaluator = BinaryClassificationEvaluator(labelCol=labels, rawPredictionCol="rawPrediction", metricName='areaUnderROC')
evaluator2 = MulticlassClassificationEvaluator(labelCol=labels, predictionCol="prediction", metricName="accuracy")

print("Best GradientBoosting model AUC = %g " % evaluator.evaluate(predictions)) #0.7698 for split[0.9, 0.1] and #0.7769 for split[0.7, 0.3]
print("Best GradientBoosting model accuracy = %g \n" % evaluator2.evaluate(predictions))

feat_import_dict = model_gbt.featureImportances
a = np.array(feat_import_dict)
print('Three most important features in GradientBoosting: %s\n' %np.array(column_names[1:]).take([heapq.nlargest(3, range(len(a)), a.take)]))
spark.stop()
