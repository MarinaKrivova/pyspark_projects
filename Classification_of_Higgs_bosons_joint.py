# The project is done as a part of Assignments in Scalable Machine Learning course 
# at the University of Sheffield

# Supervised classification of Higgs bosons

# module load apps/java/jdk1.8.0_102/binary
# module load apps/python/conda
# source activate myspark
# pyspark


from pyspark.sql import SparkSession
import pandas as pd
import numpy as np
import time
from pyspark.sql.types import DoubleType, FloatType, IntegerType
import pyspark.sql.functions as F
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import RandomForestClassifier, GBTClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.tuning import CrossValidator
from pyspark.ml.tuning import ParamGridBuilder
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
import heapq

spark = SparkSession.builder.master("local[2]").config("spark.local.dir","/fastdata/my_name").appName("COM6012_Assignment2").getOrCreate()

sc = spark.sparkContext
sc.setLogLevel("WARN")

print("\nNumber of cores: 4, Memory: 16G, Max time: 02:00:00, Output: AS2_joint.output")

nFolds = 5
print("Number of folds:", nFolds)

df_large = spark.read.csv("/data/my_name/HIGGS.csv.gz")

(df_s, df_rest) = df_large.randomSplit([0.05, 0.95], 123)
df_s.cache()


schemaNames = df_s.columns
labels = schemaNames[0]
column_names =['labels', 'lepton_pT', 'lepton_eta', 'lepton_phi',
               'missing_energy_magnitude', 'missing_energy_phi',
               'jet_1_pt', 'jet_1_eta', 'jet_1_phi', 'jet_1_b_tag',
              'jet_2_pt', 'jet_2_eta', 'jet_2_phi', 'jet_2_b-tag',
               'jet_3_pt', 'jet_3_eta', 'jet_3_phi', 'jet_3_b-tag',
               'jet_4_pt', 'jet_4_eta', 'jet_4_phi', 'jet_4_b-tag',
               'm_jj', 'm_jjj', 'm_lv', 'm_jlv', 'm_bb', 'm_wbb', 'm_wwbb']

df_s = df_s.withColumn(schemaNames[0], df_s[schemaNames[0]].cast(DoubleType()).cast(IntegerType()))

#Checking that labels are values only 1 or 0
# df_s.groupby('_c0').agg(F.count('_c3')).alias("count").show()
df_s = df_s.withColumnRenamed(schemaNames[0], column_names[0])

# Changing types of data in columns
number_features = len(schemaNames[1:])
for i in range(1, number_features+1):
    df_s = df_s.withColumn(schemaNames[i], df_s[schemaNames[i]].cast(DoubleType()))
    #df_s = df_s.withColumnRenamed(schemaNames[i], column_names[i])

(trainingData, testData) = df_s.randomSplit([0.7, 0.3], 123)
trainingData.cache()
testData.cache()

#default numTrees=20
print("\nRunning code for RF...\n")
trees_to_test = [5, 10, 20]
print("Tested Trees:", trees_to_test)

#default values for bins is 32
bins_to_test = [3, 5, 10]
print("Tested Bins:", bins_to_test)

#default maxDepth=5
depth_to_test = 10
print("Tested Depth:", depth_to_test)

feature_names = column_names[1:]

assembler = VectorAssembler(inputCols = feature_names, outputCol = 'features') 
rf = RandomForestClassifier(featuresCol='features', labelCol='labels',
    predictionCol='prediction', maxDepth=depth_to_test, impurity='entropy', maxBins=10, numTrees=20, seed=123)

pipeline = Pipeline(stages = [assembler, rf])

grid = ParamGridBuilder().addGrid(rf.numTrees, trees_to_test).addGrid(rf.maxBins, bins_to_test).build() 
evaluator = BinaryClassificationEvaluator(labelCol=labels, rawPredictionCol='rawPrediction', metricName='areaUnderROC')

cv_rf = CrossValidator(numFolds=nFolds, estimator=pipeline, estimatorParamMaps=grid, 
                    evaluator=evaluator, seed = 123)

print("Cross validation for RF... \n")
startTimeCV_rf = time.clock()
cvModel_rf = cv_rf.fit(trainingData) 
endTimeCV_rf = time.clock()
print("Cross validation for RF is completed!")
print("Time for RF:", endTimeCV_rf-startTimeCV_rf, "ms")

best_model_rf = cvModel_rf.bestModel.stages[1]

#best_depth = best_model_rf.getOrDefault("maxDepth")
#print("Best maxDepth for RF:", best_depth)

max_bins = best_model_rf.getOrDefault("maxBins")
print("Best maxBins for RF:", max_bins)

num_trees = best_model_rf.getNumTrees
print("Best numTrees for RF:", num_trees)

# !!! for predictions it should be pipeline and the best model
predictions = cvModel_rf.transform(testData).select(labels, 'rawPrediction', 'prediction').cache()
evaluator1 = BinaryClassificationEvaluator(labelCol=labels, rawPredictionCol='rawPrediction', metricName='areaUnderROC')
print("Best RandomForest model AUC = %g " % evaluator1.evaluate(predictions)) 

evaluator2 = MulticlassClassificationEvaluator(labelCol=labels, predictionCol="prediction", metricName="accuracy")
print("Best RandomForest model Accuracy = %g " % evaluator2.evaluate(predictions)) 

print("\nRunning code for GBT...")

#default maxDepth=5
depth_to_test = 10
print("Tested depth:", depth_to_test)

#default maxBins=32
bins_to_test = [3, 5, 10]
print("Tested bins:", bins_to_test)

#default maxIter=20
iter_to_test = [5, 10, 15]
print("Tested Iterations:", iter_to_test)

feature_names = column_names[1:]

assembler = VectorAssembler(inputCols = feature_names, outputCol = 'features') 
gbt = GBTClassifier(featuresCol='features', labelCol=labels, predictionCol='prediction', 
                    maxDepth=depth_to_test, maxBins=5, maxIter=10, seed=123)
        
pipeline = Pipeline(stages = [assembler, gbt])

grid = ParamGridBuilder().addGrid(gbt.maxBins, bins_to_test).addGrid(gbt.maxIter, iter_to_test).build() 

evaluator = BinaryClassificationEvaluator(labelCol=labels, rawPredictionCol='rawPrediction', metricName='areaUnderROC')

cv_gbt = CrossValidator(numFolds=nFolds, estimator=pipeline, estimatorParamMaps=grid, 
                    evaluator=evaluator, seed = 123)

print("Cross validation for GBT...\n")
startTimeCV_gbt = time.clock()
cvModel_gbt = cv_gbt.fit(trainingData) 
endTimeCV_gbt = time.clock()
print("Cross validation for RF is completed!")
print("Time for GBT:", endTimeCV_gbt-startTimeCV_gbt, "ms")

best_model_gbt = cvModel_gbt.bestModel.stages[1]

#best_depth = best_model_gbt.getOrDefault("maxDepth")
#print("Best maxDepth for GBT:", best_depth)

max_bins = best_model_gbt.getOrDefault("maxBins")
print("Best maxBins for GBT:", max_bins)

max_Iter = best_model_gbt.getOrDefault("maxIter")
print("Best max_Iter for GBT:", max_Iter)

# !!! for predictions it should be pipeline and the best model
predictions = cvModel_gbt.transform(testData).select(labels, 'rawPrediction', 'prediction').cache()
evaluator1 = BinaryClassificationEvaluator(labelCol=labels, rrawPredictionCol='rawPrediction', metricName='areaUnderROC')
print("Best GBT model AUC = %g " % evaluator1.evaluate(predictions)) 

evaluator2 = MulticlassClassificationEvaluator(labelCol=labels, predictionCol="prediction", metricName="accuracy")
print("Best GBT model Accuracy = %g " % evaluator2.evaluate(predictions))  

spark.stop()
