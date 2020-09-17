# The project is done as a part of Assignments in Scalable Machine Learning course 
# at the University of Sheffield

# Tandem learning for prediction of insurance claim payments based on 
# vehicle characteristics.

from pyspark.sql import SparkSession
import pandas as pd
import numpy as np
import time
from pyspark.sql.types import DoubleType, IntegerType
import pyspark.sql.functions as F
from pyspark.ml import Pipeline
#from pyspark.ml.feature import OneHotEncoder, StringIndexer
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.feature import OneHotEncoderEstimator, VectorAssembler
from pyspark.ml.regression import LinearRegression, GeneralizedLinearRegression
from pyspark.ml.classification import RandomForestClassifier, GBTClassifier,DecisionTreeClassifier, LogisticRegression
import pickle

spark = SparkSession.builder.master("local[2]")\
	.config("spark.local.dir","/fastdata/my_name")\
	.appName("COM6012_Assignment2").getOrCreate()

sc = spark.sparkContext
sc.setLogLevel("WARN")


# %%time
# df_large =  spark.read.csv("/data/my_name/train_set.csv", header=True)
df_s0 =  spark.read.csv("/data/my_name/train_set.csv", header=True)
# df_large =  spark.read.csv("train_set.csv", header=True)
# (df_s0, rest) = df_large.randomSplit([0.0005, 0.9995], 123)

# Currently, GeneralizedLinearRegression only supports number of features <= 4096. Found 4164 in the input dataset.
# so we need to drop some colunms to decrease number of features
# large_categor = ['Blind_Make','Blind_Model', 'Blind_Submodel']
large_categor = ['Blind_Submodel', 'Model_Year']
string_categories = ['Cat'+str(i) for i in range(1, 13)]+ ["NVCat"]
int_categories = ['Calendar_Year', 'OrdCat']
# int_categories = ['Calendar_Year', 'OrdCat']
numerical_features = ["Var"+str(i) for i in range(1,9)] #+["NVVar"+str(i) for i in range(1,5)]
new_columns = int_categories + numerical_features  + string_categories + ['Claim_Amount'] + large_categor

print("Dropped columns:", [col for col in df_s0.columns if col not in new_columns])

df_s = df_s0.select(new_columns)
df_s.cache()

# %%time
# real missing values appears only in Cat12
for categor in new_columns:
    none_values = df_s.filter(df_s[categor].isNull()==True).count()
    if none_values != 0:
        print("Column %s had %i None values" % (categor, none_values))
        df_s = df_s.withColumn(categor, F.when(df_s[categor].isNull()==True, "ZZZ").otherwise(df_s[categor]))

#testing target column
none_values = df_s.filter(df_s['Claim_Amount'].isNull()==True).count()
if none_values != 0:
    print("Column Claim_Amount had %i None values" % none_values)
    df_s = df_s.withColumn('Claim_Amount', F.when(df_s['Claim_Amount'].isNull()==True, "0").otherwise(df_s['Claim_Amount']))

# with open('mapping_dict3.txt', 'rb') as handle:
#     mapping_dict = pickle.loads(handle.read())

with open('/home/my_name/mapping_dict4.txt', 'rb') as handle:
    mapping_dict = pickle.loads(handle.read())

for categor in large_categor+string_categories:
    categor_dict = mapping_dict[categor]
    udf_categor_decode = F.UserDefinedFunction(lambda x : categor_dict[x])
    df_s = df_s.withColumn(categor, udf_categor_decode(categor))

# new_columns = int_categories + numerical_features  + string_categories + ['Claim_Amount'] + large_categor
for col in new_columns:
    df_s = df_s.withColumn(col, df_s[col].cast(DoubleType()))
    df_s = df_s.withColumn(col, F.when(df_s[col].isNull()==True, "0").otherwise(df_s[col]).cast(DoubleType()))

df_s = df_s.select(new_columns).cache()

df_s = df_s.withColumn("y", F.when(df_s["Claim_Amount"] !=0, 1).otherwise(df_s["Claim_Amount"]))
df_s.cache()
print("\nFirst feature preprocessing is done!")

print("\nStart stratified Train/Test split...")

(train_0, test_0) = df_s.filter(df_s.y ==0).randomSplit([0.7, 0.3], 123)
(train_1, test_1) = df_s.filter(df_s.y ==1).randomSplit([0.7, 0.3], 123)

train = train_0.union(train_1)
test = test_0.union(test_1)
train.cache(), test.cache()

print("Start training Classifier for the first part...")
categorical_features = int_categories + string_categories + large_categor

feature_names1 = categorical_features+numerical_features
assembler = VectorAssembler(inputCols = feature_names1, outputCol = 'features') 

depth=20

clf = DecisionTreeClassifier(labelCol="y", featuresCol="features", maxDepth=depth, impurity='entropy', predictionCol='prediction1')

pipeline = Pipeline(stages = [assembler, clf])

model_clf = pipeline.fit(train)
print("Classification training is completed!")
df_train = model_clf.transform(train)

print("Predicted number of ones: %i,actual number of ones: %i, all tested samples in train: %i" % (df_train.filter(df_train["prediction1"] !=0).count(), train.filter(train["y"] !=0).count(), train.count()))
print(" number of ones in df_s: %i, all samples in dfs: %i " % (df_s.filter(df_s['y'] ==1).count(), df_s.count()))

print("Start training Regressor for the second part...")

categorical_features2 = int_categories + string_categories + large_categor
transformed_cat_features2 = [feature_name +"_ohe" for feature_name in categorical_features2]

ohe = OneHotEncoderEstimator(inputCols = categorical_features2, outputCols=transformed_cat_features2, 
                              handleInvalid = "keep")

feature_names2 = transformed_cat_features2 + numerical_features
assembler = VectorAssembler(inputCols = feature_names2, outputCol = 'features') 
reg = GeneralizedLinearRegression(family='gamma', link='log', featuresCol='features', labelCol='Claim_Amount', maxIter=50, 
                                  predictionCol='prediction2', regParam=0.01)
# reg = LinearRegression(featuresCol='features', labelCol='Claim_Amount', predictionCol='prediction2', maxIter=50, regParam=0.1)
pipeline = Pipeline(stages = [ohe, assembler, reg])

# train2 = df_train.filter(((df_train["y"] ==1) & (df_train["prediction1"]==1))).select(categorical_features+numerical_features+["y", "Claim_Amount"])
train2 = train.filter(train["y"] ==1).select(categorical_features+numerical_features+["y", "Claim_Amount"]).cache()

model_reg = pipeline.fit(train2)
print("Regressor training is completed!")

print("Preparing final predictions...")
print("Predictions from clf")
pred_clf = model_clf.transform(test)
pred_clf = pred_clf.withColumn("id", F.monotonically_increasing_id())

print("Predictions from Regressor")
df_test = pred_clf.filter(pred_clf['prediction1']==1).select(feature_names1+['Claim_Amount',"id"]).cache()
# df_test.count()

pred_reg = model_reg.transform(df_test).select("id", "prediction2").cache()

print("Joining the results")
pred_clf=pred_clf.select("id", "prediction1", "Claim_Amount").cache()
predictions_final = pred_clf.join(pred_reg, pred_clf.id == pred_reg.id, how='left') # Could also use 'left_outer'
predictions_final = predictions_final.withColumn("prediction", 
            F.when(predictions_final["prediction2"].isNull()==True, 0).otherwise(predictions_final["prediction2"])).\
            select("prediction","Claim_Amount").cache()

print("Start evaluation of predictions for test data")
evaluator = RegressionEvaluator(labelCol="Claim_Amount", predictionCol="prediction", metricName="rmse")
evaluator2 = RegressionEvaluator(labelCol="Claim_Amount", predictionCol="prediction", metricName="mae")
rmse = evaluator.evaluate(predictions_final)
print("RMSE = %g " % rmse)
mae = evaluator2.evaluate(predictions_final)
print("MAE = %g " % mae)
spark.stop()