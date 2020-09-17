# ssh -X acq18mk@sharc.shef.ac.uk
# qrshx 
# module load apps/java/jdk1.8.0_102/binary
# module load apps/python/conda
# source activate myspark
# pyspark

from pyspark.sql import SparkSession
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS
import numpy as np
import pandas as pd
import pyspark.sql.functions as F
import matplotlib.pyplot as plt
from pyspark.ml.linalg import Vectors
from pyspark.ml.clustering import KMeans
from pyspark.ml.clustering import KMeansModel
import matplotlib.pyplot as plt
import numpy as np
import time
import pyspark.sql.functions as F


spark = SparkSession.builder.master("local[2]").config("spark.local.dir","/fastdata/my_name").appName("COM6012 Assignment2").getOrCreate()


sc = spark.sparkContext
sc.setLogLevel("WARN")
print("\nThe program has started")
df = spark.read.load("Data/ratings.csv", format="csv", inferSchema="true", header="true").select('userId', 'movieId', 'rating').cache()
print("\nData has been loaded")

# Question 1A
# manual modelling of Cross-Validation
(fold1, fold2, fold3) = df.randomSplit([1/3, 1/3, 1/3], seed = 123)
train_1 = fold2.union(fold3).cache()
test_1 = fold1.cache()
train_2 = fold1.union(fold3).cache()
test_2 = fold2.cache()
train_3 = fold2.union(fold1).cache()
test_3 = fold3.cache()
folds_list = [[train_1, test_1], [train_2, test_2],[train_3, test_3]]

evaluator_rmse = RegressionEvaluator(metricName="rmse", labelCol="rating",predictionCol="prediction")
evaluator_mae = RegressionEvaluator(metricName="mae", labelCol="rating",predictionCol="prediction")

parameters_to_test = [0.1, 0.5, 1.0]
num_models = len(parameters_to_test)
num_folds = 3

# matrices to strore results: row = model, column = fold
results_rmse = np.zeros([num_models, num_folds])
results_mae = np.zeros([num_models, num_folds])

evaluator_rmse = RegressionEvaluator(metricName="rmse", labelCol="rating",predictionCol="prediction")
evaluator_mae = RegressionEvaluator(metricName="mae", labelCol="rating",predictionCol="prediction")

print("\nStarting Cross-validation...")
# Folds training
for j in range(len(folds_list)):
    print("Fold:", j+1)
    for i, param in enumerate(parameters_to_test):
        als = ALS(rank=10, maxIter=10, regParam= param, userCol="userId", itemCol="movieId", ratingCol="rating", coldStartStrategy="drop")
        model = als.fit(folds_list[j][0])
        predictions = model.transform(folds_list[j][1])
        results_rmse[i,j] = evaluator_rmse.evaluate(predictions)
        results_mae[i,j] = evaluator_mae.evaluate(predictions)        
print("\nCross-validation is completed!")    
mean_rmse = np.mean(results_rmse, axis=1).reshape((num_models,1))
std_rmse = np.std(results_rmse, axis=1).reshape((num_models,1))
total_results_rmse = np.hstack((results_rmse, mean_rmse, std_rmse))
# np.savetxt('Rmse.csv', total_results_rmse, delimiter=',')
rmse= pd.DataFrame(total_results_rmse)
rmse.columns = ["rmse_1", "rmse_2", "rmse_3", "mean_rmse","std_rmse"]
rmse.index = ["model_1", "model_2", "model_3"]
rmse.to_csv("rmse_df.csv")
print(rmse)

mean_mae = np.mean(results_mae, axis=1).reshape((num_models,1))
std_mae = np.std(results_mae, axis=1).reshape((num_models,1))
total_results_mae = np.hstack((results_mae, mean_mae, std_mae))
# np.savetxt('MAE.csv', total_results_mae, delimiter=',')
mae= pd.DataFrame(total_results_mae)
mae.columns = ["mae_1", "mae_2", "mae_3", "mean_mae","std_mae"]
mae.index = ["model_1", "model_2", "model_3"]
mae.to_csv("mae_df.csv")
print(mae)

# Generating figures
print("Generating figures...")

# col_names_mae = ["mae_"+str(i) for i in range(1,4)]
# col_names_rmse = ["rmse_"+str(i) for i in range(1,4)]
# 
# x=np.arange(1,4,1)
# plt.figure(figsize=(12,8))
# plt.plot(x, mae.loc["model_1", col_names_mae].values, label = "MAE_model_1", marker = "o", linestyle='--', markersize=8, color = "red" )
# plt.plot(x, mae.loc["model_2", col_names_mae].values, label = "MAE_model_2", marker = "o", linestyle='--', markersize=8, color= "blue")
# plt.plot(x, mae.loc["model_3", col_names_mae].values, label = "MAE_model_3",  marker = "o", linestyle='--', markersize=8, color= "green")
# plt.plot(x, rmse.loc["model_1", col_names_rmse].values, label = "RMSE_model_1", marker = "*", linestyle='-', markersize=10, color = "red")
# plt.plot(x, rmse.loc["model_2", col_names_rmse].values, label = "RMSE_model_2", marker = "*", linestyle='-', markersize=10, color= "blue")
# plt.plot(x, rmse.loc["model_3", col_names_rmse].values, label = "RMSE_model_3",  marker = "*", linestyle='-', markersize=10, color= "green")
# plt.xticks(x)
# plt.xlabel("Fold")
# plt.legend()
# plt.savefig('Q2_rmse_mae.png')
# plt.show()

mae_means = mae["mean_mae"].values
mae_std = mae["std_mae"].values
rmse_mean = rmse["mean_rmse"].values
rmse_std = rmse["std_rmse"].values

objects = ["Model_1", "Model_2", "Model_3"]
y_pos = np.arange(len(objects))

width = 0.2 
plt.bar(y_pos, mae_means, width, align='center', alpha=0.5, color = "blue", label = "Mean MAE")
plt.bar(y_pos+width, mae_std, width, align='center', alpha=0.5, color = "red", label = "Std MAE" )
plt.xticks(y_pos+width/2, objects)

plt.bar(y_pos+width*2, rmse_mean, width, align='center', alpha=0.5, color = "green", label = "Mean RMSE")
plt.bar(y_pos+width*3, mae_std, width, align='center', alpha=0.5, color = "grey", label = "Std RMSE") 
# plt.ylabel('Total number of requests')
plt.title('Mean and std of MAE and RMSE')
plt.legend(loc='best')
plt.show()
plt.savefig('Q2_barplot_rmse_mae.png')

# plt.plot(x, mae.loc["model_1", col_names_mae].values, label = "MAE_model_1", marker = "o", linestyle='--', markersize=8, color = "red" )
# plt.plot(x, mae.loc["model_2", col_names_mae].values, label = "MAE_model_2", marker = "o", linestyle='--', markersize=8, color= "blue")
# plt.plot(x, mae.loc["model_3", col_names_mae].values, label = "MAE_model_3",  marker = "o", linestyle='--', markersize=8, color= "green")
# plt.xticks(x)
# plt.xlabel("Fold")
# plt.ylabel('MAE')
# plt.legend()
# plt.savefig('Q2_mae.png')
# plt.show()
# 
# plt.plot(x, rmse.loc["model_1", col_names_rmse].values, label = "Rmse_model_1", marker = "o", linestyle='--', markersize=8, color = "red" )
# plt.plot(x, rmse.loc["model_2", col_names_rmse].values, label = "Rmse_model_2", marker = "o", linestyle='--', markersize=8, color= "blue")
# plt.plot(x, rmse.loc["model_3", col_names_rmse].values, label = "Rmse_model_3",  marker = "o", linestyle='--', markersize=8, color= "green")
# plt.xticks(x)
# plt.xlabel("Fold")
# plt.ylabel("RMSE")
# plt.legend()
# plt.savefig('Q2_rmse.png')
# plt.show()


# Question 1C
split_1 = fold1.cache()
split_2 = fold2.cache()
split_3 = fold3.cache()

splits = [split_1, split_2, split_3]
als = ALS(maxIter=10, regParam=0.1, userCol="userId", itemCol="movieId", ratingCol="rating", coldStartStrategy="drop")

# convert the data to dense vector
def transData(data):
    return data.rdd.map(lambda x: [x[0], Vectors.dense(x[1])]).toDF(['movieId','features'])

#Preprocessing of genome-scores genome-scores.csv
df_gen = spark.read.load("Data/genome-scores.csv",
                     format="csv", inferSchema="true", header="true").cache()

df_check = df_gen.groupBy("movieId").count()
df_check.show(10)
print("Number of movies in genome-scores.csv", df_check.count())
print("Number of movies with 1128 tags:", df_check.filter(F.col("count") ==1128).count())

#The tags are the same so the tags with max relevance will appear to be most relevant
df_gen_max_relevance =df_gen.groupBy("movieId").agg(F.max("relevance")).cache()
df_gen_max_relevance.show()

df_dict = df_gen_max_relevance.toPandas().set_index('movieId').T.to_dict('list')

for x in list(df_dict.items()):
    tag_x = df_gen.filter((F.col("movieId")==x[0]) & (F.col("relevance")==x[1][0])).first()[1]
    df_dict[x[0]].append(tag_x)

#create DataFrame in Pandas
df_tagID = pd.DataFrame(data = df_dict.values(), index= df_dict.keys()).reset_index()
df_tagID.columns = ["movieId","max_relevance","tag"]
df_tagID.head()
df_tagID.to_csv("FINAL-genome-scores_analysis.csv")

# df_tagID = pd.read_csv("FINAL-genome-scores_analysis.csv").drop('Unnamed: 0', axis =1)
df_tags = pd.read_csv("Data/genome-tags.csv")

def get_results(df_pred, split, df_tagID, df_tags, list_movies_gen, list_tags):
    """ file_name - name of the files with labels for predicted labels => "predictions_kmean_1.csv"
        split - number
        df_tagID - FINAL-genome-scores_analysis.csv
        df_tags - Data/ml-25m/genome-tags.csv
    """
#     df_pred = spark.read.load(file_name, format="csv", inferSchema="true", header="true").cache()
                     
    top_clusters = df_pred.groupBy('prediction').count().orderBy("count", ascending=False).limit(3)
    top_clusters.show()
    top_clusters_list = top_clusters.rdd.map(lambda x: x[0]).collect()
    
    print("TOP3 clusters in split %i:" % split)
    print("\t\t", top_clusters_list)
        
#   print("Number of unique moviesId len(list_movies_gen)
    all_best_tags={}
    for i, cluster in enumerate(top_clusters_list):
        df_cluster = df_pred.filter(F.col('prediction') == cluster).cache()
        list_cluster = df_cluster.select("movieId").rdd.map(lambda x: x[0]).collect() 
        print("Split %i cluster %i (cluster label %i):" % (split, i+1, cluster))
        print("\tNumber of points:", len(list_cluster)) 
        
        # Not all moviesId can be found in genome-scores.csv
        list_common = list(set(list_cluster) & set(list_movies_gen))
        print("\tNumber of cluster's movies found in genome-scores.csv:", len(list_common) )

        max_relevant_df = df_tagID.loc[list_common, :].groupby("tag")["max_relevance"].agg(["count", "sum"]).sort_values("sum", ascending=False).iloc[:3,:].reset_index()
        
        print("\n\nTOP3 relevant tags")
        print(max_relevant_df)
            
        max_relevant_tagId = max_relevant_df["tag"].values
        
        # Not all tagIds can be found in genome-tags.csv
        list_common_tags = list(set(max_relevant_tagId) & set(list_tags))
        best_tags = df_tags.loc[list_common_tags,:]
        print("\nMatching tag names:")
        print(best_tags)
    return best_tags

list_movies_gen = df_tagID.movieId.values
df_tagID.set_index("movieId", inplace=True)
    
list_tags = df_tags.tagId.values
df_tags.set_index("tagId", inplace=True)

results =[]
for i, split_df in enumerate(splits):

    print("Training ALS in %d split" % i)
    model_als = als.fit(split_df)
    factors = model_als.itemFactors
    factors.toPandas().to_csv('factors_'+str(i+1)+'.csv')
    
    print("Training Kmeans in %d split" % i)
    df_transforned= transData(factors)
    kmeans = KMeans(initMode = 'k-means||').setK(25).setSeed(123)
    model_kmeans = kmeans.fit(df_transforned)
    
    print("Dealing with tags in %d split" % i)
    df_pred = model_kmeans.transform(df_transforned)
#     predictions.toPandas().to_csv('predictions_kmean_'+str(i+1)+'.csv')
    tags_split_i = get_results(df_pred, i, df_tagID, df_tags, list_movies_gen, list_tags)
    results.append([i, tags_split_i])

print("Successfully completed!")


spark.stop()

