# The project is done as a part of Assignments in Scalable Machine Learning course 
# at the University of Sheffield

from pyspark.sql import SparkSession
import pyspark.sql.functions as F
import pickle

spark = SparkSession.builder.master("local[2]")\
	.config("spark.local.dir","/fastdata/my_name")\
	.appName("COM6012_Assignment2").getOrCreate()

sc = spark.sparkContext
sc.setLogLevel("WARN")
df_large =  spark.read.csv("/data/my_name/train_set.csv", header=True)

large_categor = ['Blind_Make','Blind_Model', 'Blind_Submodel', 'Model_Year']
string_categories = ['Cat'+str(i) for i in range(1, 13)]+ ["NVCat"]

# (df_s0, rest) = df_large.select(large_categor+string_categories).randomSplit([0.1, 0.9], 123)
df_s = df_large.select(large_categor).cache()

top_N = 20
mapping_dict = {}
for categor in large_categor:
    value_counts = df_s.groupby(categor).count().orderBy("count", ascending=False).cache()
    top_categor = value_counts.limit(top_N).select(categor).rdd.map(lambda x:x[0]).collect()
    dict_id_categor = dict(enumerate(top_categor, 1)) #start enumerating from zero to reserve 0 for missing and unfrequent data 
    list_categor_id = [(value,key) for (key, value) in dict_id_categor.items()]
    last_value = value_counts.filter(value_counts[categor]== top_categor[-1]).select("count").rdd.map(lambda x:x[0]).collect()[0]
    other_categor = value_counts.filter(value_counts["count"] <= last_value).rdd.map(lambda x: x[0]).collect()
    zeros = [0]*len(other_categor[1:])
    other_categor_zeros = list(zip(other_categor, zeros))
    all_mapping_list = list_categor_id + other_categor_zeros
    mapping_dict[categor] = dict(all_mapping_list)
    print("Analysis of %s is completed" % categor)
    

# print("Columns:", df_s0.columns)
print("Analysing small subset for string categories\n")
(df_s, rest) = df_large.select(string_categories).randomSplit([0.005, 0.995], 123)
df_s.cache()

for categor in string_categories:
    none_values = df_s.filter(df_s[categor].isNull()==True).count()
    if none_values != 0:
        print("Column %s had %i None values" % (categor, none_values))
        df_small = df_s.withColumn(categor, F.when(df_s[categor].isNull()==True, "ZZZ").otherwise(df_s[categor]))

for categor in string_categories:
    unique = df_s.select(categor).distinct().rdd.map(lambda r: r[0]).collect()
    dict_id_categor = dict(enumerate(unique,1))
    dict_categor_id = {value:key for (key, value) in dict_id_categor.items()}
    mapping_dict[categor] = dict_categor_id

for key in mapping_dict:
    print(key,":", len(mapping_dict[key]))

categor="Cat12"
if 'ZZZ' not in mapping_dict[categor]:
    mapping_dict[categor] = {'ZZZ': 1, 'F': 2, 'E': 3, 'B': 4, 'D': 5, 'C': 6, 'A': 7}

mapping_dict["Blind_Submodel"]['AE.3.0']=0

with open('/home/my_name/mapping_dict3.txt', 'wb') as handle:
    pickle.dump(mapping_dict, handle)


spark.stop()



