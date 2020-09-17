# The project is done as a part of Assignments in Scalable Machine Learning course 
# at the University of Sheffield

from pyspark.sql import SparkSession
from pyspark.sql.types import StringType, datetime, DateType, TimestampType
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time
import pyspark.sql.functions as F
from datetime import datetime

spark = SparkSession.builder.master("local[2]").appName("COM6012 Assignment1").getOrCreate()

sc = spark.sparkContext
sc.setLogLevel("WARN")

df_0 = spark.read.text("Data/NASA_access_log_Jul95.gz").cache()

# Question 1A

# exploratory analysis showed that there is one record not from Jul/1995
# so we can filter data at first
df = df_0.filter(df_0.value.contains("/Jul/1995")).cache()

#Extract date and time
regExpDate = '\d{2}\/Jul/1995:\d{2}:\d{2}:\d{2}'
df = df.withColumn('date_time', F.regexp_extract('value', regExpDate, 0))
# df.show(5)

# create new columns - transform into Timestamp
udf_new_date = F.UserDefinedFunction(lambda x: datetime.strptime(x, '%d/%b/%Y:%H:%M:%S'), TimestampType())
df = df.withColumn('new_date', udf_new_date('date_time'))
df.printSchema()

df = df.withColumn('day', F.dayofmonth('new_date'))
df = df.withColumn('hour', F.hour('new_date'))

# categorise hour intervals

udf_categor = F.UserDefinedFunction(lambda x : '1' if x < 4 else ("2" if x < 8 else ("3" if x<12 else ("4" if x<16 else("5" if x<20 else "6")))))
df = df.withColumn('hour_categor', udf_categor('hour'))

grouped_categor_hour = df.groupby('hour_categor').agg(F.count("value")).alias("count").orderBy("hour_categor").cache()
# grouped_categor_hour.show()

grouped_day =  df.groupby('day').count().orderBy('day').cache()
grouped_day.show(31)

# There are only 28 days found in the dataset
grouped_categor_hour = grouped_categor_hour.withColumn('average_28', F.col("count(value)")/28)

# However, there are 31 days in July, so assuming 0 requests for the rest 3 days we can have: 
grouped_categor_hour = grouped_categor_hour.withColumn('average_31', F.col("count(value)")/31)

answer_df = grouped_categor_hour.select("hour_categor", 'average_28', 'average_31')

udf_categor_decode = F.UserDefinedFunction(lambda x : "00:00:00-03:59:59" if x == "1" \
	else ("04:00:00-07:59:59" if x =="2" else ("08:00:00-11:59:59" if x=="3" \
	else ("12:00:00-15:59:59" if x=="4" else("16:00:00-19:59:59" if x=="5" \
	else "20:00:00-23:59:59")))))

answer_df = answer_df.withColumn('hour_categor_times', udf_categor_decode("hour_categor"))
answer_df.show()
answer_df.toPandas().to_csv('Q1_averages.csv')

# Question 1B

avgs = answer_df.select("average_28").rdd.map(lambda x: x[0]).collect()
categor_hours = answer_df.select("hour_categor_times").rdd.map(lambda x: x[0]).collect()
categor_hours

objects = categor_hours
y_pos = np.arange(len(objects))
plt.bar(y_pos, avgs, align='center', alpha=0.5, color = "blue")
plt.xticks(y_pos, objects, rotation='vertical')
plt.ylabel('Average number')
plt.title('Average number of requests per day in July')
plt.gcf().subplots_adjust(bottom=0.3)
plt.savefig('Q1_averages.png')
plt.show()

# Question 1C

# exploratory analysis showed that there is one record not from Jul/1995
# so we can filter data at first

# df_0 = spark.read.text("NASA_access_log_Jul95.gz").cache()
df = df.filter(df.value.contains(".html")).cache()
df.count()


#as from the post on 28/02/20,
# /aa/xx.html` and `/bb/xx.html` are considered to have the same filename xx

udf_file = F.UserDefinedFunction(lambda x: x.split(".html")[0].split('/')[-1], StringType())
df = df.withColumn('file_name', udf_file('value'))

df_grouped = df.groupby('file_name').count().orderBy("count", ascending=False).limit(20)
df_grouped.show(truncate=False)
df_grouped.toPandas().to_csv('Q1_filenames.csv')

# Question 1D
file_names = df_grouped.select("file_name").rdd.map(lambda x:x[0]).collect()
file_num = df_grouped.select("count").rdd.map(lambda x:x[0]).collect()

objects = file_names
y_pos = np.arange(len(objects))

plt.bar(y_pos, file_num, align='center', alpha=0.5, color = "blue")
plt.xticks(y_pos, objects, rotation='vertical')
# plt.xticks(y_pos, objects, rotation=45)
plt.ylabel('Total number of requests')
plt.title('Top 20 files')
plt.gcf().subplots_adjust(bottom=0.3)
plt.savefig('Q1_filenames.png')
plt.show()

spark.stop()
