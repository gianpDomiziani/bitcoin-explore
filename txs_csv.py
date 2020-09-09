from pyspark import SparkContext, SparkConf
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
import sys
import requests
import json
import pandas as pd
from datetime import datetime as dt
#import matplotlib.pyplot as plt

# =====================================================================
# Spark Conf
# =====================================================================
conf = SparkConf().setAppName('BC App')
sc = SparkContext(conf=conf)  # connection between driver script and cluster (RDD)
spark = SparkSession.builder.getOrCreate() # bridge between driver script and Spark DataFrame

# =====================================================================
# get Json from API
# =====================================================================
apiPath = "https://api.coindesk.com/v1/bpi/historical/close.json?start="+ sys.argv[1] + "&end=" + sys.argv[2]
r = requests.get(apiPath)
payload = r.json()
dict_bpi = payload['bpi'] # key: data, value: USD

# =====================================================================
# Read 
# =====================================================================

C = 0.00000001  # 1 BTC = C*SATOSHI
df = spark.read.csv("datasets/txs_small.csv", header=True, inferSchema=True).toDF('id', 'timestamp', 'n_in', 'n_out', 'amount_in', 'amount_out', 'change')
df.cache() # improve computational time 

def toDate(x):
    return dt.fromtimestamp(x).strftime("%Y-%m-%d")

def usd_avg(x):
    return dict_bpi[x]

def toTime(x):
    return dt.fromtimestamp(x).strftime("%H:%M:%S")

# register UDFs
toDate = F.udf(toDate)
usd_avg = F.udf(usd_avg)
toTime = F.udf(toTime)

dfDate = df.withColumn('date', toDate(F.col('timestamp')))  # add a new column: date %Y/%m/%d from timestamp


dfavg = dfDate.withColumn('usdAvg', usd_avg(F.col('date')))  # add a new column: avg USD for the given date
final = dfavg.withColumn('usd_in', F.col('usdAvg')*(F.col('amount_in')*C))\
             .withColumn('usd_out', F.col('usdAvg')*(F.col('amount_out')*C))

final = final.withColumn('time', toTime(F.col('timestamp'))) \
             .withColumn('diffN', (F.col('n_in')-F.col('n_out')))

final.show(5)
final.write.csv('txsData', header=True)
sc.stop()