from pyspark import SparkContext, SparkConf
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.ml.feature import VectorAssembler, StandardScaler, StringIndexer
from pyspark.ml import Pipeline
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator
from pyspark.ml.stat import Correlation


import pandas as pd
import numpy as np
import seaborn as sns
from datetime import datetime as dt
import matplotlib.pyplot as plt
# =======================
# Spark Conf
# =======================
conf = SparkConf().setAppName('BC App')
sc = SparkContext(conf=conf)  # connection between driver script and cluster (RDD)
spark = SparkSession.builder.getOrCreate() # bridge between driver script and Spark DataFrame
# ======================
# Read 
# ======================
df = spark.read.csv("txsData/txs_smallAVG.csv", header=True, inferSchema=True).toDF('id', 'timestamp', 'n_in', 'n_out', 'amount_in', 'amount_out', 'change', 'date', 'usdAvg', 'usd_in', 'usd_out')
df.cache() # improves computational time 
df.printSchema()
# ========================
# Correlation Heat Map
# ========================
features = ['timestamp', 'n_in', 'n_out', 'amount_in', 'amount_out', 'change']
assembler = VectorAssembler(inputCols=features, outputCol='corr_features')
vector_corr = assembler.transform(df).select('corr_features')
matrix = Correlation.corr(vector_corr, 'corr_features').collect()[0][0]
corrmatrix = matrix.toArray().tolist()  # Correlation Matrix

# Plot Correlation HeatMap
def heatmap(X, columns, name):

        df = pd.DataFrame(X, columns)
        # Set up the matplotlib figure
        f, ax = plt.subplots(figsize=(10, 10))
        plt.title(name)
        sns.heatmap(df, vmin=-1, vmax=1, center=0, annot=True, xticklabels=True, yticklabels=True)
        fname = name + '.png'
        plt.savefig(fname)
        plt.close()
        pass
heatmap(corrmatrix, features, 'txs_corrmatrix')


sc.stop()