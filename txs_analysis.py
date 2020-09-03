from pyspark import SparkContext, SparkConf
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.ml.feature import VectorAssembler, StandardScaler, StringIndexer
from pyspark.ml import Pipeline
from pyspark.ml.stat import Correlation

# Clustering model
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator
from sklearn.decomposition import KernelPCA

#utils
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
# Read dataset
# ======================
df = spark.read.csv("txsData/txs_smallAVG.csv", header=True, inferSchema=True).toDF('id', 'timestamp', 'n_in', 'n_out', 'amount_in', 'amount_out', 'change', 'date', 'usdAvg', 'usd_in', 'usd_out')
df.cache() # improves computational time 
df.printSchema()
# ========================
# Correlation Heat Map
# ========================
features_corr = ['timestamp', 'n_in', 'n_out', 'amount_in', 'amount_out', 'change']
assembler = VectorAssembler(inputCols=features_corr, outputCol='corr_features')
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
        fname = 'plots/' + name + '.png'
        plt.savefig(fname)
        plt.close()
        pass
heatmap(corrmatrix, features_corr, 'txs_corrmatrix')
# ==========================
# K-Means Model
# ==========================
features = ['timestamp', 'n_in', 'n_out', 'amount_in', 'amount_out', 'change']
assembler = VectorAssembler(inputCols=features, outputCol='features')
scalar = StandardScaler(inputCol='features', outputCol='scFeatures', withStd=True, withMean=True)
#pipeline
pipeline = Pipeline(stages=[assembler, scalar])
pipelineFit = pipeline.fit(df)
df_sc = pipelineFit.transform(df)
# Trains a k-means model.
def getSilhouette(df, model='KMeans'):
    silhouette_ls = []
    if model == 'KMeans':
        for i in range(1, 10):
            kmeans = KMeans().setK(i+1).setSeed(123)
            model_k = kmeans.fit(df)
            # Make predictions
            predictions = model_k.transform(df)
            # Evaluate clustering by computing Silhouette score
            evaluator = ClusteringEvaluator()
            silhouette_ls.append(round(evaluator.evaluate(predictions), 2))
    return silhouette_ls

silhouette_km = np.array(getSilhouette(df_sc))
x = [2, 3, 4, 5, 6, 7, 8, 9, 10]
plt.plot(x, silhouette_km, linewidth=1, marker='.', label='Kmeans')
plt.title('silhouette metrics: KMeans')
plt.grid()
plt.savefig('plots/KmeansSil.png')
plt.close()

sc.stop()