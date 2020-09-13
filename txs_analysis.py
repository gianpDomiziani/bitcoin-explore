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
spark.sparkContext.setLogLevel('ERROR')
# ======================
# Read dataset
# ======================
df = spark.read.csv("txsData/txs_small.csv", header=True, inferSchema=True).toDF('id', 'timestamp', 'n_in', 'n_out', 'amount_in', 'amount_out', 'change', 'date', 'usdAvg', 'SA', 'usd_in', 'usd_out', 'time', 'diffN')
df.cache() # improves computational time 
df.printSchema()
# ========================
# Correlation Heat Map
# ========================
features_corr = ['n_in', 'n_out', 'amount_in', 'amount_out', 'change', 'diffN', 'SA']
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
features = ['n_in', 'n_out', 'amount_in', 'amount_out', 'change', 'diffN', 'SA']
assembler = VectorAssembler(inputCols=features, outputCol='features')
scalar = StandardScaler(inputCol='features', outputCol='scFeatures', withStd=True, withMean=True)
#pipeline
pipeline = Pipeline(stages=[assembler, scalar])
pipelineFit = pipeline.fit(df)
df_sc = pipelineFit.transform(df)
# find the best k for a k-means model.
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
        best_k = silhouette_ls.index(np.min(silhouette_ls))+2
        print('****************************SILHOUETTE*************************************************')
        print(f'The best K is: {best_k} associated with a silhoutte of: {np.min(silhouette_ls)}')
    return silhouette_ls, best_k

sil_ls, k = getSilhouette(df_sc)
silhouette_km = np.array(sil_ls)
x = [2, 3, 4, 5, 6, 7, 8, 9, 10]
plt.plot(x, silhouette_km, linewidth=1, marker='.', label='Kmeans')
plt.xlabel('K')
plt.title('silhouette metrics: KMeans')
plt.grid()
plt.savefig('plots/KmeansSil_diffN.png')
plt.close()

# K Means with the best K
kmeans = KMeans().setK(k).setSeed(12)
model = kmeans.fit(df_sc)
predictions = model.transform(df_sc)
# Save the DF with labels
final = predictions.select('prediction', *features)
final.printSchema()
final.write.csv('txsDataWithLabels', header=True)
sc.stop()