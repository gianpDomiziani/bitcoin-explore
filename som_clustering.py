import susi
from susi.SOMPlots import plot_umatrix

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ==== Preprocessing ====
X = pd.read_csv('txsDataWithLabels/txsWithLabels.csv', header=0, usecols=['prediction','n_in', 'n_out', 'amount_in', 'amount_out',
                                                             'change', 'diffN', 'SA'])

y = X['prediction'].to_numpy()
X = X.drop('prediction', axis=1).to_numpy()
plt.scatter(X[:, 3], X[:, 4], c=y)
plt.show()
# Classify an plot
som = susi.SOMClustering(
    n_rows=30,
    n_columns=30,
    random_state=12
)

som.fit(X)
print('SOM fitted!')
#U-Matrix
u_matrix = som.get_u_matrix()
plot_umatrix(u_matrix, 30, 30)
plt.show()

#BMUs
som_array = som.unsuper_som_  # weight vectors of the SOM
bmus = som.get_bmus(X, som_array)  # Best Match Unit per sample


