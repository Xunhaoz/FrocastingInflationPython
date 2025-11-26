import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

data = pd.read_csv("data/data.csv", index_col=0)
data['dummy'] = 0
data.loc['2008-11-01', 'dummy'] = 1
pca = PCA(n_components=4).fit_transform(data)
pca = pd.DataFrame(pca, columns=[f'PCA_{_}' for _ in range(pca.shape[1])], index=data.index)
data = pd.concat([data, pca], axis=1)
data.describe().to_csv('data/data_description.csv')

for col in data.columns:
    ax = data[[col]].plot(kind='hist', subplots=False)
    ax.figure.savefig(f'plots/hist_{col}.png')
