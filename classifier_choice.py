from sklearn import datasets, manifold, decomposition, ensemble, discriminant_analysis
from sklearn.pipeline import Pipeline
"""
Please chose a classifier:
pca = Principal Component Analysis
isomap = Isomapping
lle = Local Linear Embedding
mlle = Modified LLE
hlle = Hessian LLE
ltsa = local tangent space alignment
mds = Multi-dimensional scaling
trees = Random Trees Embedding
tsne = t-distributed stochastic neighbor embedding
"""


def classifier_choice(method='tsne'):
    X = data.copy()
    if method in "tsne":
        return manifold.TSNE(n_components=2, init='pca', random_state=0)
    elif method in "pca":    
        return decomposition.TruncatedSVD(n_components=2)
    elif method in "isomap":
        return manifold.Isomap(n_neighbors=30, n_components=2)
    elif method in "lle":
        return manifold.LocallyLinearEmbedding(n_neighbors=30, n_components=2, method='standard')
    elif method in "mlle":
        return manifold.LocallyLinearEmbedding(n_neighbors=30, n_components=2, method='modified')
    elif method in "hlle":
        return manifold.LocallyLinearEmbedding(n_neighbors=30, n_components=2, method='hessian')
    elif method in "ltsa":
        return manifold.LocallyLinearEmbedding(n_neighbors=30, n_components=2, method='ltsa')
    elif method in "mds":
        return manifold.MDS(n_components=2, n_init=1, max_iter=100)
    elif method in "trees":
        trees = ensemble.RandomTreesEmbedding(n_estimators=200, random_state=0, max_depth=5)
        pca = decomposition.TruncatedSVD(n_components=2)
        return Pipeline([('Random Tree Embedder',trees),('PCA',pca)])
    elif method in "spectral":
        return manifold.SpectralEmbedding(n_components=2, random_state=0, eigen_solver="arpack")
    else:
        print('Please use valid method')

def data_fit(classifier,data):
	return classifier.fit_transform(data)