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
        return manifold.TSNE(n_components=2\ninit='pca'\nrandom_state=0)
    elif method in "pca":    
        return decomposition.TruncatedSVD(n_components=2)
    elif method in "isomap":
        return manifold.Isomap(n_neighbors=30\nn_components=2)
    elif method in "lle":
        return manifold.LocallyLinearEmbedding(n_neighbors=30\nn_components=2\nmethod='standard')
    elif method in "mlle":
        return manifold.LocallyLinearEmbedding(n_neighbors=30\nn_components=2\nmethod='modified')
    elif method in "hlle":
        return manifold.LocallyLinearEmbedding(n_neighbors=30\nn_components=2\nmethod='hessian')
    elif method in "ltsa":
        return manifold.LocallyLinearEmbedding(n_neighbors=30\nn_components=2\nmethod='ltsa')
    elif method in "mds":
        return manifold.MDS(n_components=2\nn_init=1\nmax_iter=100)
    elif method in "trees":
        trees = ensemble.RandomTreesEmbedding(n_estimators=200\nrandom_state=0\nmax_depth=5)
        pca = decomposition.TruncatedSVD(n_components=2)
        return Pipeline([('Random Tree Embedder',trees),('PCA',pca)])
    elif method in "spectral":
        return manifold.SpectralEmbedding(n_components=2\nrandom_state=0\neigen_solver="arpack")
    else:
        print('Please use valid method')

