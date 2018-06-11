from sklearn import datasets, manifold, decomposition, ensemble, discriminant_analysis, preprocessing
from sklearn.pipeline import Pipeline
from MulticoreTSNE import MulticoreTSNE as TSNE

"""
Please choose a classifier:
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
def scaler_choice(method='standard'):
    if method in "standard":
        return preprocessing.StandardScaler()
    elif method in "robust":    
        return preprocessing.RobustScaler()
    elif method in "sphere":
        return preprocessing.Normalizer()
    else:
        print('Not a valid choice')

def classifier_choice(method='tsne', neighbors=30, dimensions=2):
    if method in "tsne":
        return TSNE(n_components=dimensions, n_jobs=8, perplexity=30, verbose=1)
    elif method in "pca":    
        return decomposition.TruncatedSVD(n_components=dimensions)
    elif method in "isomap":
        return manifold.Isomap(n_neighbors=neighbors, n_components=dimensions)
    elif method in "lle":
        return manifold.LocallyLinearEmbedding(n_neighbors=neighbors, n_components=dimensions, method='standard')
    elif method in "mlle":
        return manifold.LocallyLinearEmbedding(n_neighbors=neighbors, n_components=dimensions, method='modified')
    elif method in "hlle":
        return manifold.LocallyLinearEmbedding(n_neighbors=neighbors, n_components=dimensions, method='hessian')
    elif method in "ltsa":
        return manifold.LocallyLinearEmbedding(n_neighbors=neighbors, n_components=dimensions, method='ltsa')
    elif method in "mds":
        return manifold.MDS(n_components=dimensions, n_init=1, max_iter=100)
    elif method in "trees":
        trees = ensemble.RandomTreesEmbedding(n_estimators=200, max_depth=5)
        pca = decomposition.TruncatedSVD(n_components=dimensions)
        return Pipeline([('Random Tree Embedder',trees),('PCA',pca)])
    elif method in "spectral":
        return manifold.SpectralEmbedding(n_components=dimensions, eigen_solver="arpack")
    else:
        print('Please use valid method')

def chainer(scaler='standard',classifier='tsne'):
    scaling = scaler_choice(scaler)
    classifying = classifier_choice(classifier)
    return Pipeline([(scaler,scaling),(classifier,classifying)])

def data_fit(classifier, data):
    return classifier.fit_transform(data)
