import pandas as pd
import numpy as np
from time import time

from classifier_choice import *
from well_load import *

well_data = wells_load("wells")


t0 = time()
piper = chainer('robust','tsne')
well_clusters = data_fit(piper,well_data.drop(['Well','TD'],axis=1))
print(time()-t0)

np.save('well_clusters.npy',well_clusters)