import pandas as pd
import numpy as np
from time import time

from classifier_choice import *
from well_load_fluids import *

well_data = wells_load("./oilwells/",'fluids.csv')


t0 = time()
piper = chainer('robust','tsne')
well_clusters = data_fit(piper,well_data.drop(['Well','TD','DTS_I7','DTS_MLFILLED','LITHESA10','LITHESA9_FILTR','LITHESA9_I4', 'LITHESA9_I8', 'LITHESA9_I8I4', 'PHIE_I7', 'PHIE_I8I7', 'PHIE_MLFILLED',  'FLUID', 'Flag_gas', 'Flag_oil', 'Flag_cond'],axis=1))
print(time()-t0)

well_data_out = well_data.copy()
well_data_out['x'] = well_clusters[:,0]
well_data_out['y'] = well_clusters[:,1]
well_data_out.head()
well_data_out.to_pickle("5_unlabeled_well_dataframe.pkl")