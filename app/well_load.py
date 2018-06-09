import os, sys
import pandas as pd
import numpy as np
import welly as welly
from welly import Well

def wells_load(path2files):
    filelist = os.listdir(path2files)
    wellsdataframe=pd.DataFrame()
    for f in filelist:
        #Read in the LAS file
        w = Well.from_las(path2files+f)

        #convert well data to a pandas dataframe
        w_df = pd.DataFrame(w.data)

        #add well name to first column
        w_df['Well'] = f

        #add depth column
        dt = w.data['DT']
        w_df["TD"] = dt.basis

        #Drops columns only with NaNs
        a = w_df.dropna(axis=1, how='all')

        #Replace NaNs with mean of the log
        b = a.fillna(method='ffill')
        c = b.fillna(method='bfill')
        wellsdataframe = pd.concat([wellsdataframe, c], axis=0)
        return wellsdataframe
