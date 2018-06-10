import os, sys
import pandas as pd
import numpy as np
import welly as welly
from welly import Well

def wells_load(path2files,fluids,resample=0):
    filelist = os.listdir(path2files)
    fluid_csv=fluids
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

        #Adding labels for fluids
        #Reading in the fluid information and setting a fluid column in well dataframe
        fluid_labels = pd.read_csv(fluid_csv)
        fluids = fluid_labels[fluid_labels['Well'] == f]
        fluids = fluids.fillna(value=99999)

        #Setting top and base limits for various fluids
        topgas=int(fluids.iloc[0]['topgas'])
        basegas=int(fluids.iloc[0]['basegas'])
        topoil=int(fluids.iloc[0]['topoil'])
        baseoil=int(fluids.iloc[0]['baseoil'])
        topcond=int(fluids.iloc[0]['topcond'])
        basecond=int(fluids.iloc[0]['basecond'])

        #Assigning fluid fluid_labels
        #Gas=1, Oil=2 Condensate=4
        #Gas and Oil = 3
        #Gas and Condensate = 5
        #Oil and Condensate = 6
        w_df['FLUID'] = 0
        w_df['Flag_gas']=w_df['TD'].between(topgas,basegas).astype('int')*1
        w_df['Flag_oil']=w_df['TD'].between(topoil,baseoil).astype('int')*2
        w_df['Flag_cond']=w_df['TD'].between(topcond,basecond).astype('int')*4
        w_df['FLUID']=w_df['Flag_gas'] + w_df['Flag_oil'] + w_df['Flag_cond']
        w_df.drop(['Flag_gas', 'Flag_oil','Flag_cond'], axis=1)

        #Resampling by a factor
        if resample>0:
            w_df = w_df.iloc[::int(resample),:]

        #Drops columns only with NaNs
        a = w_df.dropna(axis=1, how='all')

        #Replace NaNs with mean of the log
        b = a.fillna(method='ffill')
        c = b.fillna(method='bfill')
        wellsdataframe = pd.concat([wellsdataframe, c], axis=0)
    return wellsdataframe
