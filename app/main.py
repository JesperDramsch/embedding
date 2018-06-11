from functools import partial
from random import random
from threading import Thread
import time

from bokeh.models import ColumnDataSource, TableColumn, Plot, Range1d
from bokeh.plotting import curdoc, figure
from bokeh.models.widgets import Button, Dropdown
from bokeh.layouts import row, column
from bokeh.models import CustomJS, ColumnDataSource
from bokeh.plotting import figure, output_file, show
from bokeh.models.widgets import DataTable, DateFormatter, TableColumn
from bokeh.models.glyphs import ImageURL
from bokeh.models import Circle
from bokeh.models import PolyDrawTool
from tornado import gen
import matplotlib.pyplot as plt

import numpy as np
import sys
import pandas as pd

import lasio
from well_load import *
from classifier_choice import *

import matplotlib.colors as colors
from mpl_toolkits.axes_grid1 import make_axes_locatable


# 1=sandstone  2=c_siltstone   3=f_siltstone 
# 4=marine_silt_shale 5=mudstone 6=wackestone 7=dolomite
# 8=packstone 9=bafflestone
facies_colors = ['#F4D03F', '#F5B041','#DC7633','#6E2C00',
       '#1B4F72','#2E86C1', '#AED6F1', '#A569BD', '#196F3D']

facies_labels = ['a', 'b', 'c', 'd', 'e',
                 'f', 'g','h', 'i']
#facies_color_map is a dictionary that maps facies labels
#to their respective colors
facies_color_map = {}
for ind, label in enumerate(facies_labels):
    facies_color_map[label] = facies_colors[ind]

def label_facies(row, labels):
    return labels[ row['cluster'] -1]
    
#training_data.loc[:,'cluster'] = training_data.apply(lambda row: label_facies(row, facies_labels), axis=1)

def  make_facies_log_plotmake_fa(logs, facies_colors):
    import matplotlib.colors as colors
    #make sure logs are sorted by depth
    logs = logs.sort_values(by='TD')
    cmap_facies = colors.ListedColormap(
            facies_colors[0:len(facies_colors)], 'indexed')
    
    ztop=logs.TD.min(); zbot=logs.TD.max()
    
    cluster=np.repeat(np.expand_dims(logs['cluster'].values,1), 100, 1)

    f, ax = plt.subplots(nrows=1, ncols=6, figsize=(8, 12))
    ax[0].plot(logs.GR, logs.TD, '-g')
    ax[1].plot(logs.LITHESA9_I4, logs.TD, '-')
    ax[2].plot(logs.LITHESA9_I8I4, logs.TD, '-', color='0.5')
    ax[3].plot(logs.RDEP, logs.TD, '-', color='r')
    ax[4].plot(logs.RMED, logs.TD, '-', color='black')
    im=ax[5].imshow(cluster, interpolation='none', aspect='auto',
                    cmap=cmap_facies,vmin=1,vmax=9)
    
    divider = make_axes_locatable(ax[5])
    cax = divider.append_axes("right", size="20%", pad=0.05)
    cbar=plt.colorbar(im, cax=cax)
    cbar.set_label((17*' ').join([' a ', 'b', 'c', 
                                'd', ' e ', ' f ', ' g  ', 
                                ' h ', ' i ']))
    cbar.set_ticks(range(0,1)); cbar.set_ticklabels('')
    
    for i in range(len(ax)-1):
        ax[i].set_ylim(ztop,zbot)
        ax[i].invert_yaxis()
        ax[i].grid()
        ax[i].locator_params(axis='x', nbins=3)
    
    ax[0].set_xlabel("GR")
    ax[0].set_xlim(logs.GR.min(),logs.GR.max())
    ax[1].set_xlabel("LITHESA9_I4")
    ax[1].set_xlim(logs.LITHESA9_I4.min(),logs.LITHESA9_I4.max())
    ax[2].set_xlabel("LITHESA9_I8I4")
    ax[2].set_xlim(logs.LITHESA9_I8I4.min(),logs.LITHESA9_I8I4.max())
    ax[3].set_xlabel("RDEP")
    ax[3].set_xlim(logs.RDEP.min(),logs.RDEP.max())
    ax[4].set_xlabel("RMED")
    ax[4].set_xlim(logs.RMED.min(),logs.RMED.max())
    ax[5].set_xlabel('Cluster')
    
    ax[1].set_yticklabels([]); ax[2].set_yticklabels([]); ax[3].set_yticklabels([])
    ax[4].set_yticklabels([]); ax[5].set_yticklabels([])
    ax[5].set_xticklabels([])
    f.suptitle('Well: %s'%logs.iloc[0]['Well'], fontsize=14,y=0.94)
    return f
    


colors = ["springgreen","tomato","steelblue","tan","teal","thistle","turquoise","violet"]

#load data
df = pd.read_pickle("app/static/well_dataframe.pkl")
#df = df[np.where(df.Well == "16_7-11.las", True, False)]

print(df.keys())
#X = np.random.rand(len(df), 10)
cluster = [0] * len(df)
color = ["black"] * len(df)

df["color"] = pd.Series(color, index=df.index)
df["cluster"] = pd.Series(cluster, index=df.index)
#df["x"] = pd.Series(X[:, 0], index=df.index)
#df["y"] = pd.Series(X[:, 1], index=df.index)

#fig = make_facies_log_plotmake_fa(df, facies_colors)
#fig.savefig("app/static/mpl_fig.png")
source = ColumnDataSource(data=df)

# This is important! Save curdoc() to make sure all threads
# see the same document.
doc = curdoc()

TOOLS = "lasso_select, pan,box_select,reset,help,poly_draw"

# create a new plot and add a renderer
left = figure(tools=TOOLS, width=300, height=300, title=None, x_range=[df['x'].min(), df['x'].max()], y_range=[df['y'].min(), df['y'].max()], output_backend="webgl")
left.circle('x', 'y', source=source, color='color', size=0.1)

# create another new plot and add a renderer
right = figure(tools=TOOLS, width=300, height=300, title=None, x_range=[df['GR'].min(), df['GR'].max()], y_range=[df['TD'].min(), df['TD'].max()], output_backend="webgl")
right.circle('GR', 'TD', source=source, color='color', size=0.1)

#Functions needed to update the graph based on long running task
@gen.coroutine
def update(x, y):
    source.data = dict(x=x, y0=y, y1=np.sin(y))

def blocking_task():
    while True:
        # do some blocking computation
        X = np.random.rand(100, 10)
        doc.add_next_tick_callback(partial(update, x=X[:, 0], y=X[:, 1]))

def run_task():
    thread = Thread(target=blocking_task)
    thread.start()

def make_plot():
    #fig = make_facies_log_plotmake_fa(source.to_df(), facies_colors)
    #fig.savefig("app/static/mpl_fig.png")
    print(source.to_df().keys())
    #print(source.to_df().color.unique(), source.to_df().cluster.unique())
    #source.to_df().to_pickle("app/static/5_wells_p_30_no_pca_with_clusters.pkl")

run_button = Button(label="Run Dim-Red")
#This is our Cluster making hack
cluster_button = Button(label="Add Cluster")

menu = [(key, "_".join(key.lower().split(" "))) for key in df.keys()]
dropdown = Dropdown(label="Available Logs", button_type="warning", menu=menu)

def function_to_call(attr, old, new):
    print(dropdown.value)

dropdown.on_change('value', function_to_call)
dropdown.on_click(function_to_call)

i = 1
def callback():
    global i
    i = i + 1
    source.callback = CustomJS(args=dict(colors=colors, clusters=i), code="""
	 var inds = cb_obj.getv('selected')['1d'].indices;
         var d1 = cb_obj.data;
	 if (inds.length == 0) { return; }
	 for (i = 0; i < inds.length; i++) {
             d1['cluster'][inds[i]] = clusters;
             d1['color'][inds[i]] = colors[clusters];
	 }
     cb_obj.change.emit();
    """)
    source.selected.indices = []


cluster_button.on_click(callback)
run_button.on_click(run_task)

#make_plots_button.on_click(make_plot)

save_callback = CustomJS(args=dict(source=source), code="""
    function download(content, fileName, contentType) {
        var a = document.createElement("a");
        var file = new Blob([content], {type: contentType});
        a.href = URL.createObjectURL(file);
        a.download = fileName;
        a.click();
    }
    console.log(source.data);
    download(JSON.stringify(source.data['cluster']), "output.json", 'text\plain');
""")


make_plots_button = Button(label="Make Well Logs", callback=save_callback)

source.callback = CustomJS(args=dict(colors=colors, clusters=i), code="""
	 var inds = cb_obj.getv('selected')['1d'].indices;
         var d1 = cb_obj.data;
	 if (inds.length == 0) { return; }
	 for (i = 0; i < inds.length; i++) {
             d1['cluster'][inds[i]] = clusters;
             d1['color'][inds[i]] = colors[clusters];
	 }
     cb_obj.change.emit();
    """)

data_table_force_change = CustomJS(args=dict(source=source), code="""
    source.change.emit()
""")
source.js_on_change('data', data_table_force_change)

xdr = Range1d(start=0, end=300)
ydr = Range1d(start=0, end=500)

plot = Plot(title=None, x_range=xdr, y_range=ydr, plot_width=300, plot_height=500,
    h_symmetry=False, v_symmetry=False, min_border=0, toolbar_location=None)

#image1 = ImageURL(url="./static/mpl_fig.png")#, x=0, y="y1", w="w1", h="h1", anchor="center")
#plot.add_glyph(source, image1)
x_range = (0, 300) # could be anything - e.g.(0,1)
y_range = (0, 500)
p = figure(x_range=x_range, y_range=y_range)
#img_path = 'https://bokeh.pydata.org/en/latest/_static/images/logo.png'
img_path = "app/static/mpl_fig.png"
p.image_url(url=[img_path],x=x_range[0],y=y_range[1],w=x_range[1]-x_range[0],h=y_range[1]-y_range[0])

doc.add_root(row(column(row(left, right), row(p)), column(dropdown, cluster_button, run_button, make_plots_button)))