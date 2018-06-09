from functools import partial
from random import random
from threading import Thread
import time

from bokeh.models import ColumnDataSource, TableColumn
from bokeh.plotting import curdoc, figure
from bokeh.models.widgets import Button, Dropdown
from bokeh.layouts import row, column
from bokeh.models import CustomJS, ColumnDataSource
from bokeh.plotting import figure, output_file, show
from bokeh.models.widgets import DataTable, DateFormatter, TableColumn
from bokeh.models import Circle
from bokeh.models import PolyDrawTool
from tornado import gen

import numpy as np
import sys
import pandas as pd

import lasio

from well_load import wells_load
colors = ["springgreen","tomato","steelblue","tan","teal","thistle","turquoise","violet"]

#load data
df = wells_load("./static/data/peters_wells/")

print(df.keys())
X = np.random.rand(len(df), 10)
cluster = [0] * len(df)
color = ["black"] * len(df)

df["color"] = pd.Series(color, index=df.index)
df["cluster"] = pd.Series(cluster, index=df.index)
df["x"] = pd.Series(X[:, 0], index=df.index)
df["y"] = pd.Series(X[:, 1], index=df.index)

source = ColumnDataSource(data=df)

# This is important! Save curdoc() to make sure all threads
# see the same document.
doc = curdoc()

TOOLS = "lasso_select, pan,box_select,reset,help,poly_draw"

# create a new plot and add a renderer
left = figure(tools=TOOLS, width=300, height=300, title=None, x_range=[0, 1], y_range=[0, 1], output_backend="webgl")
left.circle('x', 'y', source=source, color='color')

# create another new plot and add a renderer
right = figure(tools=TOOLS, width=300, height=300, title=None, x_range=[df['GR'].min(), df['GR'].max()], y_range=[df['TD'].min(), df['TD'].max()], output_backend="webgl")
right.circle('GR', 'TD', source=source, color='color')

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

run_button = Button(label="Run Dim-Red")

#This is our Cluster making hack
cluster_button = Button(label="Add Cluster")

menu = [(key, "_".join(key.lower().split(" "))) for key in df.keys()]
dropdown = Dropdown(label="Available Logs", button_type="warning", menu=menu)

def function_to_call(attr, old, new):
    print(dropdown.value)

dropdown.on_change('value', function_to_call)
dropdown.on_click(function_to_call)

i = 0
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

doc.add_root(row(left, right, column(dropdown, cluster_button, run_button)))