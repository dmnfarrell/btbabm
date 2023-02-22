#!/usr/bin/env python

"""
    ABM util classes for BTB
    Created Jan 2023
    Copyright (C) Damien Farrell
"""

import sys,os,random,time,string
import subprocess
import math
import numpy as np
import pandas as pd
pd.set_option('display.width', 200)
pd.set_option('max_colwidth', 200)
import pylab as plt
import networkx as nx
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio import SeqIO, AlignIO
import json
import toytree

def random_sequence(length=50):
    seq=''
    for count in range(length):
        seq += random.choice("CGTA")
    return seq

def random_color(seed=None):
    if seed != None:
        random.seed = seed
    return tuple([np.random.random_sample() for i in range(3)])

def random_hex_color():
    r = lambda: random.randint(0,255)
    c='#%02X%02X%02X' % (r(),r(),r())
    return c

def create_closest_n_graph(n, num_closest_nodes):
    nodes = [i for i in range(n)]
    G = nx.Graph()
    G.add_nodes_from(nodes)

    def euclidean_distance(node1, node2):
        x1, y1 = node1
        x2, y2 = node2
        return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5

    positions = {node: (random.random(), random.random()) for node in nodes}
    for node in nodes:
        distances = [(euclidean_distance(positions[node], positions[other_node]), \
        other_node) for other_node in nodes if other_node != node]
        distances.sort()
        for i in range(num_closest_nodes):
            closest_node = distances[i][1]
            G.add_edge(node, closest_node)
    return G

def draw_tree(filename,df=None,col=None,width=500,**kwargs):

    tre = toytree.tree(filename)
    if df is not None:
        cmap = ({c:random_hex_color() for c in df[col].unique()})
        df['color'] = df[col].apply(lambda x: cmap[x])
        idx=tre.get_tip_labels()
        df=df.loc[idx]
        tip_colors = list(df.color)
        node_sizes=[0 if i else 6 for i in tre.get_node_values(None, 1, 0)]
        node_colors = [cmap[df.loc[n][col]] if n in df.index else 'black' for n in tre.get_node_values('name', True, True)]
    else:
        tip_colors = None
        node_colors = None
        node_sizes = None

    canvas,axes,mark = tre.draw(scalebar=True,edge_widths=.5,height=600,width=width,
                                tip_labels_colors=tip_colors,node_colors=node_colors,node_sizes=node_sizes,**kwargs)
    return canvas

def plot_grid(model,ax,colorby='loc_type', ns='herd_size', cmap='Blues', title='', **kwargs):
    """Custom draw method for model graph network"""

    from matplotlib.colors import ListedColormap, LinearSegmentedColormap
    graph = model.G

    if colorby == 'loc_type':
        states = [f.loc_type for f in model.grid.get_all_cell_contents()]
        colormap = {'farm':'#73A8E6','sett':'#E67385'}
        node_colors = [colormap[i] for i in states]
        cmap=None
    elif colorby == 'herd_class':
        states = [f.herd_class for f in model.grid.get_all_cell_contents()]
        colormap = {'beef':'lightgreen','dairy':'yellow','beef suckler':'orange',
                    'fattening':'khaki',None:'gray'}
        node_colors = [colormap[i] for i in states]
        cmap=None
    elif colorby == 'strain':
        states = [n.main_strain() for n in model.grid.get_all_cell_contents()]
        colormap = strain_map
    elif colorby == 'herd_size':
        node_colors = [len(n) for n in model.grid.get_all_cell_contents()]
    elif colorby == 'num_infected':
        node_colors = [len(n.get_infected()) for n in model.grid.get_all_cell_contents()]
    elif colorby == 'perc_infected':
        node_colors = [len(n.get_infected())/len(n)*200 for n in model.grid.get_all_cell_contents()]
    if ns == 'herd_size':
        sizes = [len(n)*20 for n in model.grid.get_all_cell_contents()]
    elif ns == 'num_infected':
        sizes = [len(n.get_infected())*20 for n in model.grid.get_all_cell_contents()]
    elif ns == 'perc_infected':
        sizes = [len(n.get_infected())/len(n)*200 for n in model.grid.get_all_cell_contents()]
    else:
        sizes=50

    ec = ['red' if n.loc_type=='sett' else 'black' for n in model.grid.get_all_cell_contents()]
    pos = nx.kamada_kawai_layout(graph)
    nx.draw(graph, pos, width=.1, node_color=node_colors,node_size=sizes, cmap=cmap,
            edgecolors=ec,linewidths=0.6,alpha=0.8,
            font_size=8,ax=ax, **kwargs)
    #ax.legend()
    ax.set_title(title,fontsize=20)
    #plt.colorbar(ax)
    return

def plot_inf_data(model):
    #fig,ax=plt.subplots(2,2,figsize=(12,4))
    df=model.get_infected_data()
    cols = ['inf_start','inf_time','moves']
    axs=df[cols].hist(grid=False,ec='black',bins=20)
    #df.species.value_counts().plot(ax=axs.flat[3])
    return axs.flat[0].get_figure()

def plot_herds_data(model):
    df = model.get_herds_data()
    cols = ['size','infected']
    axs=df[cols].hist(grid=False,ec='black',bins=20)
    return axs.flat[3].get_figure()

def plot_by_species(model):

    df=model.get_animal_data()
    x=pd.pivot_table(df,index='species',columns=['state'],values='id',aggfunc='count')
    #print (x)
    ax=x.plot(kind='bar')
    return ax.get_figure()
