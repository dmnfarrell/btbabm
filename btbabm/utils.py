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
import geopandas as gpd
from shapely.geometry import Point,MultiPoint,MultiPolygon

strain_names = string.ascii_letters[:10].upper()
strain_cmap = ({c:random_color(seed=1) for c in strain_names})
strain_cmap[None] = 'gray'

def get_short_uid():
    import hashlib
    import base64
    # Generate a unique hash value using the current time and random bytes
    hash_value = hashlib.sha256(f"{time.time()}:{os.urandom(16)}".encode()).digest()
    # Encode the hash value using base64 without padding characters
    encoded_value = base64.b64encode(hash_value, altchars=b"-_").rstrip(b"=")
    # Convert the bytes to a string
    uid = encoded_value.decode("ascii")[:6]
    return uid

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

def create_graph(graph_type, graph_seed, size=10):
    """Predefined graphs"""

    pos=None
    if graph_type == 'erdos_renyi':
        G = nx.erdos_renyi_graph(n=size, p=0.2, seed=graph_seed)
    elif graph_type == 'barabasi_albert':
        G = nx.barabasi_albert_graph(n=size, m=3, seed=graph_seed)
    elif graph_type == 'watts_strogatz':
        G = nx.watts_strogatz_graph(n=size, k=4, p=0.1, seed=graph_seed)
    elif graph_type == 'powerlaw_cluster':
        G = nx.powerlaw_cluster_graph(n=size, m=3, p=0.5, seed=graph_seed)
    elif graph_type == 'random_geometric':
        G = nx.random_geometric_graph(n=size, p=0.2, seed=graph_seed)
    return G, pos

def create_herd_sett_graph(farms=20,setts=5):
    """Custom herd/sett graph with spatial positions"""

    n=farms+setts
    gdf = random_geodataframe(n,ratio=setts/n)
    G,pos = delaunay_pysal(gdf, 'ID', attrs=['loc_type'])
    #add more edges for herds
    new_edges = add_random_edges(G,4)
    G.add_edges_from(new_edges)
    return G,pos

def random_points(n):
    """Random points"""

    points = []
    bounds = [10,10,100,100]
    minx, miny, maxx, maxy = bounds
    x = np.random.uniform( minx, maxx, n)
    y = np.random.uniform( miny, maxy, n)
    return x, y

def random_geodataframe(n, ratio=0.2):
    """Random geodataframe"""

    x,y = random_points(n)
    df = pd.DataFrame()
    df['points'] = list(zip(x,y))
    df['points'] = df['points'].apply(Point)
    gdf = gpd.GeoDataFrame(df, geometry='points')
    gdf['ID'] = range(n)
    gdf['loc_type'] = np.random.choice(['herd','sett'], n, p=[1-ratio,ratio])
    return gdf

def gdf_to_distgraph(gdf):
    """Convert geodataframe to graph"""

    from libpysal import weights, examples
    coordinates = np.column_stack((gdf.geometry.x, gdf.geometry.y))
    dist = weights.DistanceBand.from_array(coordinates, threshold=50000)
    knn3 = weights.KNN.from_dataframe(gdf, k=3)
    G = knn3.to_networkx()
    pos = dict(zip(G.nodes, coordinates))
    return G,pos

def delaunay_pysal(gdf, key='SeqID', attrs=[]):
    """Get delaunay graph from gdf of points using libpysal"""

    from libpysal import weights, examples
    from libpysal.cg import voronoi_frames

    coordinates = np.column_stack((gdf.geometry.x, gdf.geometry.y))
    distances = gdf.geometry.apply(lambda x: gdf.distance(x))
    #print (distances)
    cells, generators = voronoi_frames(coordinates, clip="extent")
    delaunay = weights.Rook.from_dataframe(cells)
    G = delaunay.to_networkx()
    #rename nodes
    mapping = dict(zip(G.nodes,gdf[key]))
    #print (mapping)
    G = nx.relabel_nodes(G, mapping)
    pos = dict(zip(G.nodes, coordinates))
    nx.set_node_attributes(G, pos, 'pos')
    #print (positions)
    for col in attrs:
        vals = dict(zip(G.nodes, gdf[col]))
        nx.set_node_attributes(G, vals, col)
    #add lengths
    lengths={}
    for edge in G.edges():
        a,b = edge
        dist = int(math.sqrt(sum([(a - b) ** 2 for a, b in zip(pos[a],pos[b])])))
        lengths[edge] = dist
    nx.set_edge_attributes(G, lengths, 'length')

    #add names to nodes - not needed?
    for i, node in enumerate(G.nodes()):
        G.nodes[node]['label'] = gdf.iloc[i][key]

    return G, pos

def add_random_edges(G, new_connections=1):
    """
      Add new connections to random other nodes, up to new_connections
      https://stackoverflow.com/questions/42591549/add-and-delete-a-random-edge-in-networkx
    """
    new_edges = []

    loctypes = nx.get_node_attributes(G,'loc_type')
    for node in G.nodes():
        if loctypes[node] == 'sett':
            continue
        # find the other nodes this one is connected to
        connected = [to for (fr, to) in G.edges(node)]
        # and find the remainder of nodes, which are candidates for new edges
        unconnected = [n for n in G.nodes() if not n in connected]

        # probabilistically add a random edge
        if len(unconnected): # only try if new edge is possible
            #if random.random() < p_new_connection:
            for new in random.sample(unconnected,new_connections):
                #new = random.choice(unconnected)
                if new == node or loctypes[new] == 'sett':
                    continue
                G.add_edge(node, new)
                #print ("\tnew edge:\t {} -- {}".format(node, new))
                new_edges.append( (node, new) )
                # book-keeping, in case both add and remove done in same cycle
                unconnected.remove(new)
                connected.append(new)
    return new_edges

def plot_grid(model,ax,pos=None,colorby='loc_type', ns='herd_size', cmap='Blues', title='', **kwargs):
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
        node_colors = [strain_cmap[n] for n in states]
    elif colorby == 'herd_size':
        node_colors = [len(n) for n in model.grid.get_all_cell_contents()]
    elif colorby == 'num_infected':
        node_colors = [len(n.get_infected()) for n in model.grid.get_all_cell_contents()]
    elif colorby == 'perc_infected':
        node_colors = [len(n.get_infected())/len(n)*200 for n in model.grid.get_all_cell_contents()]
    if ns == 'herd_size':
        sizes = [len(n)*10 for n in model.grid.get_all_cell_contents()]
    elif ns == 'num_infected':
        sizes = [len(n.get_infected())*10 for n in model.grid.get_all_cell_contents()]
    elif ns == 'perc_infected':
        sizes = [len(n.get_infected())/len(n)*200 for n in model.grid.get_all_cell_contents()]
    else:
        sizes=50

    ec = ['red' if n.loc_type=='sett' else 'black' for n in model.grid.get_all_cell_contents()]
    if pos == None:
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
