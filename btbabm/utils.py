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
from Bio import SeqIO, AlignIO, Phylo
import json
import toytree
import geopandas as gpd
from shapely.geometry import Point,MultiPoint,MultiPolygon

def random_color(seed=None):
    """random rgb color"""

    if seed != None:
        random.seed = seed
    return tuple([np.random.random_sample() for i in range(3)])

strain_names = string.ascii_letters[:10].upper()
#strain_cmap = ({c:random_color(seed=1) for c in strain_names})
strain_cmap = {'A':'coral','B':'dodgerblue','C':'lightgreen','D':'mediumpurple',
                'E':'orange','F':'pink','G':'gold','H':'cyan','I':'beige','J':'red',
                'K':'brown'}
strain_cmap[None] = 'gray'

def get_short_uid(n=10):
    """Short uids up to n chars long"""

    import hashlib
    import base64
    # Generate a unique hash value using the current time and random bytes
    hash_value = hashlib.sha256(f"{time.time()}:{os.urandom(16)}".encode()).digest()
    # Encode the hash value using base64 without padding characters
    encoded_value = base64.b64encode(hash_value, altchars=b"-_").rstrip(b"=")
    # Convert the bytes to a string
    uid = encoded_value.decode("ascii")[:n]
    return uid

def random_sequence(length=50):
    seq=''
    for count in range(length):
        seq += random.choice("CGTA")
    return seq

def get_nonredundant_alignment(aln):
    """Informative positions from aln"""

    keepcols = []
    for i in range(aln.get_alignment_length()):
        column = aln[:, i]
        if len(set(column)) > 1:
            keepcols.append(i)
    new_records = []
    for record in aln:
        new_seq = ''.join([record.seq[i] for i in keepcols])
        new_records.append(SeqRecord(Seq(new_seq), id=record.id, description=record.description))

    # Create a new MultipleSeqAlignment object from the new list of SeqRecord objects
    new = AlignIO.MultipleSeqAlignment(new_records)
    return new

def random_hex_color():
    """random hex color"""

    r = lambda: np.random.randint(0,255)
    c='#%02X%02X%02X' % (r(),r(),r())
    return c

def random_hex_colors(n=1,seed=None):
    if seed != None:
        np.random.seed(seed)
    return [random_hex_color() for i in range(n)]

def hex_colors(n=1, cmap='viridis'):

    from matplotlib.colors import ListedColormap
    # Create a colormap
    cmap = plt.cm.get_cmap(cmap)
    # Generate a list of n hex colors
    colors = cmap.colors[:n]
    colors = ListedColormap(colors).colors
    return colors

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

def closest_n_graph(n, num_closest_nodes):

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

def herd_sett_graph(farms=20,setts=5, seed=None):
    """Custom herd/sett graph with spatial positions"""

    n=farms+setts
    gdf = random_herds_setts(n,ratio=setts/n, seed=seed)
    G,pos = delaunay_pysal(gdf, 'ID', attrs=['loc_type'])
    #G,pos = geodataframe_to_graph(gdf, 'ID', attrs=['loc_type'])
    #add more edges for herds
    #new_edges = add_random_edges(G,4)
    #G.add_edges_from(new_edges)
    #sparsify local edges
    remove_random_edges(G)
    return G,pos,gdf

def random_points(n, bounds=(10,10,1000,1000), seed=None):
    """Random points"""

    np.random.seed(seed)
    points = []
    minx, miny, maxx, maxy = bounds
    x = np.random.uniform( minx, maxx, n)
    y = np.random.uniform( miny, maxy, n)
    return x, y

def random_geodataframe(n, bounds=(10,10,1000,1000), seed=None):
    """Random geodataframe of points"""

    x,y = random_points(n, bounds, seed)
    df = pd.DataFrame()
    df['geometry'] = list(zip(x,y))
    df['geometry'] = df['geometry'].apply(Point)
    gdf = gpd.GeoDataFrame(df, geometry='geometry')
    gdf['ID'] = range(n)
    return gdf

def jitter_points(r, scale=2e-2):
    """Jitter GeoDataFrame points"""

    a=np.random.normal(0,scale)
    b=np.random.normal(0,scale)
    if (r.geometry.is_empty): return Point()
    x,y = r.geometry.x+a,r.geometry.y+b
    return Point(x,y)

def random_herds_setts(n, ratio=0.2, seed=None):
    gdf = random_geodataframe(n, seed=seed)
    gdf['loc_type'] = np.random.choice(['herd','sett'], n, p=[1-ratio,ratio])
    return gdf

def geodataframe_to_graph(gdf, key=None, attrs=[], d=200):
    """Convert geodataframe to graph with edges at distance threshold"""

    from scipy.spatial import distance_matrix

    cent = gdf.geometry.values
    coords = [(i.x,i.y) for i in cent]
    distances = distance_matrix(coords,coords)

    # Create an empty graph
    G = nx.Graph()
    for i in range(len(gdf)):
        G.add_node(i, pos=cent[i])
    # Loop through all pairs of centroids
    for i in range(len(gdf)):
        for j in range(i+1, len(gdf)):
            if distances[i][j] <= d:
                G.add_edge(i, j, weight=distances[i][j])

    pos = dict(zip(G.nodes, coords))
    nx.set_node_attributes(G, pos, 'pos')
    #rename nodes
    if key != None:
        mapping = dict(zip(G.nodes,gdf[key]))
        #print (mapping)
        G = nx.relabel_nodes(G, mapping)
    for col in attrs:
        vals = dict(zip(G.nodes, gdf[col]))
        nx.set_node_attributes(G, vals, col)
    # Assign edge weights as distances
    for u, v in G.edges():
        d = ((pos[u][0] - pos[v][0])**2 + (pos[u][1] - pos[v][1])**2)**0.5
        G.add_edge(u, v, weight=round(d,1))

    return G,pos

def graph_to_geodataframe(G):
    """Convert graph positions to gdf"""

    points = []
    types = []
    for node in G.nodes():
        pos = G.nodes[node]['pos']
        point = Point(pos)
        points.append(point)
        types.append(G.nodes[node]['loc_type'])

    # create a GeoDataFrame from the list of points
    gdf = gpd.GeoDataFrame({'ID': list(G.nodes()), 'geometry': points, 'loc_type': types})
    return gdf

def delaunay_pysal(gdf, key=None, attrs=[]):
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
    if key != None:
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
    return G, pos

def get_community(G,pos):
    """Get network community using Louvain method"""

    from networkx.algorithms import community
    comm=community.louvain_communities(G, resolution=.5, seed=5)
    set_node_community(G, comm)
    clrs = ({c:random_color() for c in range(len(comm)+1)})
    node_color = [clrs[G.nodes[v]['community']] for v in G.nodes]
    fig,ax=plt.subplots(figsize=(12,10))
    nx.draw(G, pos, node_size=400, node_color=node_color, with_labels=True, font_size=8, ax=ax)
    return G

def remove_random_edges(G, prob=0.02):
    """Remove random edges"""

    for i in range(10):
        for edge in G.edges():
            if random.random() < prob:
                G.remove_edge(*edge)
    return

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

def get_largest_poly(x):
    if type(x) is MultiPolygon:
        return max(x.geoms, key=lambda a: a.area)
    else:
        return x

def count_fragments(x):
    if type(x) is MultiPolygon:
        return len(x.geoms)
    else:
        return 1

def generate_land_parcels(cells=100,herds=10,empty=0, fragments=0,
                        fragmented_farms=1, seed=None,
                        bounds=None, crs=None):
    """
    Simulate land parcels with fragmentation.
    Args:
        cells: number of points to make initial cell polyons
        herds: number of farms
        empty: fraction of fragments that are empty
        fragments: number of empty cells to add back as fragments
        fragmented_farms: no. of fragmented farms
        bounds: bounds of points [x1,y1,x2,y2]
        crs: crs of parcels if needed
    """

    from shapely.geometry import Point,MultiPoint,MultiPolygon,Polygon
    from libpysal import weights, examples
    from libpysal.cg import voronoi_frames
    from sklearn.cluster import KMeans

    n = cells
    k = herds
    if seed != None:
        np.random.seed(seed)

    if bounds == None:
        bounds = [1,1,1000,1000]
    x1,y1,x2,y2 = bounds
    x,y = np.random.randint(x1,x2,n),np.random.randint(y1,y2,n)
    coords = np.column_stack((x, y))
    cells, generators = voronoi_frames(coords, clip="extent")
    centroids = cells.geometry.centroid
    #cluster parcels into 'herds'
    kmeans = KMeans(n_clusters=k,n_init='auto').fit(coords)
    cells['cluster'] = kmeans.labels_.astype('object')

    #remove some proportion of cells randomly
    e = cells.sample(frac=empty, random_state=seed)
    cells.loc[e.index,'cluster'] = 'empty'

    #create new GeoDataFrame
    poly=[]
    data = {'cluster':[]}
    for c,g in cells.groupby('cluster'):
        if c == 'empty':
            continue
        poly.append(MultiPolygon(list(g.geometry)))
        data['cluster'].append(c)
    farms = gpd.GeoDataFrame(data=data,geometry=poly)

    #merge contiguous fragments of same herd
    farms = farms.dissolve(by='cluster').reset_index()
    #remove polygons with 'holes'
    def no_holes(x):
        if type(x) is MultiPolygon:
            return MultiPolygon(Polygon(p.exterior) for p in x.geoms)
        else:
            return Polygon(x.exterior)
    farms.geometry = farms.geometry.apply(no_holes)
    #print (farms)

    #remove empty cells inside parcels
    e = e[e.within(farms.unary_union)==False]

    #assign some of the empty cells as fragments
    #indexes = farms.sample(fragmented_farms).index
    for index in farms.sample(fragmented_farms).index:
        for i,r in cells.loc[e.index].sample(fragments).iterrows():
            #print (index)
            poly = farms.iloc[index].geometry
            if type(poly) is MultiPolygon:
                geom = poly.geoms
                new = MultiPolygon(list(geom) + [r.geometry])
            else:
                geom = poly
                new = MultiPolygon([geom,r.geometry])
            farms.loc[index,'geometry'] = new

    #merge contiguous fragments again in case we added fragments
    farms = farms.dissolve(by='cluster').reset_index()

    #farms['herd'] = farms.apply(lambda x: get_short_uid(8),1)
    farms['herd'] = farms.cluster
    farms['fragments'] = farms.geometry.apply(count_fragments)
    farms['color'] = random_hex_colors(len(farms),seed=seed)
    farms['loc_type'] = 'herd'
    if crs != None:
        fards = farms.set_crs(crs)
    return farms

def pashuffle(data, perc=.1, seed=None):
    """Partial shuffle list"""

    #random.seed(seed)
    for index, letter in enumerate(data):
        if random.randrange(0, 100) < perc*100:
            new_index = random.randrange(0, len(data))
            data[index], data[new_index] = data[new_index], data[index]
    return data

def contiguous_parcels(parcels):
    """Get all contiguous parcels. Returns dict of indexes"""

    res = {}
    for i,r in parcels.iterrows():
        #print (r.herd)
        polygon = r.geometry
        spatial_index = parcels.sindex
        possible_matches_index = list(spatial_index.intersection(polygon.bounds))
        possible_matches = parcels.iloc[possible_matches_index]
        x = possible_matches[possible_matches.intersects(polygon)]
        #if len(x)>1:
        res[i] = list(x.index)
    return res

def land_parcels_to_graph(farms,dist=100,attrs=['loc_type','herd'], **kwargs):
    """
    Create simulated land parcels and associated contact network.
    Args:
        farms: land parcels, geodataframe of multipolygons
        dist: max distance at which to connect two nodes
        attrs: attributes in parcels gdf to add to graph nodes
        see utils.generate_land_parcels
    Returns:
        centroids - geodataframe
        graph - networkx graph
        pos - positions for graph
    """

    #add setts
    #setts = random_geodataframe(setts)
    #setts['loc_type']='sett'
    #setts['color'] = 'blue'

    #get centroids of parcels
    farms = farms.reset_index(drop=True)
    larg = farms.geometry.apply(get_largest_poly)
    cent = gpd.GeoDataFrame(data=farms.copy(),geometry=larg.geometry.centroid)#.reset_index(drop=True)
    cent['loc_type'] = 'herd'

    #make network graph
    G,pos = geodataframe_to_graph(cent, d=dist, attrs=attrs)

    print ('%s herds in network' %len(G.nodes()))
    #add egdes where nodes have contiguous parcels
    cont = contiguous_parcels(farms)
    for n in list(G.nodes):
        if n not in cont:
            continue
        for j in cont[n]:
            if j!=n and not G.has_edge(j,n):
                G.add_edge(n,j)
    #add more edges for setts if near any other parcels
    #print (len(G.nodes()))
    return G, pos, cent

def plot_grid(model,ax,pos=None,colorby='loc_type', ns='herd_size',
                cmap='Blues', title='', **kwargs):
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
        sizes = [len(n.get_infected())*10+1 for n in model.grid.get_all_cell_contents()]
    elif ns == 'perc_infected':
        sizes = [len(n.get_infected())/len(n)*200 for n in model.grid.get_all_cell_contents()]
    else:
        sizes=50

    ec = ['red' if n.loc_type=='sett' else 'black' for n in model.grid.get_all_cell_contents()]
    if pos == None:
        pos = model.pos

    nx.draw(graph, pos, width=.1, node_color=node_colors,node_size=sizes, #cmap=cmap,
            edgecolors=ec,linewidths=0.8,alpha=0.8,
            font_size=8,ax=ax, **kwargs)
    #ax.legend()
    ax.set_title(title,fontsize=20)
    #plt.colorbar(ax)
    plt.tight_layout()
    return

def plot_grid_bokeh(model,title='', ns='num_infected', parcels=None):
    """Plot netword for model with bokeh"""

    from bokeh.plotting import show,figure, from_networkx
    from bokeh.models import (BoxZoomTool, Circle, HoverTool, PanTool, Label, LabelSet,
                              MultiLine, Plot, Range1d, ResetTool)
    from bokeh.models.sources import ColumnDataSource, GeoJSONDataSource

    G = model.G.copy()
    for n in G.nodes:
        G.nodes[n]['agent'] = 'X'

    cmap = strain_cmap
    states = [f.main_strain() for f in model.grid.get_all_cell_contents()]
    node_colors = [cmap[n] for n in states]
    #data for herds/setts
    data = model.get_herds_data()
    #print (data)

    if ns == 'num_infected':
        data['ns'] = data['infected']+1
    elif ns == 'herd_size':
        data['ns'] = data['size']+1

    def calc_layout(G: nx.Graph, scale=None, center=None):
        return {i:list(model.pos[i]) for i in model.pos}

    if model.pos == None:
        pos = nx.spring_layout
    else:
        pos = calc_layout

    plot = Plot(sizing_mode='stretch_width', plot_height=600, title=title)

    graph_renderer = from_networkx(G, pos, scale=1, center=(0,0))
    if data is not None:
        source = ColumnDataSource(data=data)
        graph_renderer.node_renderer.data_source = source

    # Customize the node appearance
    if node_colors is None:
        node_colors = [None] * len(G.nodes)

    graph_renderer.node_renderer.data_source.data['fill_color'] = node_colors
    graph_renderer.node_renderer.glyph = Circle(size=8, fill_color='fill_color', fill_alpha=0.6)
    graph_renderer.node_renderer.glyph.size = 'ns'
    # Customize the edge appearance
    graph_renderer.edge_renderer.glyph = MultiLine(line_color="#CCCCCC", line_alpha=0.8, line_width=.5)
    # Add the graph to the plot
    plot.renderers.append(graph_renderer)
    node_hover_tool = HoverTool(tooltips=[("id", "@id"), ("loc_type", "@loc_type"),
                                          ("size", "@size"), ("infected", "@infected")])

    if parcels is not None:
        geosource = GeoJSONDataSource(geojson = parcels.to_json())
        plot.patches('xs','ys', source=geosource, fill_alpha=.4, line_width=0.2, fill_color='gray', line_color='black')

    plot.add_tools(node_hover_tool, BoxZoomTool(), PanTool(), ResetTool())
    plot.toolbar.logo = None
    return plot

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

def plot_parcels_graph(parcels, cent, G, pos, labels=True):
    """Show parcels and associated graph"""

    fig,ax=plt.subplots(1,2,figsize=(18,8))
    parcels.loc[0,'color'] = 'red'
    parcels.plot(color=parcels.color,ec='.3',alpha=0.6,ax=ax[0])
    for idx, row in cent.iterrows():
        ax[0].text(row.geometry.centroid.x, row.geometry.centroid.y,
                row['herd'], ha='center', fontsize=8)

    ax[0].axis('off')
    nx.draw(G,pos,node_size=100,width=0.5,ax=ax[1],node_color=cent.color,with_labels=True,alpha=0.6,font_size=8)
    weights = nx.get_edge_attributes(G, 'weight')
    #w=nx.draw_networkx_edge_labels(G, pos, weights, font_size=8, ax=axs[1])
    return fig

def draw_tree(filename,df=None,col=None,cmap=None,node_sizes=0,width=500,height=500,**kwargs):
    """Draw newick tree with toytree"""

    tre = toytree.tree(filename)
    if df is not None:
        labels = df[col].unique()
        if cmap == None:
            cmap = ({c:random_hex_color() if c in labels else 'black' for c in labels})

        df['color'] = df[col].apply(lambda x: cmap[x])
        idx=tre.get_tip_labels()
        df=df.loc[idx]
        tip_colors = list(df.color)
        node_sizes=[0 if i else 6 for i in tre.get_node_values(None, 1, 0)]
        node_colors = [cmap[df.loc[n][col]] if n in df.index else 'black' for n in tre.get_node_values('name', True, True)]
    else:
        tip_colors = None
        node_colors = None

    canvas,axes,mark = tre.draw(scalebar=True,edge_widths=.5,height=height,width=width,
                                tip_labels_colors=tip_colors,node_colors=node_colors,
                                node_sizes=node_sizes,**kwargs)
    return canvas

def run_fasttree(infile, outfile='tree.newick', bootstraps=100):
    """Run fasttree"""

    fc = 'fasttree'
    cmd = '{fc} -nt {i} > {o}'.format(fc=fc,b=bootstraps,i=infile,o=outfile)
    tmp = subprocess.check_output(cmd, shell=True, stderr=subprocess.STDOUT)
    return

def convert_branch_lengths(treefile, outfile, snps):
    """Re-scale branch lengths to no. of snps"""

    tree = Phylo.read(treefile, "newick")
    for parent in tree.find_clades(terminal=False, order="level"):
            for child in parent.clades:
                if child.branch_length:
                    child.branch_length *= snps
    Phylo.write(tree, outfile, "newick")
    return

def diffseqs(seq1,seq2):
    """Diff two sequences"""

    c=0
    for i in range(len(seq1)):
        if seq1[i] != seq2[i]:
            c+=1
    return c

def snp_dist_matrix(aln):
    """Get pairwise snps distances from biopython
       Args:
        aln:
            Biopython multiple sequence alignment object.
        returns:
            a matrix as pandas dataframe
    """

    names=[s.id for s in aln]
    m = []
    for i in aln:
        x=[]
        for j in aln:
            d = diffseqs(i,j)
            x.append(d)
        #print (x)
        m.append(x)
    m = pd.DataFrame(m,index=names,columns=names)
    return m
