#!/usr/bin/env python3

"""
    ABM Model panel dashboard
    Created Jan 2023
    Copyright (C) Damien Farrell
"""

import sys,os,time
import pylab as plt
import panel as pn
import panel.widgets as pnw
pn.extension('tabulator', css_files=[pn.io.resources.CSS_URLS['font-awesome']])

from btbabm.models import FarmPathogenModel
from btbabm import utils

def dashboard():

    def run_model(farms, animals, setts, cctrans, staytime, inftime, steps, delay, graph_type, graph_seed, refresh):

        def callback(x):
            str_pane.value += str(x)+'\n'

        model = FarmPathogenModel(farms, animals, setts, staytime, inftime, cctrans, 5, 200,
                              graph_type, graph_seed=None,
                              callback=callback)
        callback(model)
        fig1,ax1 = plt.subplots(1,1,figsize=(15,10))
        grid_pane.object = fig1
        fig2,ax2 = plt.subplots(1,1,figsize=(10,6))
        plot_pane1.object = fig2
        fig3,ax3 = plt.subplots(1,1,figsize=(10,6))
        plot_pane2.object = fig3
        progress.max=steps
        progress.value = 0

        showsteps = list(range(1,steps+1,refresh))
        #step through the model and plot at each step
        for i in range(1,steps+1):
            #callback(model)
            #callback(model.get_farms())
            model.step()
            plt.close()
            if i in showsteps:
                ax1.clear()
                ns=nodesize_input.value
                y=model.year
                mov=len(model.get_moves())
                deaths=model.deaths
                total = len(model.get_animals())
                col = colorby_input.value
                text='day=%s year=%s moves=%s deaths=%s animals=%s' %(i,y,mov,deaths,total)
                utils.plot_grid(model,ax=ax1,
                          title=text, colorby=col, cmap=cmap_input.value,
                          ns=ns, with_labels=labels_input.value)
                grid_pane.param.trigger('object')
                ax2.clear()
                #s = model.circulating_strains()
                d=model.get_infected_data()
                df_pane.value = d
                hd = model.get_herds_data()
                df2_pane.value = hd
                fig2 = utils.plot_inf_data(model)
                plot_pane1.object = fig2
                plot_pane1.param.trigger('object')
                df = model.get_column_data()
                ax3.clear()
                df.plot(ax=ax3)
                ax3.set_xlim(0,steps)
                plot_pane2.param.trigger('object')
                html=html_tree(model)
                tree_pane.object = html
                out = model.G.nodes

            progress.value += 1
            time.sleep(delay)
        plt.clf()

    def html_tree(model):
        result = model.make_phylogeny()
        if result==None:
            return '<p>no tree</p>'
        cl = model.get_clades('tree.newick')
        idf = model.get_infected_data()
        x=idf.merge(cl,left_on='id',right_on='SequenceName')
        x=x.set_index('SequenceName')
        x.index = x.index.map(str)
        tre=toytree.tree('tree.newick')
        col='strain'
        canvas = draw_tree('tree.newick',x,col,tip_labels=False,width=500)
        toyplot.html.render(canvas, "temp.html")
        with open('temp.html', 'r') as f:
            html = f.read()
        return html

    def set_stop(event):
        global stop
        stop = True
        print ('STOP')

    graph_types = ['watts_strogatz','erdos_renyi','barabasi_albert','random_geometric','custom']
    farm_types = ['mixed','beef','dairy','suckler']
    cmaps = ['Blues','Reds','Greens','RdBu','coolwarm','summer','winter','icefire','hot','viridis']
    grid_pane = pn.pane.Matplotlib(plt.Figure(),tight=True,width=900,height=620)
    plot_pane1 = pn.pane.Matplotlib(plt.Figure(),height=300)
    plot_pane2 = pn.pane.Matplotlib(plt.Figure(),height=300)
    tree_pane = pn.pane.HTML()
    str_pane = pnw.TextAreaInput(disabled=True,height=600,width=400)
    df_pane = pnw.Tabulator(show_index=False,height=600)
    df2_pane = pnw.Tabulator(show_index=False,height=600)

    w=120
    colorby = ['loc_type','herd_class','herd_size','num_infected']
    go_btn = pnw.Button(name='run',width=w,button_type='success')
    stop_btn = pnw.Button(name='stop',width=w,button_type='danger')
    farms_input = pnw.IntSlider(name='farms',value=20,start=5,end=1000,step=1,width=w)
    animals_input = pnw.IntSlider(name='cows',value=400,start=10,end=5000,step=10,width=w)
    setts_input = pnw.IntSlider(name='setts',value=5,start=1,end=100,step=1,width=w)
    farmtypes_input = pnw.Select(name='farm types',options=farm_types,width=w)
    cctrans_input = pnw.FloatSlider(name='CC trans',value=0.01,step=.001,start=0,end=1,width=w)
    bctrans_input = pnw.FloatSlider(name='BC trans',value=0.01,step=.001,start=0,end=1,width=w)
    staytime_input = pnw.FloatSlider(name='mean stay time',value=100,step=1,start=5,end=1000,width=w)
    inftime_input = pnw.FloatSlider(name='mean inf. time',value=60,step=1,start=5,end=600,width=w)
    steps_input = pnw.IntSlider(name='steps',value=10,start=1,end=2000,width=w)
    refresh_input = pnw.IntSlider(name='refresh rate',value=1,start=1,end=100,width=w)
    delay_input = pnw.FloatSlider(name='step delay',value=0,start=0,end=3,step=.2,width=w)
    graph_input = pnw.Select(name='graph type',options=graph_types,width=w)
    graph_seed_input = pnw.IntInput(name='graph seed',value=10,width=w)
    #seed_input = pnw.Select(name='graph seed',options=['random'],width=w)
    colorby_input = pnw.Select(name='color by',options=colorby,width=w)
    cmap_input = pnw.Select(name='colormap',options=cmaps,width=w)
    nodesize_input = pnw.Select(name='node size',options=colorby[2:],width=w)
    labels_input = pnw.Checkbox(name='node labels',value=False,width=w)
    progress = pn.indicators.Progress(name='Progress', value=0, width=400, bar_color='primary')

    widgets = pn.Column(pn.Tabs(('model',pn.WidgetBox(go_btn,farms_input,animals_input,setts_input,farmtypes_input,cctrans_input,bctrans_input,staytime_input,inftime_input,
                    steps_input,refresh_input,delay_input)),
                                ('options',pn.WidgetBox(graph_input,graph_seed_input,colorby_input,cmap_input,nodesize_input,labels_input))), width=w+30)
    #widgets = pn.Column(go_btn)

    def execute(event):
        #run the model with widget
        run_model(farms_input.value, animals_input.value, setts_input.value, cctrans_input.value, staytime_input.value, inftime_input.value,
                  steps_input.value, delay_input.value,graph_input.value, graph_seed_input.value, refresh_input.value)

    go_btn.param.watch(execute, 'clicks')

    app = pn.Row(pn.Column(widgets),pn.Column(progress,grid_pane,sizing_mode='stretch_width'),
                 pn.Tabs(('plots',pn.Column(plot_pane1,plot_pane2)), ('tree',tree_pane), ('inf_data',df_pane), ('herd_data',df2_pane), ('debug',str_pane)),
                 sizing_mode='stretch_both',background='WhiteSmoke')

    return app

bootstrap = pn.template.BootstrapTemplate(title='BTB Farm Spread ABM Simulation')
            #favicon='static/logo.png',logo='static/logo.png',header_color='blue')
pn.config.sizing_mode = 'stretch_width'
app = dashboard()
bootstrap.main.append(app)
bootstrap.servable()

if __name__ == '__main__':
    pn.serve(bootstrap, port=5010)
