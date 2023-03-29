#!/usr/bin/env python

"""
    ABM Model classes for BTB
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
import enum

from mesa import Agent, Model
from mesa.time import RandomActivation
from mesa.space import NetworkGrid
from mesa.datacollection import DataCollector

from . import utils

'''
Parameters
mean_stay_time - mean time in a herd
mean_inf_time - mean infection time
cctrans - cow-cow transmission prob
infected_start - time step infection starts
seq_length - sequence length
herd_class - type of herd
'''

strain_names = string.ascii_letters[:10].upper()
HERD_CLASSES = ['beef','dairy','beef suckler','fattening']

def load_model(filename):
    """Load model object"""

    import bz2, pickle
    ifile = bz2.BZ2File(filename,'rb')
    obj = pickle.load(ifile)
    ifile.close()
    return obj

def grid_from_spatial_graph(model, G):
    """
    Great the networkgrid from a predefined graph,
    used so we can convert from spatial data.
    """

    grid = NetworkGrid(G)
    locs = nx.get_node_attributes(G, 'loc_type')
    for node in G.nodes():
        #print (node, locs[node])
        nodetype = locs[node]
        if nodetype == 'sett':
            sett = Sett(node, model)
            grid.place_agent(sett, node)
        elif nodetype == 'herd':
            f = Farm(node, model)
            grid.place_agent(f, node)
    return grid

class State(enum.IntEnum):
    SUSCEPTIBLE = 0
    INFECTED = 1
    LATENT = 2
    REMOVED = 3

class Strain:
    def __init__(self, name, sequence):
        self.name = name
        self.sequence = sequence
        #mutation rate per nucl per year
        self.mutation_rate = 5e-2

    def mutate(self, p=0.05, n=1):
        """Mutate the sequence in place.
        Args:
            p: probability of a mutation
            n: number of mutations
        """

        mr = self.mutation_rate
        mutseq = list(self.sequence)
        if random.uniform(0, 1) > p:
            return
        for i in range(n):
            pos = random.randint(0,len(mutseq)-1)
            mutseq[pos] = random.choice(['A','T','G','C'])
        self.sequence = "".join(mutseq)
        return #Strain(self.name, mutseq)

    def copy(self):
        import copy
        return copy.deepcopy(self)

    def __repr__(self):
        return 'strain %s' %self.name

class Location(object):
    def __init__(self, unique_id, model):

        self.unique_id = unique_id
        self.position = None
        self.model = model
        self.animals = []
        self.loc_type = None
        return

    def animal_ids(self):
        return [a.unique_id for a in self.animals]

    def infected(self):
        if len(self.get_infected())>0:
            return True
        return False

    def get_infected(self):
        """Get all infected animals"""

        x=[]
        for a in self.animals:
            if a.state == State.INFECTED:
                x.append(a)
        return x

    def get_strains(self):
        """Strain(s) on farm"""
        return [a.strain.name for a in self.get_infected() if a.strain!=None]

    def main_strain(self):
        """Main strain"""

        x = self.get_strains()
        if len(x)>0:
            return x[0]

    def __len__(self):
        return len(self.animals)

    #def __str__(self):
    #    return json.dumps(dict(self), ensure_ascii=False)

    #def toJSON(self):
    #    return json.dumps(dict(self), ensure_ascii=False)

class Farm(Location):
     def __init__(self, unique_id, model, herd_class=None):
        super().__init__(unique_id, model)
        self.moves = 0
        self.herd_class = random.choice(HERD_CLASSES)
        self.loc_type = 'farm'
        return

     def __repr__(self):
        return 'herd:%s (%s cows)' %(self.unique_id,len(self.animals))

class Sett(Location):
     def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        self.size = int(random.normalvariate(20,10))
        self.loc_type = 'sett'
        self.herd_class = None
        self.moves = []
        return

     def __repr__(self):
        return 'sett:%s (%s badgers)' %(self.unique_id,len(self.animals))

class Animal(object):
    """Animal agent that moves from farm to farm """

    def __init__(self, model):
        self.unique_id = utils.get_short_uid()
        self.location = None
        self.state = State.SUSCEPTIBLE
        self.strain = None
        self.infection_start = 0
        self.age = 1
        self.death_time = None
        self.model = model
        #self.farm = None
        return

    def contact_neighbours(self):
        """
        Contact a neighbouring node. Only farms with nearby generate_land_parcels
        should have connected nodes. Any animal can contact one in the other location.
        """

        t = self.model.schedule.time
        #if t-self.time_last_move<10:
        #    return
        curr = self.location
        G = self.model.G
        neighbors = G.neighbors(curr)
        nn = list(neighbors)
        if len(nn) > 0:
            new = random.choice(nn)
            loc = self.model.get_node(new)
            if self.state == State.INFECTED:
                #if type(loc) is Farm:
                #interact with random
                for animal in loc.animals:
                    if animal.state == State.SUSCEPTIBLE:
                        animal.infect(self.strain, self.model.bctrans)

            elif self.state == State.SUSCEPTIBLE:
                for animal in loc.animals:
                    if animal.state == State.INFECTED:
                        self.infect(animal.strain, self.model.bctrans)

            self.time_last_move = t
        return

    def infect(self, strain, ptrans):
        """Infect the animal with some probability"""

        if random.uniform(0, 1) < ptrans:
            self.state = State.LATENT
            #strain may mutate upon transmission
            self.strain = strain.copy()
            strain.mutate()
            self.infection_start = self.model.schedule.time
            #if self.model.callback != None:
            #    self.model.callback('infected %s time:%s' %(self,self.model.schedule.time))
            return True
        return False

class Badger(Animal):
    """Animal agent that acts as wildlife reservoir"""

    def __init__(self, model):
        super().__init__(model)
        self.species = 'badger'
        self.time_last_move = 0
        self.death_age = abs(int(random.normalvariate(5*365,1000)))
        #self.latency_period = abs(int(random.normalvariate(200,100)))
        self.latency_period = random.randint(1,model.mean_latency_time)
        self.time_to_death = abs(int(random.normalvariate(model.mean_inf_time*2,100)))
        self.moves = []

    def contact(self):
        """Contact animals on same node"""

        curr = self.location
        sett = self.model.get_node(curr)
        if self.state == State.SUSCEPTIBLE:
            for animal in sett.get_infected():
                if animal.state == State.INFECTED:
                    self.infect(animal.strain, self.model.bctrans)
        return

    def step(self):
        """Do step"""

        self.age +=1
        self.contact()
        self.contact_neighbours()
        self.infection_status()
        return

    def infection_status(self):
        """Check infection status and remove animal if dead."""

        if self.state == State.INFECTED or self.age>self.death_age:
            t = self.model.schedule.time-self.infection_start
            if t >= self.time_to_death:
                self.model.remove_animal(self)
                self.death_time = self.model.schedule.time
                current_sett = self.model.get_node(self.location)
                self.model.add_badger(sett=current_sett)

    def __repr__(self):
        return 'badger:%s (sett %s)' %(self.unique_id,self.location)

class Cow(Animal):
    """Animal agent that moves from farm to farm """

    def __init__(self, model):
        super().__init__(model)
        self.species = 'cow'
        #stay time at current farm, updated when moved
        self.stay_time = abs(int(random.normalvariate(model.mean_stay_time,100)))
        #natural age
        self.death_age = abs(int(random.normalvariate(4*365,1000)))
        #time after infection to death
        self.time_to_death = abs(int(random.normalvariate(model.mean_inf_time,100)))
        #latency period when not infectious
        #self.latency_period = abs(int(random.normalvariate(model.mean_latency_time,100)))
        self.latency_period = random.randint(1,model.mean_latency_time)
        #time spent at current farm
        self.time_at_farm = 0
        self.moves = []
        return

    def contact(self):
        """Contact animals on same node"""

        curr = self.location
        farm = self.model.get_node(curr)
        if self.state == State.SUSCEPTIBLE:
            for animal in farm.get_infected():
                if animal.state == State.INFECTED:
                    self.infect(animal.strain, self.model.cctrans)
        return

    def step(self):
        """
        Step in simulation.
        Contact with neighbors or moves to anywhere.
        Animal can be infected, die or nothing.
        """

        self.time_at_farm += 1
        self.age +=1
        moves = self.model.allow_moves
        if self.time_at_farm >= self.stay_time and moves==True:
            self.move()
        self.contact()
        self.contact_neighbours()
        #self.model.callback(model.grid.get_all_cell_contents())
        self.infection_status()
        return

    def move(self):
        """Move to another farm anywhere on network. Can include moves only
        to similar type farms etc."""

        model = self.model
        curr = self.location
        current_farm = model.get_node(curr)
        G = self.model.G

        #get random farm
        next_farm = model.get_random_location()
        if type(next_farm) is Sett:
            return
        #print (next_farm)
        #print ('move')
        new = next_farm.unique_id
        current_farm.animals.remove(self)
        next_farm.animals.append(self)
        self.location = new
        self.time_at_farm = 0
        self.stay_time = abs(int(random.normalvariate(model.mean_stay_time,100)))
        current_farm.moves += 1
        t = model.schedule.time
        self.moves.append([curr, new, t])
        return

    def infection_status(self):
        """Check infection status and remove animal if dead."""

        if self.state == State.SUSCEPTIBLE:
            return
        t = self.model.schedule.time-self.infection_start
        if self.state == State.INFECTED:
            if t >= self.time_to_death or self.age>self.death_age:
                self.model.remove_animal(self)
                self.death_time = self.model.schedule.time
                current_farm = self.model.get_node(self.location)
                #replace the animal in the population
                if len(current_farm.animals)<2:
                    self.model.add_animal(farm=current_farm)
                else:
                    new_farm = random.choice(self.model.get_farms())
                    #print (current_farm,new_farm,model.farms[new_farm.unique_id])
                    self.model.add_animal(farm=new_farm)
        elif self.state == State.LATENT:
            if t >= self.latency_period:
                self.state = State.INFECTED
        return

    def __repr__(self):
        return 'cow:%s (herd %s)' %(self.unique_id,self.location)

class FarmPathogenModel(Model):
    """A BTB pathogen model with herd and badger transmission"""

    def __init__(self, F=100, C=10, S=0, mean_stay_time=300, mean_inf_time=60, mean_latency_time=100,
                 cctrans=0.01, bctrans=0.001,
                 infected_start=5, allow_moves=False,
                 seq_length=100,
                 graph_type='default', graph_seed=None,
                 callback=None):

        self.num_farms = F
        self.num_setts = S
        self.num_cows = C
        self.num_badgers = S*8

        self.mean_stay_time = mean_stay_time
        self.mean_inf_time = mean_inf_time
        self.mean_latency_time = mean_latency_time
        self.cctrans = cctrans
        self.bctrans = bctrans
        self.max_herd_size = 100
        self.max_sett_size = 15

        self.base_sequence = utils.random_sequence(seq_length)
        self.start_strains = {}
        for s in strain_names:
            strain = self.start_strains[s] = Strain(s, self.base_sequence)
            strain.mutate(p=1,n=5)
            #print (s,strain.sequence)
        self.schedule = RandomActivation(self)
        self.year = 1
        self.deaths = 0
        self.agents_added = 0
        self.removed = []
        total = self.num_farms + self.num_setts
        self.allow_moves = allow_moves
        self.callback = callback

        #if graph is not None:
        if graph_type == 'default':
            #creates grid from custom graph
            graph,pos,gdf = utils.herd_sett_graph(F,S,graph_seed)
            self.grid = grid_from_spatial_graph(self, graph)
            self.G = graph
            self.pos = pos
            self.gdf = gdf
        else:
            self.G,pos = utils.create_graph(graph_type, graph_seed, total)
            self.grid = NetworkGrid(self.G)
            self.pos=None
            #add some setts first
            added=[]
            for node in random.sample(list(self.G.nodes()), self.num_setts):
                sett = Sett(node, self)
                self.grid.place_agent(sett, node)
                added.append(node)

            for node in self.G.nodes():
                if node in added: continue
                farm = Farm(node, self)
                self.grid.place_agent(farm, node)

        #add cows randomly
        strains = strain_names

        infectedcount=0
        for i in range(self.num_cows):
            animal = self.add_animal()
            if infectedcount <= infected_start:
                animal.state = State.INFECTED
                #s = strains[infectedcount]
                s = random.choice(strain_names)
                animal.strain = self.start_strains[s]
                infectedcount+=1

        #add badgers randomly
        l = self.agents_added+1
        for i in range(l,l+self.num_badgers):
            animal = self.add_badger()
            if animal == None:
                continue
            if random.choice(range(5)) == 1:
                animal.state = State.INFECTED
                #s = strains[infectedcount]
                s = random.choice(strain_names)
                animal.strain = self.start_strains[s]
                infectedcount+=1

        self.datacollector = DataCollector(
            agent_reporters={"State": "state"})

        return

    def step(self):
        """Step through model"""

        self.schedule.step()
        self.datacollector.collect(self)
        self.year = math.ceil(self.schedule.steps/365)
        #turnover farm herds randomly?

        #self.callback(len(self.get_farms()))
        return

    def run(self, steps):
        """Run n steps with progress bar"""

        from tqdm import tqdm
        with tqdm(total=steps) as pbar:
            for i in range(steps):
                self.step()
                pbar.update(1)
        return

    def get_node(self, node):
        """Get object inside a node by id"""

        return self.G.nodes[node]['agent'][0]

    def get_farms(self):
        """Get farm nodes"""

        x=self.grid.get_all_cell_contents()
        return [i for i in x if type(i) is Farm]

    def get_setts(self):
        """Get setts"""

        x=self.grid.get_all_cell_contents()
        return [i for i in x if type(i) is Sett]

    def add_animal(self, farm=None):
        """Add cow. If no farm given, add randomly"""

        animal = Cow(model=self)
        if farm == None:
            farm = random.choice(self.get_farms())

        animal.location = farm.unique_id
        farm.animals.append(animal)
        self.schedule.add(animal)
        self.agents_added += 1
        return animal

    def add_badger(self, sett=None):
        """Add badger"""

        animal = Badger(model=self)
        if sett == None:
            sett = random.choice(self.get_setts())
        if len(sett.animals)>self.max_sett_size:
            return
        animal.location = sett.unique_id
        sett.animals.append(animal)
        #print (animal.unique_id)
        self.schedule.add(animal)
        self.agents_added += 1
        return animal

    def remove_animal(self, animal):
        """Remove from model"""

        t = self.schedule.time
        loc = self.get_node(animal.location)
        animal.state = State.REMOVED
        animal.death_age = t
        self.schedule.remove(animal)
        loc.animals.remove(animal)
        self.deaths +=1
        self.removed.append(animal)
        return

    def get_animals(self, infected=False):
        """Get all animals into a list"""

        x=[]
        for f in self.grid.get_all_cell_contents():
            if infected == True:
                x.extend(f.get_infected())
            else:
                x.extend(f.animals)
        return x

    def get_herds_data(self):
        """Get summary dataframe of node info"""

        res=[]
        for f in self.grid.get_all_cell_contents():
            animals = f.animal_ids()
            res.append([f.unique_id,f.loc_type,f.herd_class,len(f.animals),len(f.get_infected()),#f.get_strains(),animals
                        f.moves])
        return pd.DataFrame(res,columns=['id','loc_type','herd_class','size','infected','moves'])

    def get_random_node(self):
        n = random.sample(self.G.nodes, 1)[0]
        return self.G.nodes[n]

    def get_random_location(self):
        return random.sample(self.grid.get_all_cell_contents(), 1)[0]

    def get_column_data(self):
        """pivot the model dataframe to get states count at each step"""

        agent_state = self.datacollector.get_agent_vars_dataframe()
        X = pd.pivot_table(agent_state.reset_index(),index='Step',columns='State',aggfunc=np.size,fill_value=0)
        labels = ['Susceptible','Infected','Latent']
        X.columns = labels[:len(X.columns)]
        return X

    def get_animal_data(self, infected=False, removed=False):
        """Data for all animals"""

        t=self.schedule.time
        x=[]
        animals = self.get_animals(infected)
        if removed == True:
            animals.extend(self.removed)
        for a in animals:
            if a.strain!=None:
                s=a.strain.name
            else:
                s=None
            x.append([a.unique_id,a.species,a.state,s,a.infection_start,t-a.infection_start,a.death_age,a.location,len(a.moves)])
        df = pd.DataFrame(x,columns=['id','species','state','strain','inf_start','inf_time','death_age','herd','moves'])
        df['year'] = df.apply(lambda x: math.ceil(x.death_age/365),1)
        return df

    def get_infected_data(self):
        """Data for all infected animals"""

        return self.get_animal_data(infected=True)

    def get_moves(self, removed=False):
        """Return all moves"""

        res = []
        animals = self.get_animals()
        if removed == True:
            animals.extend(self.removed)
        for a in animals:
            for m in a.moves:
                if len(m)>0:
                    res.append([a.unique_id,a.death_time]+m)
        return pd.DataFrame(res,columns=['id','death_time','start','end','time'])

    def get_sequences(self, removed=False, redundant=True):
        """Get strain sequences of infected"""

        seqs = []
        animals = self.get_animals(infected=True)
        if removed == True:
            animals.extend(self.removed)
        for a in animals:
            s = a.strain
            if s != None:
                seqs.append(SeqRecord(Seq(s.sequence),str(a.unique_id)))
        new = []
        if redundant == False:
            seen = set()
            for record in seqs:
                if record.seq not in seen:
                    seen.add(record.seq)
                    new.append(record)
        else:
            new = seqs
        return new

    def make_phylogeny(self, removed=False, redundant=True):
        """Phylogeny of sequences"""

        import snipgenie
        seqs = self.get_sequences(removed, redundant)
        infile = 'temp.fasta'
        SeqIO.write(seqs,infile,'fasta')
        try:
            utils.run_fasttree(infile, '.', bootstraps=50)
        except Exception as e:
            print ('fasttree error')
            print(e)
            return
        ls = len(seqs[0])
        snipgenie.trees.convert_branch_lengths('tree.newick','tree.newick', ls)
        return seqs

    def get_clades(self, newick=None, removed=False, redundant=True):
        """Get clades from newick tree"""

        if newick == None:
            seqs = self.make_phylogeny(removed, redundant)
            newick = 'tree.newick'
        import snipgenie
        cl = snipgenie.trees.get_clusters(newick).astype(str)
        return cl

    def get_geodataframe(self,removed=False):
        """Get geodataframe of all animal locations, assumes we are using a graph
        with positions.
        """

        animals = self.get_animal_data(removed=removed)
        self.gdf = self.gdf.rename(columns={'ID':'herd'})
        gdf = self.gdf.merge(animals,on='herd',how='inner')
        return gdf

    def plot(self, ax=None):
        """Shorthand for using utils.plot_grid"""

        if ax == None:
            f,ax = plt.subplots(1,1)
        utils.plot_grid(self,ax=ax,pos=self.pos,colorby='strain',ns='num_infected')
        return

    def plot_bokeh(self, title=''):
        p = utils.plot_grid_bokeh(self, title=title)
        return p

    def save(self, filename):
        """Save model to file"""

        import pickle
        import bz2
        ofile = bz2.BZ2File(filename,'wb')
        pickle.dump(self,ofile)
        return

    def __repr__(self):
        l1 = 'model with %s farms, %s setts and %s animals\n' %(self.num_farms, self.num_setts, len(self.get_animals()))
        l2 = 'CC rate:%s BC rate:%s' % (self.cctrans,self.bctrans)
        return l1+l2
