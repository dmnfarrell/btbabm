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

class State(enum.IntEnum):
    SUSCEPTIBLE = 0
    INFECTED = 1
    REMOVED = 2

class Strain:
    def __init__(self, name, sequence):
        self.name = name
        self.sequence = sequence
        #mutation rate per nucl per year
        self.mutation_rate = 5e-2

    def mutate(self, p=None, n=1):
        """Mutate the sequence in place"""

        mr = self.mutation_rate
        mutseq = list(self.sequence)
        if p == None:
            p = np.random.choice([0,1], p=[1-mr,mr])
        if p == 1:
            i=random.randint(0,len(mutseq)-1)
            mutseq[i] = random.choice(['A','T','G','C'])

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
        return [a.strain.name for a in self.animals if a.strain!=None]

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
        return

     def __repr__(self):
        return 'sett:%s (%s badgers)' %(self.unique_id,len(self.animals))

class Animal(object):
    """Animal agent that moves from farm to farm """

    def __init__(self, unique_id, model):
        self.unique_id = unique_id
        self.location = None
        self.state = State.SUSCEPTIBLE
        self.strain = None
        self.infection_start = 0
        self.age = 1
        self.death_age = abs(int(random.normalvariate(8*365,1000)))
        self.model = model
        self.farm = None
        return

    def infect(self, strain):
        """Infect the animal with some probability"""

        if random.uniform(0, 1) < self.model.infection_prob:
            self.state = State.INFECTED
            #strain may mutate upon transmission
            self.strain = strain.copy()
            strain.mutate()
            self.infection_start = self.model.schedule.time
        return

class Badger(Animal):
    """Animal agent that acts as wildlife reservoir"""

    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        self.species = 'badger'
        self.time_last_move = 0
        self.time_to_death = abs(int(random.normalvariate(model.mean_inf_time,100)))
        self.moves = []

    def step(self):

        self.age +=1
        self.move()
        curr = self.location
        sett = self.model.get_node(curr)
        if self.state == State.SUSCEPTIBLE:
            for animal in sett.get_infected():
                if animal.state == State.INFECTED:
                    self.infect(animal.strain)

        self.infection_status()
        return

    def move(self):
        """Move to a neighbouring farm"""

        t = self.model.schedule.time
        if t-self.time_last_move<10:
            return
        curr = self.location
        current_sett = self.model.get_node(curr)
        G = self.model.G
        neighbors = G.neighbors(curr)
        nn = list(neighbors)
        if len(nn) > 0:
            new = random.choice(nn)
            loc = self.model.get_node(new)
            #if self.model.callback != None:
            #    self.model.callback('BADGER MOVE')
            #    self.model.callback([self,current_sett, loc, loc.animals])
            if self.state == State.INFECTED:
                if type(loc) is Farm:
                    for animal in loc.animals:
                        if animal.state == State.SUSCEPTIBLE:
                            animal.infect(self.strain)
            self.time_last_move = t
        return

    def infection_status(self):
        """Check infection status and remove animal if dead."""

        if self.state == State.INFECTED or self.age>self.death_age:
            t = self.model.schedule.time-self.infection_start
            if t >= self.time_to_death:
                curr = self.location
                current_sett = self.model.get_node(curr)
                self.state = State.REMOVED
                self.model.schedule.remove(self)
                current_sett.animals.remove(self)
                self.model.deaths +=1
                self.model.add_badger(sett=current_sett)

    def __repr__(self):
        return 'badger:%s (sett %s)' %(self.unique_id,self.location)

class Cow(Animal):
    """Animal agent that moves from farm to farm """

    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        self.species = 'cow'
        #stay time at current farm, updated when moved
        self.stay_time = abs(int(random.normalvariate(model.mean_stay_time,100)))
        #time after infection to death
        self.time_to_death = abs(int(random.normalvariate(model.mean_inf_time,100)))
        #time spent at current farm
        self.time_at_farm = 0
        self.moves = []
        return

    def step(self):
        """
        Step in simulation
        Animal can move, be infected, die or nothing
        """

        self.time_at_farm += 1
        #print (self.model.schedule.time, self.time_at_farm,self.stay_time)
        if self.time_at_farm >= self.stay_time:
            self.move()
        curr = self.location
        #self.model.callback(model.grid.get_all_cell_contents())
        farm = self.model.get_node(curr)

        #check if infected animals on farm
        if self.state == State.SUSCEPTIBLE:
            for animal in farm.get_infected():
                if animal.state == State.INFECTED:
                    self.infect(animal.strain)
        self.infection_status()
        return

    def move(self):
        """Move to another farm"""

        curr = self.location
        current_farm = self.model.get_node(curr)
        G = self.model.G
        neighbors = G.neighbors(curr)
        nn = list(neighbors)

        if len(nn) > 0:
            new = random.choice(nn)
            next_farm = self.model.get_node(new)
            if not type(next_farm) is Farm:
                return

        #if self.model.callback != None:
        #    self.model.callback('MOVE')
        #    self.model.callback([self,current_farm, next_farm, current_farm.animals])
        current_farm.animals.remove(self)
        next_farm.animals.append(self)
        self.location = new
        self.time_at_farm = 0
        self.stay_time = abs(int(random.normalvariate(150,100)))
        current_farm.moves += 1
        t = self.model.schedule.time
        self.moves.append([curr, new, t])
        return

    def infection_status(self):
        """Check infection status and remove animal if dead."""

        if self.state == State.INFECTED:
            t = self.model.schedule.time-self.infection_start
            if t >= self.time_to_death:
                curr = self.location
                current_farm = self.model.get_node(curr)
                self.state = State.REMOVED
                self.model.schedule.remove(self)
                current_farm.animals.remove(self)
                self.model.deaths +=1

                #replace the animal in the population
                if len(current_farm.animals)<2:
                    self.model.add_animal(farm=current_farm)
                else:
                    new_farm = random.choice(self.model.get_farms())
                    #print (current_farm,new_farm,model.farms[new_farm.unique_id])
                    self.model.add_animal(farm=new_farm)
        return

    def __repr__(self):
        return 'cow:%s (herd %s)' %(self.unique_id,self.location)


class FarmPathogenModel(Model):
    def __init__(self, F, C, S, mean_stay_time=100, mean_inf_time=60, cctrans=0.01, infected_start=5, seq_length=100,
                 graph_type='custom', graph_seed=None, callback=None):

        self.num_farms = F
        self.num_setts = S
        self.num_cows = C
        self.num_badgers = S*5

        self.mean_stay_time = mean_stay_time
        self.mean_inf_time = mean_inf_time
        self.infection_prob = cctrans
        self.max_animals_per_farm = 500
        self.max_farm_size = 100

        self.base_sequence = utils.random_sequence(seq_length)
        self.start_strains = {}
        for s in strain_names:
            strain = self.start_strains[s] = Strain(s, self.base_sequence)
            strain.mutate(p=1)
            #print (s,strain.sequence)
        self.schedule = RandomActivation(self)
        self.year = 1
        self.deaths = 0
        self.agents_added = 0
        total = self.num_farms + self.num_setts

        self.callback = callback

        if graph_type == 'erdos_renyi':
            self.G = nx.erdos_renyi_graph(n=total, p=0.2, seed=graph_seed)
        elif graph_type == 'barabasi_albert':
            self.G = nx.barabasi_albert_graph(n=total, m=3, seed=graph_seed)
        elif graph_type == 'watts_strogatz':
            self.G = nx.watts_strogatz_graph(n=total, k=4, p=0.1, seed=graph_seed)
        elif graph_type == 'random_geometric':
            self.G = nx.random_geometric_graph(total, 0.125, seed=graph_seed)
        elif 'custom':
            self.G = utils.create_closest_n_graph(total,3)
        self.grid = NetworkGrid(self.G)

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
        infectedcount=0
        for i in range(self.num_cows):
            animal = self.add_animal(i)
            if infectedcount <= infected_start:
                animal.state = State.INFECTED
                s = random.choice(strain_names)
                animal.strain = self.start_strains[s]
            infectedcount+=1

        #add badgers randomly
        l = self.agents_added+1
        for i in range(l,l+self.num_badgers):
            animal = self.add_badger(i)
            if random.choice(range(5)) == 1:
                animal.state = State.INFECTED
                s = random.choice(strain_names)
                animal.strain = self.start_strains[s]

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

    def get_node(self, node):
        """Get object inside a node by id"""

        return self.G.nodes[node]['agent'][0]

    '''def get_node(self, node):
        x=model.grid.get_cell_list_contents([node])
        if len(x)>0:
            return x[0]'''

    def get_farms(self):
        x=self.grid.get_all_cell_contents()
        return [i for i in x if type(i) is Farm]

    def get_setts(self):
        x=self.grid.get_all_cell_contents()
        return [i for i in x if type(i) is Sett]

    '''def get_farms(self):
        """Get farms in graph"""

        f = [self.get_node(n) for n in self.G.nodes]
        return [i for i in f if type(i) == Farm]

    def get_setts(self):

        f = [self.get_node(n) for n in self.G.nodes]
        return [i for i in f if type(i) == Sett]'''

    def add_animal(self, uid=None, farm=None):
        """Add cow. If no farm given, add randomly"""

        if uid==None:
            uid = self.agents_added+1

        animal = Cow(uid, model=self)
        if farm == None:
            farm = random.choice(self.get_farms())
        #print (farm)
        animal.location = farm.unique_id
        farm.animals.append(animal)
        self.schedule.add(animal)
        self.agents_added += 1
        return animal

    def add_badger(self, uid=None, sett=None):

        if uid==None:
            uid = self.agents_added+1

        animal = Badger(uid, model=self)
        if sett == None:
            sett = random.choice(self.get_setts())
        animal.location = sett.unique_id
        sett.animals.append(animal)
        #print (animal.unique_id)
        self.schedule.add(animal)
        self.agents_added += 1
        return animal

    def get_animals(self):
        x=[]
        for f in self.grid.get_all_cell_contents():
            x.extend(f.animals)
        return x

    def get_herds_data(self):
        """Get summary dataframe of herd info"""

        res=[]
        for f in self.get_farms():
            animals = f.animal_ids()
            res.append([f.unique_id,f.herd_class,len(f.animals),len(f.get_infected()),f.get_strains(),animals,f.moves])
        return pd.DataFrame(res,columns=['id','herd_class','size','infected','strains','animals','moves'])

    def get_random_location(self, k=1):
        return random.sample(self.grid.get_all_cell_contents(), k)

    def get_column_data(self):
        """pivot the model dataframe to get states count at each step"""

        agent_state = self.datacollector.get_agent_vars_dataframe()
        X = pd.pivot_table(agent_state.reset_index(),index='Step',columns='State',aggfunc=np.size,fill_value=0)
        labels = ['Susceptible','Infected','Removed']
        X.columns = labels[:len(X.columns)]
        return X

    def get_infected_data(self):
        """Data for all infected animals"""

        t=self.schedule.time
        x=[]
        for f in self.grid.get_all_cell_contents():
            for a in f.get_infected():
                x.append([a.unique_id,a.species,a.strain.name,a.infection_start,t-a.infection_start,a.location,len(a.moves)])
        df = pd.DataFrame(x,columns=['id','species','strain','inf_start','inf_time','farm','moves'])
        return df

    def get_sequences(self):
        """Get seqs for all circulating strains"""

        seqs = []
        for node in self.grid.get_all_cell_contents():
            for a in node.get_infected():
                s = a.strain
                seqs.append(SeqRecord(Seq(s.sequence),str(a.unique_id)))
        return seqs

    def get_moves(self):
        """Return all moves"""

        res = []
        for farm in self.get_farms():
            for a in farm.animals:
                for m in a.moves:
                    if len(m)>0:
                        res.append([a.unique_id]+m)
        return pd.DataFrame(res,columns=['id','start','end','time'])

    def make_phylogeny(self):
        """Phylogeny of sequences"""

        import snipgenie
        seqs = self.get_sequences()
        infile = 'temp.fasta'
        SeqIO.write(seqs,infile,'fasta')
        try:
            run_fasttree(infile, '.', bootstraps=50)
        except:
            return
        ls = len(seqs[0])
        snipgenie.trees.convert_branch_lengths('tree.newick','tree.newick', ls)
        return seqs

    def get_clades(model, newick=None):
        """Get clades from newick tree"""

        if newick==None:
            seqs = model.make_phylogeny()
            newick = 'tree.newick'
        import snipgenie
        cl = snipgenie.trees.get_clusters(newick)
        return cl

    def __repr__(self):
        return 'model with %s farms, %s setts and %s animals' %(self.num_farms, self.num_setts, len(self.get_animals()))
