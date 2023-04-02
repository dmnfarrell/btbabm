<img align="right" src=logo.png width=150px>

# bTBabm - Agent based model for bovine TB spread in herds.

This is a Python package that simulates Bovine TB spread amongst herds and badgers. It uses the Mesa package for implementing an Agent based model.

## Parameters

Times are given in days since each step is a day.

* mean_stay_time - mean time in a herd
* mean_inf_time - mean infection time
* mean_latency_time - mean latency time when not infectious
* cctrans - cow-cow transmission prob
* bctrans - badger-cow transmission prob
* infected_start - how many cows to infect at start
* mean_inf_time - mean infection time length before death
* mean_stay_time - mean time on farm
* seq_length - sequence length for simulating strains/mutations
* herd_class - type of herd

## Installation

If you want to try this out, the easiest way is to install with pip:

`pip install -e git+https://github.com/dmnfarrell/btbabm.git#egg=btbabm`

## Usage

In Python you can run the model as follows:

```python
from btbabm import models
from btbabm import utils

model = models.FarmPathogenModel(F=30,C=800,S=10,mean_inf_time=20,mean_stay_time=150,
                       cctrans=0.01,seq_length=100,graph_seed=4)
#run 100 steps
for s in range(100):
  model.step()

#equivalent code with progress bar
model.run(100)

#get state data
df = model.get_column_data()
#get data for infected animals
df = model.get_infected_data()
#plot the grid/network
fig,ax=plt.subplots(1,1,figsize=(10,6))
utils.plot_grid(model,with_labels=True,ns='perc_infected',ax=ax)
```

## Dashboard

There is a panel dashboard for experimenting with the model. It can be run by executing the `dashboard.py` module.

<img src=img/dash_scr.png width=600px>

## References

* [Mesa](https://mesa.readthedocs.io/)
* [Individual-based model for the control of Bovine Viral Diarrhea spread in livestock trade networks](https://www.sciencedirect.com/science/article/pii/S0022519321002393?via%3Dihub)
* [A Practical Introduction to Mechanistic Modeling of Disease Transmission in Veterinary Science](https://www.frontiersin.org/articles/10.3389/fvets.2020.546651/full)
