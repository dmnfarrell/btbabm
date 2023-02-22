# btbabm

## Agent based model for bovine TB spread in herds.

This is a Python package that simulates Bovine TB spread amongst herds and badgers. It uses the Mesa package for implementing an Agent based model.

## Parameters

Times are given in days since each step is a day.

* mean_stay_time - mean time in a herd
* mean_inf_time - mean infection time
* cctrans - cow-cow transmission prob
* bctrans - badger-cow transmission prob
* infected_start - how many cows to infect at start
* mean_inf_time - mean infection time length before death
* mean_stay_time - mean time on farm
* seq_length - sequence length for simulating strains/mutations
* herd_class - type of herd

## Dashboard

There is a panel dashboard for experimenting with model. It can be run by executing the `dashboard.py` file.

<img align="left" src=img/dash_scr.png width=450px>

## Refs

* [Mesa](https://mesa.readthedocs.io/)
* [Individual-based model for the control of Bovine Viral Diarrhea spread in livestock trade networks](https://www.sciencedirect.com/science/article/pii/S0022519321002393?via%3Dihub)
* [A Practical Introduction to Mechanistic Modeling of Disease Transmission in Veterinary Science](https://www.frontiersin.org/articles/10.3389/fvets.2020.546651/full)
