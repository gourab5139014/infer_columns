# import warnings
import numpy as np
import pandas as pd
import scipy.stats as st
# import statsmodels as sm
import glob
from analyzers import analyzer
import logging, logging.config
import sys, traceback
# import matplotlib
# import matplotlib.pyplot as plt


logging.config.fileConfig('logging.conf')
lg = logging.getLogger("loki")

def get_uniform_distributed_integers(min_value, max_value, total, fliped):
    fake1 = np.linspace(min_value, max_value, total, dtype=int)
    replace_at = np.random.randint(min_value, max_value, size=fliped)
    for i in range(0, len(fake1)):
        if(i in replace_at):
            fake1[i] = np.random.randint(min_value, max_value)
    return fake1

if __name__ == "__main__":    
    a = analyzer()
    datasets = []
    for filename in glob.glob('data/*.csv'):
        print(filename)
        a.add_dataset(filename)
    a.run()

    # datasets = [ './data/accident_fatalities_table_cleaned.csv','./data/Ag_On_Slopes.csv','./data/Ag_P_Balance.csv','./data/agw_demand_20160428.csv','./data/AMAD11_20160429.csv','./data/AvgPrecip_NHDPv2_WBD.csv','./data/BigGameHunting_RecreationDemand.csv','./data/biodiversity_SE_NHDPv2_WBD.csv', './data/biodiversity_SW_NHDPv2_WBD.csv',  './data/biomass_NHDPv2_WBD.csv', './data/Bird_National_Metrics_20160429.csv']
    # datasets = [ './data/Bird_National_Metrics_20160429.csv']
    # datasets = [ './data/accident_fatalities_table_cleaned.csv']
    
    # try:
    #     a = analyzer()
    #     for d in datasets:
    #         a.add_dataset(d)
    #     a.run()
    # except KeyError as ke:
    #     lg.error('Need a unique dataset_id')    
   