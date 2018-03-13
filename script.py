# import warnings
import numpy as np
import pandas as pd
import scipy.stats as st
# import statsmodels as sm
import glob
from analyzers import analyzer
import logging as lg, sys, traceback
# import matplotlib
# import matplotlib.pyplot as plt
lg.basicConfig(stream=sys.stderr, level=lg.INFO)

def get_uniform_distributed_integers(min_value, max_value, total, fliped):
    fake1 = np.linspace(min_value, max_value, total, dtype=int)
    replace_at = np.random.randint(min_value, max_value, size=fliped)
    for i in range(0, len(fake1)):
        if(i in replace_at):
            fake1[i] = np.random.randint(min_value, max_value)
    return fake1

def study_dataset_from_file(d):
    dt = pd.read_csv(d)
    # print(pd.Series(dt["Year"]))
    # TODO Call kl_analyzer here
    # k = kl_divergence_analyzer()
    # k.attach_dataset(dt)
    # k.score_with_kl_divergence()
    
    # ll = log_likelihood_analyzer()
    # ll.attach_dataset(dt, d)
    # ll.score_with_log_likelihood()
    # ll.export_results_to_csv(d)

if __name__ == "__main__":
    
    # for filename in glob.glob('data/*.csv'):
    #     print(filename)
    #     with open(filename) as f:
    #         study_dataset_from_file(f)
    datasets = ['total_waterborne_commerce.csv']
    # datasets = []
    try:
        a = analyzer()
        for d in datasets:
            a.add_dataset(d)
        a.run()
    except KeyError as ke:
        lg.critical('Need a unique dataset_id')
        # print("S**t happened")
    # except:
    #     traceback.print_stack()