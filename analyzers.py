import numpy as np
import pandas as pd
import scipy.stats as st
import random
from collections import Counter
from math import log
import csv
import datetime
import traceback
import logging as lg

class analyzer(): # Contains configuration information common to all analyzers
    
    # DISTRIBUTIONS = [st.uniform, st.norm, st.zipf]
    DISTRIBUTIONS = [st.uniform, st.norm]

    def __init__(self):
        lg.debug("Analyzer Created")
        self.datasets = {} # (Dataset_tag) -> Dataset
        self.results = [] # List of tuples (<DatasetId>,<AttributeId>,<ComparedDistribution>,<Divergence>)

    def add_dataframe(self, dt:pd.DataFrame, dataset_id):        
        if dataset_id in self.datasets:
            raise KeyError('{0} is already present in Datasets')
        else:
            self.datasets[dataset_id] = dt
            lg.debug("Dataset {0} added".format(dataset_id))
    
    def add_dataset(self, dt):
        df = pd.read_csv(dt)
        if dt in self.datasets:
            raise KeyError('{0} is already present in Datasets')
        else:
            self.add_dataframe(df, dt)
    
    def save_observation(self, goodness_value, best_dst, column_name, dataset_id):
        self.results.append((self.dataset_tag, column_name, best_dst, goodness_value))
    
    def export_results_to_csv(self, filename_prefix):
        filename = filename_prefix + datetime.date.today().strftime("%Y%m%d") + ".out"
        with open(filename,'w') as ofile:
            csv_out=csv.writer(ofile)
            # csv_out.writerow(['name','num'])
            for row in self.results:
                csv_out.writerow(row)
    
    def run(self):
        for d in self.datasets:
            lg.debug('Starting analyze dataset {0}'.format(d))
            df = self.datasets[d]
            

class categorical_analyzer(analyzer):
    def __init__(self):
        lg.debug("Categorical Analyzer created")

class numerical_analyzer(analyzer):
    def __init__(self):
        lg.debug("Numerical Analyzer created")
        
class kl_divergence_analyzer(analyzer):
    def __init__(self):
        lg.debug("KL Analyzer Created")
    
    def _kl_div_scipy(self, p,q):
        p = np.asarray(p, dtype=np.float64)
        q = np.asarray(q, dtype=np.float64)
        # return np.sum(np.where(p != 0, p * np.log(p / q), 0))
        return st.entropy(p,q)

    def _my_kl_divergence(self, p,q):
        """ Returns Kl Divergence of two integer lists. Theory at https://www.countbayesie.com/blog/2017/5/9/kullback-leibler-divergence-explained
        :type p: List[int]
        :type q: List[int]
        :rtype: double
        """
        cf1 = Counter(p)
        cf2 = Counter(q)
        
        lg.debug("Initial Lengths cf1 {0} , cf2 {1}".format(len(cf1),len(cf2)))
        lg.debug(cf1.keys())
        lg.debug(cf2.keys())
        # Pre-processing for using KL Divergence of Frequency Counters cf1 and cf2
        s = set(cf1.keys())
        s = s.intersection(cf2.keys()) # Collecting all unique elements in cf1 and cf2

        # REmoving elements which are not in intersection of CF1 and CF2
        for e in list(cf1):
            if e not in s:
                cf1.pop(e, None)

        for e in list(cf2):
            if e not in s:
                cf2.pop(e, None)
        
        l1, l2 = len(cf1), len(cf2)
        # Normalizing the series to reflect probabilities of occurence
        for e in list(cf1): # Since we can't iterate over a mutable collection undergoing change
            if e in s:
                cf1[e] = float(cf1[e]/l1)
            else:
                cf1.pop(e, None)
        for f in list(cf2):
            if f in s:
                cf2[f] = float(cf2[f]/l2)
            else:
                cf2.pop(f, None)
        lg.debug("Normalized Lengths cf1 {0} , cf2 {1}".format(len(cf1),len(cf2)))
        lg.debug("Sum CF1 {0}".format(np.sum(list(cf1.values()))))
        lg.debug("Sum CF2 {0}".format(np.sum(list(cf2.values()))))
        # print(cf1.keys())
        # print(cf1.values())
        # print(cf2.keys())
        # print(cf2.values())
         
        lib_val = self._kl_div_scipy(list(cf1.values()),list(cf2.values()))
        return lib_val
    
    def _score_with_normal(self, dst):
        lower, upper = np.min(dst), np.max(dst)
        mu, sigma = np.mean(dst), np.std(dst)
        # s = np.random.truncnorm(mu, sigma, len(dst)) # TODO How many points to sample ?
        s = st.truncnorm(a = (lower - mu) / sigma, b = (upper - mu) / sigma, loc=mu, scale=sigma).rvs(len(dst))
        s = s.round().astype(int)
        return self._my_kl_divergence(dst, s)
    
    def _score_with_uniform(self, dst):
        lower, upper = np.min(dst), np.max(dst)
        u = np.random.uniform(lower, upper, len(dst)) # TODO How many points to sample ?
        u = u.round().astype(int)
        return self._my_kl_divergence(dst, u)

    def score_with_kl_divergence(self, dataset_id):
        try:
            if not self.dataset.empty: #Do the attribute scoring here
                lg.debug("Starting comparison by KL")
                # for col in self.dataset:
                #     d = pd.Series(self.dataset[col])
                #     print(d.name)
                #     n_score = self._score_with_normal(d.copy())
                #     print("Score with Normal Distribution = {0}\n".format(n_score))
                #     u_score = self._score_with_uniform(d.copy())
                #     print("Score with Uniform Distribution = {0}".format(u_score))
                d = pd.Series(self.dataset['Total'])
                lg.debug(d.name)
                n_score = self._score_with_normal(d.copy())
                lg.info("Score with Normal Distribution = {0}\n".format(n_score))
                u_score = self._score_with_uniform(d.copy())
                lg.info("Score with Uniform Distribution = {0}".format(u_score))
            else:
                raise ValueError('Non empty Dataset should be attached before starting comparison')
        except AttributeError as ae:
            lg.critical("{0}. Non empty Dataset should be attached before starting comparison".format(ae))

class log_likelihood_analyzer(numerical_analyzer):
    def __init__(self):
        lg.debug("log_likelihood Analyzer Created")
        analyzer.__init__(self)
    
    def score_with_log_likelihood(self, bins=200):
        try:
            if not self.dataset.empty: #Do the attribute scoring here
                lg.debug("Starting comparison by log_likelihood")
                for col in self.dataset:
                    # Best holders
                    best_distribution = st.norm
                    best_params = (0.0, 1.0)
                    best_sse = np.inf

                    for distribution in self.DISTRIBUTIONS: 
                        data = pd.Series(self.dataset[col])
                        lg.debug("Modelling {0} with {1}".format(data.name, distribution.name))

                        ## CODE FOR DISTRIBUTION FITTING
                        # Get histogram of original data
                        y, x = np.histogram(data, bins=bins, density=True)
                        # print("x = {0}".format(x))
                        x = (x + np.roll(x, -1))[:-1] / 2.0

                        # fit dist to data
                        params = distribution.fit(data) #TODO Study return parameters for different distributions

                        # Separate parts of parameters
                        arg = params[:-2]
                        loc = params[-2]
                        scale = params[-1]

                        # Calculate fitted PDF and error with fit in distribution
                        pdf = distribution.pdf(x, loc=loc, scale=scale, *arg)
                        sse = np.sum(np.power(y - pdf, 2.0))

                        # identify if this distribution is better
                        if best_sse > sse > 0:
                            best_distribution = distribution
                            best_params = params
                            best_sse = sse
                    self.save_observation(best_sse, best_distribution.name, col, self.dataset_tag)
                # d = pd.Series(self.dataset['Total'])
                # print(d.name)
                # n_score = self._score_with_normal(d.copy())
                # print("Score with Normal Distribution = {0}\n".format(n_score))
                # u_score = self._score_with_uniform(d.copy())
                # print("Score with Uniform Distribution = {0}".format(u_score))
            else:
                raise ValueError('Non empty Dataset should be attached before starting comparison')
        except AttributeError as ae:
            lg.critical("{0}. Non empty Dataset should be attached before starting comparison".format(ae))
        #     tb = traceback.format_exc()
        # else:
        #     tb = ""
        # finally:
        #     print(tb)