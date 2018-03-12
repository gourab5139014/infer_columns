import numpy as np
import pandas as pd
import scipy.stats as st
import random
from collections import Counter
from math import log

class analyzer(): # Contains configuration information common to all analyzers
    
    DISTRIBUTIONS = [st.uniform, st.norm, st.zipf]

    def __init__(self):
        print("Analyzer Created")
        self.results_df = pd.DataFrame()
        self.results = [] # List of tuples (<DatasetId>,<AttributeId>,<ComparedDistribution>,<Divergence>)

    def attach_dataset(self, dt:pd.DataFrame):
        self.dataset = dt
        print("Dataset configured")
        
class kl_divergence_analyzer(analyzer):
    def __init__(self):
        print("KL Analyzer Created")
    
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
        cf1, l1 = Counter(p), len(p)
        cf2, l2 = Counter(q), len(q)
        
        print("Initial Lengths cf1 {0} , cf2 {1}".format(len(cf1),len(cf2)))
        # print(cf1.keys())
        # print(cf2.keys())
        # Pre-processing for using KL Divergence of Frequency Counters cf1 and cf2
        s = set(cf1.keys())
        s = s.intersection(cf2.keys()) # Collecting all unique elements in cf1 and cf2

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
        print("Normalized Lengths cf1 {0} , cf2 {1}".format(len(cf1),len(cf2)))
        print("Sum CF1 {0}".format(np.sum(list(cf1.values()))))
        print("Sum CF2 {0}".format(np.sum(list(cf2.values()))))
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

    def score_with_kl_divergence(self):
        try:
            if not self.dataset.empty: #Do the attribute scoring here
                print("Starting comparison by KL")
                # for col in self.dataset:
                #     # print(col)
                #     d = pd.Series(self.dataset[col])
                #     print(d.name)
                #     print(self._score_with_normal(d))
                #     print(self._score_with_uniform(d))
                d = pd.Series(self.dataset['Year'])
                print(d.name)
                n_score = self._score_with_normal(d.copy())
                print("Score with Normal Distribution = {0}\n".format(n_score))
                u_score = self._score_with_uniform(d.copy())
                print("Score with Uniform Distribution = {0}".format(u_score))
            else:
                raise ValueError('Non empty Dataset should be attached before starting comparison')
        except AttributeError as ae:
            print("{0}. Non empty Dataset should be attached before starting comparison".format(ae))