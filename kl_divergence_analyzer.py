import numpy as np
import pandas as pd
from pandas.api.types import is_string_dtype, is_numeric_dtype, is_integer_dtype
import scipy.stats as st
import random
from collections import Counter
from math import log
import csv
import datetime
import traceback
import logging, logging.config


logging.config.fileConfig('logging.conf')
lg = logging.getLogger("analyzer")

DISTRIBUTION_NAMES = ["norm","uniform"] # Numerical data types that are inferred

class kl_divergence_analyzer():
    def __init__(self):
        lg.debug("KL Analyzer Created")
    
    @DeprecationWarning
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

    def _score_with_kl_divergence(self, dataset_id):
        try:
            if not self.dataset.empty: #Do the attribute scoring here
                lg.debug("Starting comparison by KL")
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

    def kl_div(self, p: pd.Series, distri_p, q:pd.Series, distri_q):
        t1 = pd.to_numeric(p, errors='coerce')
        t2 = pd.to_numeric(q, errors='coerce')
        t1 = t1[~t1.isnull()]
        t2 = t2[~t2.isnull()]
        
        mu1, sigma1 = np.mean(t1), np.std(t1)
        lower1, upper1 = np.min(t1), np.max(t1)
        if distri_p == DISTRIBUTION_NAMES[0]: # Normal
            d1 = st.norm(mu1, sigma1)
        else: # Uniform
            d1 =st.uniform(loc = lower1, scale = upper1)
        mu2, sigma2 = np.mean(t2), np.std(t2)
        lower2, upper2 = np.min(t2), np.max(t2)
        if distri_q == DISTRIBUTION_NAMES[0]: # Normal
            d2 = st.norm(mu2, sigma2)
        else: # Uniform
            d2 =st.uniform(loc = lower2, scale = upper2)
        
        # domain to evaluate PDF on
        X = np.linspace(min(lower1, lower2), max(upper1, upper2), max(len(t1), len(t2)))
        e = st.entropy(d1.pdf(X), d2.pdf(X))
        return e

    def normalized_kl_div(self, p: pd.Series, distri_p, q:pd.Series, distri_q):
        k = self.kl_div(p, distri_p, q, distri_q)
        return (1 - np.exp(-k))
