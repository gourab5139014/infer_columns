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
    THETA = 0.2 # Threshold percentage of unique values in a number sequence beyond which it would be tagged Categorical
    # DISTRIBUTIONS = [st.uniform, st.norm, st.zipf]
    DISTRIBUTIONS = [st.uniform, st.norm]

    def __init__(self):
        lg.debug("Analyzer Created")
        self.datasets = {} # (Dataset_tag) -> Dataset
        # self.results = [] # List of tuples (<DatasetId>,<AttributeId>,<ComparedDistribution>,<Divergence>)
        # self.results = pd.DataFrame([], columns=('dataset_id', 'column_name', 'distribution', 'goodness_value'))
    
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
    
    # @DeprecationWarning
    # def save_observation(self, goodness_value, best_dst, column_name, dataset_id):
    #     lg.debug('Adding {0} to results'.format((dataset_id, column_name, best_dst, goodness_value)))
    #     self.results.append((dataset_id, column_name, best_dst, goodness_value))
    
    def export_results_to_csv(self, df, filename_prefix):
        # lg.debug('I am exporting {0}'.format(df))
        filename = filename_prefix + datetime.datetime.now().strftime("_%Y%m%d_%H%M") + ".out"
        # df.to_csv(path_or_buf=filename, float_format='%.6f', index=False)
        df.to_csv(path_or_buf=filename, index=False)
        lg.info('Output written to {0}'.format(filename))
    
    def run(self):
        lg.debug("Datasets \n{0}".format(self.datasets))
        results = pd.DataFrame(columns=('dataset_id', 'column_name', 'distribution', 'goodness_value'))
        for d in self.datasets:
            lg.info('Starting analyze dataset {0}'.format(d))
            df = self.datasets[d]
            for c in df:
                data = pd.Series(df[c])
                sample = data[0]
                lg.debug(sample)
                try:
                    int(sample)
                    lg.debug("{0} is a number".format(sample))
                    duplicate_proportion = ( len(data)-len(data.unique()) ) / len(data)
                    if duplicate_proportion > self.THETA:
                        r = self._apply_categorical_analyses(data, d) # TODO Need to explore more here
                        results = results.append(r)
                    else:
                        r = self._apply_numerical_analyses(data, d)
                        results = results.append(r)
                    # lg.debug('Current collection of results is {0}'.format(results))
                except ValueError as ve:
                    lg.debug("{0} is NOT a number".format(sample))
            self.export_results_to_csv(results, "Output") 

    def _apply_categorical_analyses(self, s, name):
        return pd.DataFrame([(name, s.name, 'Categorical', 1)], columns=('dataset_id', 'column_name', 'distribution', 'goodness_value'))
    
    def _apply_numerical_analyses(self, s, name):
        lla = log_likelihood_analyzer()
        r = lla.score_for_series(s, name)
        # lg.debug('Returning from _apply_numerical_analyses {0}'.format(r))
        return r
                    

class categorical_analyzer(): #TODO Maybe we dont need these anymore
    def __init__(self):
        lg.debug("Categorical Analyzer created")

class numerical_analyzer(analyzer): #TODO Maybe we dont need these anymore
    def __init__(self):
        lg.debug("Numerical Analyzer created")
        analyzer.__init__(self)
        
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
        numerical_analyzer.__init__(self)
    
    def score_for_series(self, data:pd.Series, dataset_id, bins=200):
        observations = []
        
        try:
            lg.debug("Starting scoring {0} by log_likelihood".format(data.name))
            # Best holders initialization
            best_distribution = st.norm
            best_params = (0.0, 1.0)
            best_sse = np.inf

            for distribution in self.DISTRIBUTIONS: 
                lg.info("Modelling {0} with {1}".format(data.name, distribution.name))

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

                # self.save_observation(sse, distribution.name, data.name, dataset_id)
                observations.append((dataset_id, data.name, distribution.name, sse ))
                # identify if this distribution is better
                # if best_sse > sse > 0:
                #     best_distribution = distribution
                #     best_params = params
                #     best_sse = sse
                # self.save_observation(best_sse, best_distribution.name, col, self.dataset_tag)
        # else:
        #     raise ValueError('Non empty Dataset should be attached before starting comparison')
        except AttributeError as ae:
            lg.critical("{0}. {1}".format(ae, traceback.format_exc()))
        observations_df = pd.DataFrame(observations, columns=('dataset_id', 'column_name', 'distribution', 'goodness_value')) # Consistent with analyzer.results schema
        return observations_df

    @DeprecationWarning
    def _score_with_log_likelihood(self, bins=200):
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