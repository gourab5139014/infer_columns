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

REPORTING_SCHEMA = ('dataset_id', 'column_name', 'distribution', 'rms_normed', 'y_mean')

class analyzer(): # Contains configuration information common to all analyzers
    THETA_THRESHOLD = 0.5 # Threshold percentage of unique values in a number sequence beyond which it would be tagged Categorical
    DATATYPES = ["Numerical","Categorical", "Other"] # Data types that are inferred
    # DISTRIBUTIONS = [st.uniform, st.norm, st.zipf]
    DISTRIBUTIONS = [st.uniform, st.norm]

    def __init__(self):
        lg.debug("Analyzer Created")
        self.datasets = {} # (Dataset_tag) -> Dataset
    
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
    
    def export_results_to_csv(self, df, filename_prefix):
        # lg.debug('I am exporting {0}'.format(df))
        filename = filename_prefix + datetime.datetime.now().strftime("_%Y%m%d_%H%M") + ".out"
        # df.to_csv(path_or_buf=filename, float_format='%.6f', index=False)
        df.to_csv(path_or_buf=filename, index=False)
        lg.info('Output written to {0}'.format(filename))
    
    def infer_data_type(self, s:pd.Series):
        inferred_data_type = self.DATATYPES[-1]
        if is_numeric_dtype(s):
            lg.debug("{0} seems to be numeric".format(s.name))
            if is_integer_dtype(s):
                lg.debug("{0} unique values among {1} total values".format(s.nunique(), len(s)))
                unique_proportion = s.nunique()/len(s)
                if unique_proportion >= self.THETA_THRESHOLD:
                    inferred_data_type = self.DATATYPES[1] #Categorical
                else:
                    inferred_data_type = self.DATATYPES[0] #Numerical
            else: # Real numbers
                inferred_data_type = self.DATATYPES[0] #Numerical
        elif is_string_dtype(s):
            lg.debug("{0} seems to be string".format(s.name))
            t = s.copy().apply(lambda x: x.strip().replace(',',''))
            t = t.apply(lambda x: x.replace('$',''))
            try:
                s_casted = pd.to_numeric(t)
                s = s_casted
                inferred_data_type = self.DATATYPES[0]
            except ValueError as ve:
                lg.debug("{0} DEFINITELY is a string".format(s.name)) # Reported as other vy default
            # df[c] = df[c].apply(lambda x: x.strip().replace(',',''))
        return inferred_data_type, s

    def run(self):
        # lg.debug("Datasets \n{0}".format(self.datasets))
        results = pd.DataFrame(columns=REPORTING_SCHEMA)
        for d in self.datasets:
            lg.info('Analyzing dataset {0}'.format(d))
            df = self.datasets[d]
            df = df.dropna(axis=1, how='all') # Drop columns which contain only NaN
            for c in df: # For each column in Dataset
                try :
                    lg.debug(df[c].dtype)
                    datatype, df[c] = self.infer_data_type(df[c])
                    # lg.debug("Results {0}".format(results))
                    if datatype == self.DATATYPES[0]: # Numerical
                        # Do something
                        lg.debug("Marking {0}".format(self.DATATYPES[0]))
                        r = self._apply_numerical_analyses(pd.Series(df[c]), d)
                        results = results.append(r)
                    elif datatype == self.DATATYPES[1]: # Categorical
                        lg.debug("Marking {0}".format(self.DATATYPES[1]))
                        r = self._apply_categorical_analyses(df[c], d) # TODO Need to explore more here
                        results = results.append(r)
                        # results = results.append([(d, c, self.DATATYPES[1], 1 )])
                    else : # Case for "Other" data type. Always the last listed datatype
                        # Log this column as other
                        r = pd.DataFrame([(d, c, self.DATATYPES[-1], 1, 0)], columns=REPORTING_SCHEMA)
                        results = results.append(r)
                except :
                    lg.exception(traceback.print_exc())
            
        self.export_results_to_csv(results, "./outputs/ConsolidateOP") 

    def _apply_categorical_analyses(self, s:pd.Series, name):
        return pd.DataFrame([(name, s.name, 'Categorical', 1, 0)], columns=REPORTING_SCHEMA)
    
    def _apply_numerical_analyses(self, s, name):
        lla = log_likelihood_analyzer()
        r = lla.score_for_series(s, name)
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
        # lg.debug("Scoring {0}".format(data))
        try:
            lg.debug("Starting scoring {0} by log_likelihood".format(data.name))
            # Best holders initialization
            best_distribution = st.norm
            best_params = (0.0, 1.0)
            best_sse = np.inf

            for distribution in self.DISTRIBUTIONS: 
                lg.info("Modelling {0}({1}) with {2}".format(data.name, len(data), distribution.name))

                ## CODE FOR DISTRIBUTION FITTING
                # Get histogram of original data
                y, x = np.histogram(data, bins=bins, density=False)
                # lg.debug("y = {0}".format(y))
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
                data_mean = np.mean(data)
                rms = np.sqrt(sse) / max(data_mean ,(np.max(data) - np.min(data)))
                # rms = np.sqrt(sse) / (np.max(data) - np.min(data))
                lg.debug("{0}.{1} by {2} = {3}".format(dataset_id, data.name, distribution.name, rms))
                observations.append((dataset_id, data.name, distribution.name, rms , data_mean))
                # observations.append((dataset_id, data.name, distribution.name, sse ))
        # else:
        #     raise ValueError('Non empty Dataset should be attached before starting comparison')
        except AttributeError as ae:
            lg.critical("{0}. {1}".format(ae, traceback.format_exc()))
        observations_df = pd.DataFrame(observations, columns=(REPORTING_SCHEMA))
        return observations_df
