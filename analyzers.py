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
from kl_divergence_analyzer import kl_divergence_analyzer
import Levenshtein
from ngram import NGram

logging.config.fileConfig('logging.conf')
lg = logging.getLogger("analyzer")

ATTRIBUTES_PROFILE_SCHEMA = ('dataset_id', 'column_name', 'distribution', 'rms_normed', 'y_mean')
RESULTS_SCHEMA = ('dataset_id1', 'column_name1', 'type1', 'dataset_id2', 'column_name2', 'type2', 'kl_divergence', 'lex_distance_lv','lex_distance_ng')

class analyzer(): # Contains configuration information common to all analyzers
    THETA_THRESHOLD = 0.5 # Threshold percentage of unique values in a number sequence beyond which it would be tagged Categorical
    
    DATATYPES = ["Numerical","Categorical", "Other"] # Data types that are inferred
    # DISTRIBUTIONS = [st.uniform, st.norm, st.zipf]
    DISTRIBUTIONS = [st.norm, st.uniform]
    DISTRIBUTION_NAMES = ["norm","uniform"] # Numerical data types that are inferred

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
        filename = filename_prefix + datetime.datetime.now().strftime("_%Y%m%d_%H%M") + ".csv"
        # df.to_csv(path_or_buf=filename, float_format='%.6f', index=False)
        df.to_csv(path_or_buf=filename, index=False)
        lg.info('Attributes profile written to {0}'.format(filename))
    
    def _export_best_kl_div_pairs(self, rdf, filename_prefix):
        # print('I am exporting {0}'.format(rdf))
        filename = filename_prefix + datetime.datetime.now().strftime("_%Y%m%d_%H%M") + ".csv"
        # df.to_csv(path_or_buf=filename, float_format='%.6f', index=False)
        # res = rdf.groupby(['dataset_id1','column_name1'], as_index=False).agg({'kl_divergence':'min', 'dataset_id2':'first', 'column_name2':'first', 'lex_distance_lv':'first', 'lex_distance_ng':'first'})
        res = rdf.groupby([RESULTS_SCHEMA[0],RESULTS_SCHEMA[1]], as_index=False).agg({
            RESULTS_SCHEMA[6]:'min', 
            RESULTS_SCHEMA[3]:'first', 
            RESULTS_SCHEMA[4]:'first', 
            RESULTS_SCHEMA[7]:'first', 
            RESULTS_SCHEMA[8]:'first'})
        res.sort_index(axis=1, inplace=True)
        res.to_csv(path_or_buf=filename, index=False)
        lg.info('Best KL divergence written to {0}'.format(filename))

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
            t = pd.to_numeric(s, errors='coerce')
            s = t.dropna()
        elif is_string_dtype(s):
            lg.debug("{0} seems to be string".format(s.name))
            try:
                t = s.copy().apply(lambda x: x.strip().replace(',',''))
                t = t.apply(lambda x: x.replace('$',''))
                s_casted = pd.to_numeric(t)
                s = s_casted
                inferred_data_type = self.DATATYPES[0]
            except ValueError as ve:
                lg.debug("{0} DEFINITELY is a string".format(s.name)) # Reported as other vy default
            except AttributeError as ae: # Pandas incorrectly inferrs column type to be object whereas it is numeric
                t = pd.to_numeric(s, errors='coerce')
                s = t.dropna()
                lg.debug("{0} invalid values in {1}".format(s.isnull().sum(), s.name))
                inferred_data_type = self.DATATYPES[0]
            # df[c] = df[c].apply(lambda x: x.strip().replace(',',''))
        lg.debug("Marking {1} as {0}".format(inferred_data_type, s.name))
        return inferred_data_type, s

    def run(self):
        # lg.debug("Datasets \n{0}".format(self.datasets))
        results = pd.DataFrame(columns=ATTRIBUTES_PROFILE_SCHEMA) # These results are JUST profiles of individual attributes
        for d in self.datasets:
            lg.info('Analyzing dataset {0}'.format(d))
            df = self.datasets[d]
            df = df.dropna(axis=1, how='all') # Drop columns which contain only NaN
            for c in df: # For each column in Dataset
                try :
                    lg.debug(df[c].dtype)
                    datatype, t = self.infer_data_type(df[c])
                    # lg.debug("Results {0}".format(results))
                    if datatype == self.DATATYPES[0]: # Numerical
                        # Do something
                        lg.debug("Marking {0}".format(self.DATATYPES[0]))
                        # r = self._apply_numerical_analyses(pd.Series(df[c]), d)
                        r = self._apply_numerical_analyses(pd.Series(t), d)
                        results = results.append(r)
                    elif datatype == self.DATATYPES[1]: # Categorical
                        lg.debug("Marking {0}".format(self.DATATYPES[1]))
                        # r = self._apply_categorical_analyses(df[c], d) # TODO Need to explore more here
                        r = self._apply_categorical_analyses(t, d) # TODO Need to explore more here
                        results = results.append(r)
                        # results = results.append([(d, c, self.DATATYPES[1], 1 )])
                    else : # Case for "Other" data type. Always the last listed datatype
                        # Log this column as other
                        r = pd.DataFrame([(d, c, self.DATATYPES[-1], 1, 0)], columns=ATTRIBUTES_PROFILE_SCHEMA)
                        results = results.append(r)
                except :
                    lg.exception(traceback.print_exc())
            
        # self.export_results_to_csv(results, "./outputs/ConsolidateOP")
        comparison_results = self._export_comparisons_to_csv(results, "./outputs/ColumnSimilarities")
        self._export_best_kl_div_pairs(comparison_results,"./outputs/BestKLDpairs")
            
    def _lexicographical_distance_lv(self, rdf, i, j):
        """ Returns lexicographical Levenshtein distance of two attributes in a dataset. 
        :parameter rdf: pd.DataFrame Contains intermediate results from distribution profiling of attributes of all datasets in the schema self.ATTRIBUTES_PROFILE_SCHEMA
        :parameter i: int
        :parameter j: int
        :rtype: double Lexicographical Distance between attributes
        """
        var1 = rdf.at[i, 'column_name']
        var2 = rdf.at[j, 'column_name']
        similarity = Levenshtein.ratio(var1,var2)
        return 1.0 - similarity

    def _lexicographical_distance_ng(self, rdf, i, j):
        """ Returns lexicographical Levenshtein distance of two attributes in a dataset. 
        :parameter rdf: pd.DataFrame Contains intermediate results from distribution profiling of attributes of all datasets in the schema self.ATTRIBUTES_PROFILE_SCHEMA
        :parameter i: int
        :parameter j: int
        :rtype: double Lexicographical Distance between attributes
        """
        var1 = rdf.at[i, 'column_name']
        var2 = rdf.at[j, 'column_name']
        similarity = NGram.compare(var1,var2)
        return 1.0 - similarity

    def _export_comparisons_to_csv(self, rdf:pd.DataFrame, filename_prefix):
        rdf.reset_index(inplace=True)
        # print("At final export : {0}".format(rdf))
        filename = filename_prefix + datetime.datetime.now().strftime("_%Y%m%d_%H%M") + ".csv"
        kl_analyzer = kl_divergence_analyzer()
        results_op = pd.DataFrame(columns=RESULTS_SCHEMA)
        lmax = len(rdf)
        for i in range(0, lmax):
            # rdf_i = rdf.iat[[i]]
            lg.debug("{0} of {1} attributes compared".format(i, lmax))
            for j in range(0 , lmax):
                if i != j:
                    # rdf_j = rdf.at[[j]]
                    # distri1 = rdf_i['distribution'].item()
                    distri1 = rdf.at[i,'distribution']
                    # distri2 = rdf_i['distribution'].item()
                    distri2 = rdf.at[j,'distribution']
                    # print("rdf[{0}] = {1} AND rdf[{2}]={3}".format(i, distri1, j, distri2))
                    if(distri1 in self.DISTRIBUTION_NAMES and distri2 in self.DISTRIBUTION_NAMES): # Both attributes are Numerical
                        # d1_name = rdf_i['dataset_id'].item()
                        d1_name = rdf.at[i,'dataset_id']
                        c1_name = rdf.at[i,'column_name']
                        d1 = self.datasets[d1_name]
                        c1 = d1[c1_name]

                        d2_name = rdf.at[j,'dataset_id']
                        c2_name = rdf.at[j, 'column_name']
                        d2 = self.datasets[d2_name]
                        c2 = d2[c2_name]
                        # print("Found {0} and {1}".format(c1.name, c2.name))
                        kl = kl_analyzer.normalized_kl_div(c1, distri1, c2, distri2)
                        #lg.debug("BS")
                        ld = self._lexicographical_distance_lv(rdf, i, j)  
                        ng = self._lexicographical_distance_ng(rdf, i, j)  

                        r = pd.DataFrame([(d1_name, c1_name, distri1, d2_name, c2_name, distri2, kl, ld, ng)], columns=RESULTS_SCHEMA)
                        results_op = results_op.append(r)                      
                                      
        results_op.to_csv(path_or_buf=filename, index=False)
        lg.info('Output written to {0}'.format(filename))
        return results_op

    def _apply_categorical_analyses(self, s:pd.Series, name):
        return pd.DataFrame([(name, s.name, 'Categorical', 1, 0)], columns=ATTRIBUTES_PROFILE_SCHEMA)
    
    def _apply_numerical_analyses(self, s, dataset_name):
        lla = log_likelihood_analyzer()
        lg.debug("{0} invalid values in {1}".format(s.isnull().sum(), s.name))
        r = lla.score_for_series(s, dataset_name)
        return r
                    

class categorical_analyzer(): #TODO Maybe we dont need these anymore
    def __init__(self):
        lg.debug("Categorical Analyzer created")

class numerical_analyzer(analyzer): #TODO Maybe we dont need these anymore
    def __init__(self):
        lg.debug("Numerical Analyzer created")
        analyzer.__init__(self)
        

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
            lowest_rms = np.inf

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
                rms = np.sqrt(sse) / max(abs(data_mean) , abs(np.max(data) - np.min(data)))
                if rms < lowest_rms:
                    lowest_rms = rms
                    best_distribution = distribution.name
                # rms = np.sqrt(sse) / (np.max(data) - np.min(data))
                lg.debug("{0}.{1} by {2} = {3}".format(dataset_id, data.name, distribution.name, rms))
                # observations.append((dataset_id, data.name, distribution.name, rms , data_mean))
                
            observations.append((dataset_id, data.name, best_distribution, rms , data_mean))
        # else:
        #     raise ValueError('Non empty Dataset should be attached before starting comparison')
        except AttributeError as ae:
            lg.critical("{0}. {1}".format(ae, traceback.format_exc()))
        observations_df = pd.DataFrame(observations, columns=(ATTRIBUTES_PROFILE_SCHEMA))
        return observations_df
