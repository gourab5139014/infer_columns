# import warnings
import numpy as np
import pandas as pd
import scipy.stats as st
import statsmodels as sm
# import matplotlib
# import matplotlib.pyplot as plt

# START CONFIGURATION SECTION

DISTRIBUTIONS = [st.uniform, st.norm, st.zipf]

# END CONFIGURATION SECTION
def get_uniform_distributed_integers(min_value, max_value, total, fliped):
    fake1 = np.linspace(min_value, max_value, total, dtype=int)
    replace_at = np.random.randint(min_value, max_value, size=fliped)
    for i in range(0, len(fake1)):
        if(i in replace_at):
            fake1[i] = np.random.randint(min_value, max_value)
    return fake1

def study_dataset_from_file(d):
    dt = pd.read_csv(d)
    print(dt.head())
    # TODO Resume by studying MLE based estimation of stat params and validate if our KL div approach is valid

if __name__ == "__main__":
    datasets = ['total_waterborne_commerce.csv']
    for d in datasets:
        study_dataset_from_file(d)