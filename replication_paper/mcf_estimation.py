# ########################################################################### #
# ############ Fairness-Aware and Interpretable Policy Learning ############# #
# ############   N. Bearth, M. Lechner, J. Mareckova, F. Muny   ############# #
# ########################################################################### #

# FILE 2.) ESTIMATE SCORES WITH MCF

# Standard libraries
import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
import time

# Special libraries
# pip install mcf
from mcf import ModifiedCausalForest, McfOptPolReport

# Start timing
start_time = time.time()

# Basic settings
PATH_TO_DATA = "D:\\Data_fair\\"
PATH_MCF_OUTPUTS = "D:\\Results_fair\\MCF_outputs\\"
RANDOM_STATE = 0
np.random.seed(RANDOM_STATE)

# %% Load Data
df = pd.read_pickle(PATH_TO_DATA + "cleaned\\data_clean.pkl")
df_onehot = pd.read_pickle(PATH_TO_DATA + "cleaned\\data_clean_onehot.pkl")
cols = {k: [x for x in v if pd.notna(x)] for k, v in pd.read_pickle(
    PATH_TO_DATA + "cleaned\\columns.pkl").to_dict(orient='list').items()}
mappings = {k: {x[0]: int(x[1]) for x in v.items() if pd.notna(
    x[1])} for k, v in pd.read_pickle(
        PATH_TO_DATA + "cleaned\\mappings.pkl").to_dict(orient='dict').items()}
mappings_back = {var: {b: a for a, b in mappings[
    var].items()} for var in mappings.keys()}

# Choose setting
set_str = "05_application"

# Define for which outcomes the estimation should be performed
outcomes_list = ['outcome0131']
# All outcomes: ['outcome0131', 'outcome1324', 'outcome2031', 'outcome2631']


# %% Sample splits

# First, split 80% (train + test) and 20% (validation)
df_tr_pr, df_ev = train_test_split(
    df, test_size=0.2, stratify=df[cols['D'][0]], random_state=RANDOM_STATE)

# Then, split the 80% into two equal halves (40% and 40%)
df_tr, df_pr = train_test_split(
    df_tr_pr, test_size=0.5, stratify=df_tr_pr[cols['D'][0]],
    random_state=RANDOM_STATE)

# Check the size of each split
print(f"Train size: {len(df_tr)}")
print(f"Test size: {len(df_pr)}")
print(f"Validation size: {len(df_ev)}")

# Check if the proportions of 'treat' are preserved
print("-"*50)
print("Proportion of treatments in original data:")
print(df[cols['D'][0]].value_counts(normalize=True))
print("-"*50)
print("Proportion of treatments in df_tr:")
print(df_tr[cols['D'][0]].value_counts(normalize=True))
print("-"*50)
print("Proportion of treatments in df_pr:")
print(df_pr[cols['D'][0]].value_counts(normalize=True))
print("-"*50)
print("Proportion of treatments in df_ev:")
print(df_ev[cols['D'][0]].value_counts(normalize=True))
print("-"*50)

# Save data sets
pd.to_pickle(df_tr, PATH_TO_DATA + "\\cleaned\\data_clean_tr.pkl")
pd.to_pickle(df_pr, PATH_TO_DATA + "\\cleaned\\data_clean_pr.pkl")
pd.to_pickle(df_ev, PATH_TO_DATA + "\\cleaned\\data_clean_ev.pkl")
df_tr.to_csv(PATH_TO_DATA + "\\cleaned\\data_clean_tr.csv", index=False)
df_pr.to_csv(PATH_TO_DATA + "\\cleaned\\data_clean_pr.csv", index=False)
df_ev.to_csv(PATH_TO_DATA + "\\cleaned\\data_clean_ev.csv", index=False)

# %% Estimation
# Based on the following example
# https://github.com/MCFpy/mcf/blob/main/examples/mcf_optpol_combined.py

for out in outcomes_list:

    # Create an instance of the Modified Causal Forest model
    my_mcf = ModifiedCausalForest(
        var_id_name=cols['ID'],
        var_y_name=out,  # out variable
        var_d_name=cols['D'],    # Treatment variable
        var_x_name_ord=cols['X_ord'],  # Ordered covariates
        var_x_name_unord=cols['X_unord'],  # Unordered covariate
        gen_iate_eff=True,
        cf_compare_only_to_zero=True,
        gen_outpath=f"{PATH_MCF_OUTPUTS}{set_str}_{out}"
    )
    # Train the forest
    my_mcf.train(df_tr)

    # Predict scores for data used to train allocation rules
    results_pr = my_mcf.predict(df_pr)
    my_mcf.analyse(results_pr)

    # Predict scores for data used to evaluate allocation rules
    results_ev = my_mcf.predict(df_ev)

    # Create report
    my_report = McfOptPolReport(
        mcf=my_mcf,
        outputpath=f"{PATH_MCF_OUTPUTS}{set_str}_{out}")
    my_report.report()

    # Save IATEs
    os.makedirs(
        PATH_TO_DATA + "\\cleaned\\" + set_str + "_" + out + "\\",
        exist_ok=True)
    pd.to_pickle(
        results_pr['iate_data_df'],
        PATH_TO_DATA + "\\cleaned\\" + set_str + "_" + out + "\\iates_pr.pkl")
    pd.to_pickle(
        results_ev['iate_data_df'],
        PATH_TO_DATA + "\\cleaned\\" + set_str + "_" + out + "\\iates_ev.pkl")
    results_pr['iate_data_df'].to_csv(
        PATH_TO_DATA + "\\cleaned\\" + set_str + "_" + out + "\\iates_pr.csv",
        index=False)
    results_ev['iate_data_df'].to_csv(
        PATH_TO_DATA + "\\cleaned\\" + set_str + "_" + out + "\\iates_ev.csv",
        index=False)

end_time = time.time()

print("Start:", start_time)
print("End:", end_time)
print("Elapsed (seconds):", end_time - start_time)
