# ########################################################################### #
# #############   Policy learning with sensitive attributes   ############### #
# ############# N. Bearth, M. Lechner, J. Mareckova, F. Muny  ############### #
# ########################################################################### #

# FILE 2.) ESTIMATE POLIC SCORES WITH MCF

# Standard libraries
import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from scipy.stats import chi2_contingency

# Special lbraries
# pip install mcf
from mcf import ModifiedCausalForest, McfOptPolReport

# Basic settings
PATH_TO_DATA = "D:\\Fabian_Muny\\Data_fair\\"
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


# %% Descriptive analysis of correlations
def print_correlation(A, B, A_name="A", B_name="B"):
    with np.errstate(divide='ignore', invalid="ignore"):
        ccc = np.corrcoef(A, B)[0, 1]
    print(f"corr({A_name}, {B_name}) = {ccc:.3}")


def correlation_df(A, B, A_name="", B_name=""):
    corr = pd.DataFrame(columns=A.columns, index=B.columns)
    corr.index.name = B_name
    corr.columns.name = A_name
    for j in A.columns:
        for i in B.columns:
            with np.errstate(divide='ignore', invalid="ignore"):
                corr.loc[i, j] = np.corrcoef(A.loc[:, j], B.loc[:, i])[0, 1]
    return corr


print("-"*70)
print("Correlations of sensitive attribute with treatments [-100, 100]")
print("-"*70)
print(correlation_df(
    df_onehot[cols["S_onehot"]],
    df_onehot[cols["D_onehot"]],
    "Sensitive",
    "Treatment")*100)
print("-"*70)
print("Correlations of sensitive attribute with outcome [-100, 100]")
print("-"*70)
print(correlation_df(
    df_onehot[cols["S_onehot"]],
    df_onehot[cols["Y"]],
    "Sensitive",
    "Outcome")*100)
print("-"*70)
print(
      "Correlations of sensitive attribute with decision variables "
      "[-100, 100]")
print("-"*70)
print(correlation_df(
    df_onehot[cols["S_onehot"]],
    df_onehot[cols["A_onehot"]],
    "Sensitive",
    "Decision")*100)
print("-"*70)

# Chi squared test of observed assignments
obs = pd.crosstab(df[cols['D']].squeeze(), df[cols['S']].astype(
    str).agg(''.join, axis=1))
res = chi2_contingency(obs)
print("Chi2 independence test (H0: independence):")
print(f"test statistic: {res.statistic:.5f}")
print(f"p-value: {res.pvalue:.5f}")
# Independence rejected

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

for out in cols['Y']:

    # Create an instance of the Modified Causal Forest model
    my_mcf = ModifiedCausalForest(
        var_id_name=cols['ID'],
        var_y_name=out,  # out variable
        var_d_name=cols['D'],    # Treatment variable
        var_x_name_ord=cols['X_ord'],  # Ordered covariates
        var_x_name_unord=cols['X_unord'],  # Unordered covariate
        gen_iate_eff=True,
        cf_compare_only_to_zero=True,
        gen_outpath=f"D:\\Fabian_Muny\\Results_fair\\MCF_outputs\\{set_str}_{out}"
    )
    # Train the forest
    tree_df, fill_y_df, outpath_train = my_mcf.train(df_tr)

    # Predict policy scores for data used to train allocation rules
    results_pr, _ = my_mcf.predict(df_pr)
    my_mcf.analyse(results_pr)

    # Predict policy scores for data used to evaluate allocation rules
    results_ev, _ = my_mcf.predict(df_ev)

    # Create report
    my_report = McfOptPolReport(
        mcf=my_mcf,
        outputpath=f"D:\\Fabian_Muny\\Results_fair\\MCF_outputs\\{set_str}_{out}")
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
