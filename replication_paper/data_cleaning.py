# ########################################################################### #
# #############   Policy learning with sensitive attributes   ############### #
# ############# N. Bearth, M. Lechner, J. Mareckova, F. Muny  ############### #
# ########################################################################### #

# FILE 1.) DATA CLEANING

# Standard libraries
import numpy as np
import pandas as pd

# Special lbraries
# pip install "flaml[automl]"
from flaml import AutoML

# Basic settings
PATH_TO_DATA = "D:\\Fabian_Muny\\Data_fair\\"
RANDOM_STATE = 0
np.random.seed(RANDOM_STATE)

# %% Data
data = pd.read_csv(PATH_TO_DATA + "original\\1203_ALMP_Data_E_v1.0.0.csv")
meta = pd.read_excel(PATH_TO_DATA + "original\\1203_ALMP_Doc_Codebook_E.xlsx")
# Swiss Active Labor Market Policy Evaluation [Dataset]. Distributed by FORS,
# Lausanne. Retrieved from https://doi.org/10.23662/FORS-DS-1203-1)

# %%% Select sample
# Knaus 2022 considers German-speaking cantons only to avoid common support
# problems
df = data[data['canton_german'] == 1].copy()
df = df.drop(columns=['canton_german', 'canton_french', 'canton_italian'])

# %%%
# Create individual identifier
df['id'] = df.index
ID_cols = ['id']

# %%% D Treatment

# Print frequencies
print(pd.concat([
    df['treatment3'].value_counts(),
    df['treatment3'].value_counts(normalize=True).round(2)], axis=1))
print(pd.concat([
    df['treatment6'].value_counts(),
    df['treatment6'].value_counts(normalize=True).round(2)], axis=1))

# Add personality programs to job search. They are very similar.
df.loc[df['treatment3'] == 'personality', 'treatment3'] = 'job search'
df.loc[df['treatment6'] == 'personality', 'treatment6'] = 'job search'
df = df.drop(columns=['personality3', 'personality6'])

# Print frequencies
print(pd.concat([
    df['treatment3'].value_counts(),
    df['treatment3'].value_counts(normalize=True).round(2)], axis=1))
print(pd.concat([
    df['treatment6'].value_counts(),
    df['treatment6'].value_counts(normalize=True).round(2)], axis=1))

# Define treatment column as numeric variable
D_cols = ['treatment6']
df['no_program6'] = 1*(df['treatment6'] == 'no program')
D_cols_onehot = [
    'no_program6', 'job_search6', 'vocational6', 'computer6', 'emp_program6',
    'language6']

# Map to integer
mappings = {}
mappings['treatment3'] = {
    'no program': 0,
    'job search': 1,
    'vocational': 2,
    'computer': 3,
    'language': 4,
    'employment': 5}
mappings['treatment6'] = mappings['treatment3']
df['treatment3'] = df['treatment3'].map(mappings['treatment3'])
df['treatment6'] = df['treatment6'].map(mappings['treatment6'])
mappings_back = {var: {b: a for a, b in mappings[
    var].items()} for var in mappings.keys()}

# %%% S Sensitive attribute

# Sensitive Attributes
S_cols_ord = ['female', "swiss"]
S_cols_unord = []
S_cols_unord_onehot = []
S_cols = S_cols_ord + S_cols_unord
S_cols_onehot = S_cols_ord + S_cols_unord_onehot
print(pd.concat([
    df[S_cols].value_counts(),
    df[S_cols].value_counts(normalize=True).round(2)], axis=1).sort_index())
print(pd.concat([
    pd.concat([df['treatment6'].map(mappings_back['treatment6']), df[
        S_cols]], axis=1).value_counts(),
    pd.concat([df['treatment6'].map(mappings_back['treatment6']), df[
        S_cols]], axis=1).value_counts(normalize=True).round(2)
    ], axis=1).sort_index())

# %%% Z Other covariates

Z_cols_ord = [
    "age",
    "canton_moth_tongue",
    "cw_age",
    "cw_cooperative",
    "cw_educ_above_voc",
    "cw_educ_tertiary",
    "cw_female",
    "cw_missing",
    "cw_own_ue",
    "cw_tenure",
    "cw_voc_degree",
    "emp_share_last_2yrs",
    "emp_spells_5yrs",
    "employability",
    "foreigner_b",
    "foreigner_c",
    "gdp_pc",
    "married",
    "other_mother_tongue",
    "past_income",
    "ue_cw_allocation1",
    "ue_cw_allocation2",
    "ue_cw_allocation3",
    "ue_cw_allocation4",
    "ue_cw_allocation5",
    "ue_cw_allocation6",
    "ue_spells_last_2yrs",
    "unemp_rate",
    'qual_degree'  # added because action variable
    ]
Z_cols_unord = [
    "city",
    "prev_job_sec_cat",
    "prev_job",
    "qual",
    ]

Z_cols_unord_onehot = [
    "city_big",
    "city_medium",
    "city_no",
    "prev_job_sec1",
    "prev_job_sec2",
    "prev_job_sec3",
    "prev_job_sec_mis",
    "prev_job_manager",
    "prev_job_self",
    "prev_job_skilled",
    "prev_job_unskilled",
    "qual_semiskilled",
    "qual_unskilled",
    "qual_wo_degree",
    ]

Z_cols = Z_cols_ord + Z_cols_unord
Z_cols_onehot = Z_cols_ord + Z_cols_unord_onehot

# All covariates
X_cols_ord = S_cols_ord + Z_cols_ord
X_cols_unord = S_cols_unord + Z_cols_unord
X_cols_unord_onehot = S_cols_unord_onehot + Z_cols_unord_onehot
X_cols = S_cols + Z_cols
X_cols_onehot = S_cols_onehot + Z_cols_onehot

print(df[X_cols_onehot].describe().T)
print(df.groupby('treatment6')[X_cols_onehot].mean().T)

# Transform to integer
print(df.info())
for col in df.columns:
    if df[col].dtype == 'object':
        mappings[col] = {
            value: idx for idx, value in enumerate(df[col].unique())}
        df[col] = df[col].map(mappings[col])
print(df.info())
mappings_back = {var: {b: a for a, b in mappings[
    var].items()} for var in mappings.keys()}

# %%% Predict pseudo program start points for control group

# Remove individuals without program if employed at prediced pseudo program
# start point
# Settings for tuning
automl_settings = {
    "time_budget": -1,  # total running time in seconds
    "max_iter": 100,  # max number of iterations
    "metric": "accuracy",
    "task": "classification",  # task type
    "estimator_list": ["rf"],
    "custom_hp": {"rf": {"n_estimators": {"domain": 50}}},
    "log_file_name": "",
    "early_stop": True,
    "verbose": 3,
}
# Estimate RF for probability of pogram start in months 4-6 given covariates
rf_late = AutoML()
rf_late.fit(
    X_train=df.loc[(df[D_cols[0]] > 0), X_cols_onehot],
    y_train=df.loc[(df[D_cols[0]] > 0), 'start_q2'],
    **automl_settings)
# Predict among non-treated
p_late = rf_late.model.estimator.set_params(n_estimators=1000).predict_proba(
    X=df.loc[(df[D_cols[0]] == 0), X_cols_onehot])[:, 1]
# Generate pseudo starting point for those in no program
df['elap'] = df['start_q2']
df.loc[df[D_cols[0]] == 0, 'elap'] = np.random.binomial(n=1, p=p_late)
print(df['elap'].value_counts())
# Drop if (predicted) program start in Q2 but employed in Q1
df = df.loc[~((df['elap'] == 1) & ((df['employed1'] == 1) | (df[
    'employed2'] == 1) | (df['employed3'] == 1)))].copy()


# %%% Y Outcome

# Calculate the outcome of interest: Months employed over 3 years
Y_cols = ["outcome0131", "outcome1324", "outcome2031", "outcome2631"]

# If program start in Q1, then months 3-33, otherwise months 6-36
df["outcome0131"] = (df['elap'] == 0) * df[[f"employed{i}" for i in range(
    3, 34)]].sum(axis=1) + (df['elap'] == 1) * df[[
        f"employed{i}" for i in range(6, 37)]].sum(axis=1)
df["outcome1324"] = (df['elap'] == 0) * df[[f"employed{i}" for i in range(
    15, 27)]].sum(axis=1) + (df['elap'] == 1) * df[[
        f"employed{i}" for i in range(18, 30)]].sum(axis=1)
df["outcome2031"] = (df['elap'] == 0) * df[[f"employed{i}" for i in range(
    22, 34)]].sum(axis=1) + (df['elap'] == 1) * df[[
        f"employed{i}" for i in range(25, 37)]].sum(axis=1)
df["outcome2631"] = (df['elap'] == 0) * df[[f"employed{i}" for i in range(
    28, 34)]].sum(axis=1) + (df['elap'] == 1) * df[[
        f"employed{i}" for i in range(31, 37)]].sum(axis=1)

# Assess outcome by treatment group
for i in Y_cols:
    print(f"\n{i}")
    print(df.assign(treatment6=df['treatment6'].map(mappings_back[
        'treatment6'])).groupby('treatment6')[i].describe())

# Keep only variables of interest
df_onehot = df[ID_cols + D_cols_onehot + Y_cols + X_cols_onehot]
df = df[ID_cols + D_cols + Y_cols + X_cols]

# %%% A Decision variables

# Define decision variables (educ, age, income, employment hist)
A_cols_ord = [
    "age",
    "past_income",
    "qual_degree",
    ]
A_cols_unord = []
A_cols_unord_onehot = []
A_cols = A_cols_ord + A_cols_unord
A_cols_onehot = A_cols_ord + A_cols_unord_onehot

# %% Save

# Save data sets
df_onehot.to_pickle(PATH_TO_DATA + "cleaned\\data_clean_onehot.pkl")
df.to_pickle(PATH_TO_DATA + "cleaned\\data_clean.pkl")

# Create Dictionary for variable names
columns = {
    'ID': ID_cols,
    'D': D_cols,
    'D_onehot': D_cols_onehot,
    'S_ord': S_cols_ord,
    'S_unord': S_cols_unord,
    'S_unord_onehot': S_cols_unord_onehot,
    'S': S_cols,
    'S_onehot': S_cols_onehot,
    'Z_ord': Z_cols_ord,
    'Z_unord': Z_cols_unord,
    'Z_unord_onehot': Z_cols_unord_onehot,
    'Z': Z_cols,
    'Z_onehot': Z_cols_onehot,
    'X_ord': X_cols_ord,
    'X_unord': X_cols_unord,
    'X_unord_onehot': X_cols_unord_onehot,
    'X': X_cols,
    'X_onehot': X_cols_onehot,
    'A_ord': A_cols_ord,
    'A_unord': A_cols_unord,
    'A_unord_onehot': A_cols_unord_onehot,
    'A': A_cols,
    'A_onehot': A_cols_onehot,
    'Y': Y_cols,
    }

# Save column names
pd.DataFrame.from_dict(columns, orient='index').T.to_pickle(
    PATH_TO_DATA + "cleaned\\columns.pkl")
pd.DataFrame.from_dict(columns, orient='index').T.to_csv(
    PATH_TO_DATA + "cleaned\\columns.csv", index=False)
# Save mappings
pd.DataFrame.from_dict(mappings, orient='index').T.to_pickle(
    PATH_TO_DATA + "cleaned\\mappings.pkl")
pd.DataFrame.from_dict(mappings, orient='index').T.to_csv(
    PATH_TO_DATA + "cleaned\\mappings.csv")
