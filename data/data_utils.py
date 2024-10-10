"""
Functions to load and process the data sets used for evaluation.

Based on https://github.com/alexisjihyeross/adversarial_recourse
"""

import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'

import os
import numpy as np

from sklearn.model_selection import train_test_split

DATA_DIR = 'data/datasets'

def get_data_file(data_name):
    return os.path.join(DATA_DIR, '%s.csv' % data_name)

def process_data(data):
    if data == "compas":
        return process_compas_causal_data()
    elif data == "bail":
        return process_bail_data()
    elif data == "adult":
        return process_causal_adult()
    elif data == 'german':
        return process_german_data()
    elif data == 'loan':
        return process_loan_data()
    else:
        raise AssertionError
    
def balanced_train_test_split(X, y, test_size=0.2, random_state=None):

    # Convert to numpy (e.g., if X, Y are Pandas dataframes)
    if type(X) != np.ndarray:
        X, y = X.to_numpy(), y.to_numpy()

    # Find unique classes and their counts in the target array
    classes, class_counts = np.unique(y, return_counts=True)
    
    # Determine the minimum number of samples available in any class
    min_count = min(class_counts)
    
    train_indices = []
    test_indices = []
    
    for cls in classes:
        # Get the indices of the current class
        class_indices = np.where(y == cls)[0]
        
        # Subsample the class to match the minimum count
        if len(class_indices) > min_count:
            class_indices = np.random.choice(class_indices, min_count, replace=False)
        
        # Split the subsampled class indices into train and test sets
        train_class_indices, test_class_indices = train_test_split(
            class_indices,
            test_size=test_size,
            random_state=random_state
        )
        
        # Store the train and test indices
        train_indices.extend(train_class_indices)
        test_indices.extend(test_class_indices)
    
    # Shuffle the train and test indices
    np.random.seed(random_state)
    np.random.shuffle(train_indices)
    np.random.shuffle(test_indices)
    
    # Create the balanced train and test sets
    X_train = X[train_indices]
    y_train = y[train_indices]
    X_test = X[test_indices]
    y_test = y[test_indices]
    
    return X_train, y_train, X_test, y_test

def train_test_split_ours(X, Y, ratio=0.8):
    """
    Return a random train/test split

    Inputs:     X: np.array (N, D)
                Y: np.array (N, )
                ratio: float, percentage of the dataset used as training data

    Outputs:    X_train: np.array (M, D)
                Y_train: np.array (M, )
                X_test: np.array(N-M, D)
                Y_test: np.array(N-M, )
    """
    # Convert to numpy (e.g., if X, Y are Pandas dataframes)
    if type(X) != np.ndarray:
        X, Y = X.to_numpy(), Y.to_numpy()

    # Shuffle indices
    N_data = X.shape[0]
    train_indeces, test_indeces = train_test_split(np.arange(N_data), stratify=Y, test_size=ratio)

    # Extract train and test set
    N_train = int(N_data * ratio)
    X_train, Y_train = X[train_indeces], Y[train_indeces]
    X_test, Y_test = X[test_indeces], Y[test_indeces]
    return X_train, Y_train, X_test, Y_test


def process_compas_causal_data():
    data_file = get_data_file("compas-scores-two-years")
    compas_df = pd.read_csv(data_file, index_col=0)

    # Standard way to process the data, as done in the ProPublica notebook
    compas_df = compas_df.loc[(compas_df['days_b_screening_arrest'] <= 30) &
                              (compas_df['days_b_screening_arrest'] >= -30) &
                              (compas_df['is_recid'] != -1) &
                              (compas_df['c_charge_degree'] != "O") &
                              (compas_df['score_text'] != "NA")]
    compas_df['age'] = (pd.to_datetime(compas_df['c_jail_in']) - pd.to_datetime(compas_df['dob'])).dt.days/365

    # We use the variables in the causal graph of Nabi & Shpitser, 2018
    X = compas_df[['age', 'race', 'sex', 'priors_count']]
    X['isMale'] = X.apply(lambda row: 1 if 'Male' in row['sex'] else 0, axis=1)
    X['isCaucasian'] = X.apply(lambda row: 1 if 'Caucasian' in row['race'] else 0, axis=1)
    X = X.drop(['sex', 'race'], axis=1)

    # Swap order of features to simplify learning the SCM
    X = X[['age', 'isMale', 'isCaucasian', 'priors_count']]

    # Favourable outcome is no recidivism
    y = compas_df.apply(lambda row: 1.0 if row['two_year_recid'] == 0 else 0.0, axis=1)

    columns = X.columns
    means = [0 for i in range(X.shape[-1])]
    std = [1 for i in range(X.shape[-1])]

    # Only the number of prior counts is actionable
    compas_actionable_features = ["priors_count"]
    actionable_ids = [idx for idx, col in enumerate(X.columns) if col in compas_actionable_features]

    # Number of priors cannot go below 0
    feature_limits = np.array([[-1, 1]]).repeat(X.shape[1], axis=0) * 1e10
    feature_limits[np.where(X.columns == 'priors_count')[0]] = np.array([0, 10e10])

    # Standarize continuous features
    compas_categorical_names = ['isMale', 'isCaucasian']
    for col_idx, col in enumerate(X.columns):
        if col not in compas_categorical_names:
            means[col_idx] = X[col].mean(axis=0)
            std[col_idx] = X[col].std(axis=0)
            X[col] = (X[col] - X[col].mean(axis=0)) / X[col].std(axis=0)
            feature_limits[col_idx] = (feature_limits[col_idx] - means[col_idx]) / std[col_idx]

    # Get the indices for increasing and decreasing features
    compas_increasing_actionable_features = []
    compas_decreasing_actionable_features = ["priors_count"]
    increasing_ids = [idx for idx, col in enumerate(X.columns) if col in compas_increasing_actionable_features]
    decreasing_ids = [idx for idx, col in enumerate(X.columns) if col in compas_decreasing_actionable_features]

    constraints = {'actionable': actionable_ids, 'increasing': increasing_ids,
                  'decreasing': decreasing_ids, 'limits': feature_limits}

    return X, y, constraints


def process_causal_adult():
    data_file = get_data_file("adult")
    adult_df = pd.read_csv(data_file).reset_index(drop=True)
    adult_df.columns = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation',
                        'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week',
                        'native-country', 'label']  # proper name of each of the features
    adult_df = adult_df.dropna()

    #  We use the variables in the causal graph of Nabi & Shpitser, 2018
    adult_df = adult_df.drop(['workclass', 'fnlwgt', 'education', 'occupation', 'relationship', 'race', 'capital-gain',
                              'capital-loss'], axis=1)
    adult_df['native-country-United-States'] = adult_df.apply(lambda row: 1 if 'United-States' in row['native-country'] else 0, axis=1)
    adult_df['marital-status-Married'] = adult_df.apply(lambda row: 1 if 'Married' in row['marital-status'] else 0, axis=1)
    adult_df['isMale'] = adult_df.apply(lambda row: 1 if 'Male' in row['sex'] else 0, axis=1)
    adult_df = adult_df.drop(['native-country', 'marital-status', 'sex'], axis=1)
    X = adult_df.drop('label', axis=1)

    # Target is whether the individual has a yearly income greater than 50k
    y = adult_df['label'].replace(' <=50K', 0.0)
    y = y.apply(lambda x: 1.0 if x == ' >50K' else 0)

    # Re-arange to follow the causal graph
    columns = ['isMale', 'age', 'native-country-United-States', 'marital-status-Married', 'education-num', 'hours-per-week']
    X = X[columns]

    adult_actionable_features = ["education-num", "hours-per-week"]
    actionable_ids = [idx for idx, col in enumerate(X.columns) if col in adult_actionable_features]

    feature_limits = np.array([[-1, 1]]).repeat(X.shape[1], axis=0) * 1e10
    feature_limits[np.where(X.columns == 'education-num')[0]] = np.array([1, 16])
    feature_limits[np.where(X.columns == 'hours-per-week')[0]] = np.array([0, 100])

    # Standarize continuous features#
    means = [0 for i in range(X.shape[-1])]
    std = [1 for i in range(X.shape[-1])]
    adult_categorical_names = ['isMale', 'native-country-United-States', 'marital-status-Married']
    for col_idx, col in enumerate(X.columns):
        if col not in adult_categorical_names:
            means[col_idx] = X[col].mean(axis=0)
            std[col_idx] = X[col].std(axis=0)
            X[col] = (X[col] - X[col].mean(axis=0)) / X[col].std(axis=0)
            feature_limits[col_idx] = (feature_limits[col_idx] - means[col_idx]) / std[col_idx]

    adult_increasing_actionable_features = ["education-num"]
    adult_decreasing_actionable_features = []
    increasing_ids = [idx for idx, col in enumerate(X.columns) if col in adult_increasing_actionable_features]
    decreasing_ids = [idx for idx, col in enumerate(X.columns) if col in adult_decreasing_actionable_features]

    constraints = {'actionable': actionable_ids, 'increasing': increasing_ids,
                   'decreasing': decreasing_ids, 'limits': feature_limits}

    return X, y, constraints


def process_german_data():
    data_file = get_data_file("SouthGermanCredit.asc")
    data_df = pd.read_csv(data_file.replace(".csv", ""), sep=' ')
    data_df = data_df.dropna()

    X = data_df.drop('kredit', axis=1)
    y = data_df.apply(lambda row: 1 if row['kredit'] == 1 else 0, axis=1)

    actionable_features = ["laufzeit", "hoehe"]
    actionable_indices = [idx for idx, col in enumerate(X.columns) if col in actionable_features]

    feature_limits = np.array([[-1, 1]]).repeat(X.shape[1], axis=0) * 1e10
    feature_limits[np.where(X.columns == 'laufzeit')[0]] = np.array([0, 10e10])
    feature_limits[np.where(X.columns == 'hoehe')[0]] = np.array([0, 10e10])

    # normalize continuous features
    means = [0 for i in range(X.shape[-1])]
    std = [1 for i in range(X.shape[-1])]
    categorical_names = ['laufkont', 'moral', 'verw', 'sparkont', 'beszeit', 'rate', 'famges', 'buerge', 'wohnzeit',
                         'verm', 'weitkred', 'wohn', 'bishkred', 'beruf', 'pers', 'telef', 'gastarb']
    for col_idx, col in enumerate(X.columns):
        if col not in categorical_names:
            means[col_idx] = X[col].mean(axis=0)
            std[col_idx] = X[col].std(axis=0)
            X[col] = (X[col] - X[col].mean(axis=0)) / X[col].std(axis=0)
            feature_limits[col_idx] = (feature_limits[col_idx] - means[col_idx]) / std[col_idx]

    increasing_actionable_features = []
    increasing_ids = [idx for idx, col in enumerate(X.columns) if col in increasing_actionable_features]
    decreasing_actionable_features = []
    decreasing_ids = [idx for idx, col in enumerate(X.columns) if col in decreasing_actionable_features]

    constraints = {'actionable': actionable_indices, 'increasing': increasing_ids,
                   'decreasing': decreasing_ids, 'limits': feature_limits}

    return X, y, constraints


def process_bail_data():
    data_file = get_data_file("bail")
    bail_df = pd.read_csv(data_file)
        
    '''
    From the data documentation: 

    If (FILE = 3), the value of ALCHY is recorded as zero, but is meaningless.
    If (FILE = 3), the value of JUNKY is recorded as zero, but is meaningless.
    PRIORS: the value -9 indicates that this information is missing.
    SCHOOL: the value zero indicates that this information is missing.
    For individuals for whom RECID equals zero, the value of TIME is meaningless.

    We set these values to nan so they do not affect binning

    https://www.ncjrs.gov/pdffiles1/Digitization/115306NCJRS.pdf
    '''

    bail_df.loc[bail_df["FILE"] == 3, "ALCHY"] = np.nan
    bail_df.loc[bail_df["FILE"] == 3, "JUNKY"] = np.nan
    bail_df.loc[bail_df["PRIORS"] == -9, "PRIORS"] = np.nan
    bail_df.loc[bail_df["SCHOOL"] == 0, "SCHOOL"] = np.nan
    bail_df['label'] = bail_df.apply(lambda row: 1.0 if row['RECID'] == 0 else 0.0, axis=1)

    bail_df = bail_df.dropna()
    X = bail_df.copy()

    y = X['label']
    X = X.drop(['RECID', 'label', 'TIME', 'FILE'], axis=1)

    # Actionable features
    bail_actionable_features = ["SCHOOL", "RULE"]
    actionable_ids = [idx for idx, col in enumerate(X.columns) if col in bail_actionable_features]

    # Bounds
    feature_limits = np.array([[-1, 1]]).repeat(X.shape[1], axis=0) * 1e10
    feature_limits[np.where(X.columns == 'SCHOOL')[0]] = np.array([1, 19])
    feature_limits[np.where(X.columns == 'RULE')[0]] = np.array([0, 1e10])

    # normalize continuous features
    bail_categorical_features = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    bail_categorical_names = [X.columns[i] for i in bail_categorical_features]
    means = [0 for i in range(X.shape[-1])]
    std = [1 for i in range(X.shape[-1])]
    for col_idx, col in enumerate(X.columns):
        if col not in bail_categorical_names:
            means[col_idx] = X[col].mean(axis=0)
            std[col_idx] = X[col].std(axis=0)    
            X[col] = (X[col] - X[col].mean(axis=0)) / X[col].std(axis=0)
            feature_limits[col_idx] = (feature_limits[col_idx] - means[col_idx]) / std[col_idx]

    bail_increasing_actionable_features = ["SCHOOL"]
    increasing_ids = [idx for idx, col in enumerate(X.columns) if col in bail_increasing_actionable_features]
    bail_decreasing_actionable_features = ["RULE"]
    decreasing_ids = [idx for idx, col in enumerate(X.columns) if col in bail_decreasing_actionable_features]

    constraints = {'actionable': actionable_ids, 'increasing': increasing_ids,
                   'decreasing': decreasing_ids, 'limits': feature_limits}

    return X, y, constraints

def process_loan_data():
    X = np.load('data/datasets/loan_X.npy')
    Y = np.load('data/datasets/loan_Y.npy')

    increasing = [2, 5, 6]
    decreasing = []
    categorical = [0]
    actionable = [2, 5, 6]

    feature_limits = np.array([[-1, 1]]).repeat(X.shape[1], axis=0) * 1e10
    feature_limits[2, 1] = X[:, 2].max()

    constraints = {'actionable': actionable, 'increasing': increasing,
                   'decreasing': decreasing, 'limits': feature_limits}

    return X, Y, constraints