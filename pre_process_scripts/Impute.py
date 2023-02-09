import numpy as np
from scipy import stats

def impute():
    return complete_case()

def complete_case(df):
    """
    Complete case analysis
    :param df: Dataframe
    :return: Dataframe with complete cases
    """
    df = df.dropna()
    return df

#Average imputation
def average_impute(df):
    """
    Impute missing values with the mean/mode
    :param df: Dataframe
    :return: Dataframe with imputed values
    """
    for col in df.columns:
        #if numeric
        if df[col].dtype == 'float64' or df[col].dtype == 'int64':
            df[col].fillna(df[col].mean(), inplace=True)
        #if categorical
        elif df[col].dtype == 'object':
            df[col].fillna(df[col].mode()[0], inplace=True)
    return df

def get_data_avg(df):
    """
    Gets the average value of each column
    :param df: Dataframe
    :return: Dictionary of average values
    """
    avg = {}
    for col in df.columns:
        #if numeric
        if df[col].dtype == 'float64' or df[col].dtype == 'int64':
            avg[col] = df[col].mean()
        #if categorical
        elif df[col].dtype == 'object':
            avg[col] = df[col].mode()[0]
    return avg

def impute_data(df, avg):
    """
    Impute missing values with the mean/mode
    :param df: Dataframe
    :param avg: Dictionary of average values
    :return: Dataframe with imputed values
    """
    for col in df.columns:
        df[col].fillna(avg[col], inplace=True)
    return df

def remove_outliers(df):
    """
    Remove outliers from data, replacing them with n/a
    :param df: Dataframe
    :return: Dataframe with outliers removed
    """
    for col in df.columns:
        if df[col].dtype == 'float64' or df[col].dtype == 'int64':
            mean = np.mean(df[col])
            std = np.std(df[col])
            
            z_scores = (df[col] - mean) / std
            outliers = df[np.abs(z_scores) > 3]
            outlier_indices = outliers.index

            df.loc[outlier_indices, col] = np.nan
    
    return df


    
    
