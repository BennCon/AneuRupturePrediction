import numpy as np
from scipy import stats

def router(train, test, config):
    method = config["method"]
    if method == "complete_case":
        train = complete_case(train)
        test = complete_case(test)
    elif method == "avg":
        train_avgs = get_data_avg(train)
        train = impute_data(train, train_avgs)
        test = impute_data(test, train_avgs)

    return train, test


def complete_case(df):
    """
    Complete case analysis
    :param df: Dataframe
    :return: Dataframe with complete cases
    """
    df = df.dropna()
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


    
    
