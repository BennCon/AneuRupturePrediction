import numpy as np
import scipy.stats as stats

def router(train, test, config):
    method = config['method']
    if method == None:
        return train, test

    method_config = config["methods"][method]
    if method == 'z_score':
        print("Removing outliers using z-score")
        threshold = method_config['threshold']
        means, stds = calc_stats(train)
        train = z_removal(train, means, stds, threshold)
        test = z_removal(test, means, stds, threshold)

    return train, test


def z_removal(df, means, stds, threshold=3):
    """
    Remove outliers from data, replacing them with n/a
    :param df: Dataframe
    :param means: Dictionary of means
    :param stds: Dictionary of standard deviations
    :param threshold: Threshold for outlier removal
    :return: Dataframe with outliers removed
    """
    for col in df.columns:
        if col in means and col in stds:
            mean = means[col]
            std = stds[col]
            z_scores = (df[col] - mean) / std
            outliers = df[np.abs(z_scores) > threshold]
            outlier_indices = outliers.index

            df.loc[outlier_indices, col] = np.nan
        
    return df

def calc_stats(df):
    """
    Calculate the mean and standard deviation of each column in a dataframe
    :param df: Dataframe
    :return: Dictionary of means
    :return: Dictionary of standard deviations
    """
    means = {}
    stds = {}
    for col in df.columns:
        if df[col].dtype == 'float64' or df[col].dtype == 'int64':
                mean = np.mean(df[col])
                std = np.std(df[col])

                means[col] = mean
                stds[col] = std
        
    return means, stds
