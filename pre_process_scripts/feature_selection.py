import numpy as np


def select(df, config, return_cols=True):
    method = config['method']
    params = config['methods'][method]
    if method == 'manual':
        retain_cols = params['features']
    elif method == 'correlation':
        retain_cols = corr_select(df, params['threshold'])

    if return_cols:
        return retain_cols
    else:
        return df[retain_cols]


#Select features by correlation, don't remove the target
def corr_select(df, threshold):
    """
    Select features by correlation
    """
    corr = df.corr()
    corr = corr.abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    #Rewrite the above line with np.bool, as this is deprecated

    
    to_drop = [column for column in upper.columns if any(upper[column] > threshold) and column != "ruptureStatus"]
    return [col for col in df.columns if col not in to_drop]


