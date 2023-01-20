import numpy as np


def select(df, config, return_cols=False):
    """
    If return _cols is True, returns the columns to be retained,
    otherwise returns the modified dataframe
    """
    method = config['method']
    modified_df = None
    method_config = config['methods'][method]
    if method == 'manual':
        modified_df = manual_select(method_config['features'])
    elif method == 'correlation':
        modified_df = corr_select(method_config['threshold'])
    else:
        raise Exception('Feature selection method not supported')
    
    if return_cols:
        return modified_df.columns
    else:
        return modified_df



def manual_select(df, features):
    """
    Select features manually
    """
    return df[features]

#Select features by correlation, don't remove the target
def corr_select(df, threshold):
    """
    Select features by correlation
    """
    corr = df.corr()
    corr = corr.abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(np.bool))
    
    to_drop = [column for column in upper.columns if any(upper[column] > threshold) and column != "ruptureStatus"]
    return df.drop(to_drop, axis=1)


