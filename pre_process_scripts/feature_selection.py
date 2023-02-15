import numpy as np

def router(train, test, config):
    method = config['method']
    if method == None:
        retain_cols = train.columns
    else: 
        method_config = config['methods'][method] 
        if method == 'manual':
            retain_cols = method_config['features']
        elif method == 'correlation':
            retain_cols = corr_select(train, method_config['threshold'])
        
    return train[retain_cols], test[retain_cols]


#Select features by correlation, don't remove the target
def corr_select(df, threshold):
    """
    Select features by correlation
    """
    corr = df.corr()
    corr = corr.abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > threshold) and column != "ruptureStatus"]

    return [col for col in df.columns if col not in to_drop]


