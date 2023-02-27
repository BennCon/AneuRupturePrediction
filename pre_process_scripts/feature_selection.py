import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel

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
        elif method == 'random_forest':
            retain_cols = random_forest_select(train, method_config)
        
    # print(f"Retaining {len(retain_cols)} features from {len(train.columns)}")
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


def random_forest_select(train, config):
    """
    Select features using random forest
    """
    n_estimators, random_state, n_jobs = config['n_estimators'], config['random_state'], config['n_jobs']

    X = train.drop("ruptureStatus", axis=1)
    y = train["ruptureStatus"]

    clf = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state, n_jobs=n_jobs)
    clf.fit(X, y)
    model = SelectFromModel(clf, prefit=True)

    selected_features = X.columns[(model.get_support())]
    
    #Add target back
    selected_features = np.append(selected_features, "ruptureStatus")
    return selected_features
