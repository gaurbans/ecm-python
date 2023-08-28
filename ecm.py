import pandas as pd
import sys
import statsmodels.api as sm

def ecm(y, xeq, xtr, includeIntercept=True, weights=None):
    if (xtr is not None):
        if (type(xtr)!=pd.core.frame.DataFrame):
            sys.exit("xtr is not of type 'Pandas DataFrame'. Please input xtr as a Pandas DataFrame.")
        if (sum(xtr.columns.str.contains('^delta|Lag[0-9]$')) > 0):
            print(Warning("You have column name(s) in xtr that begin with 'delta' or end with 'Lag[0-9]'. It is strongly recommended that you change this, otherwise the function 'ecmpredict' may result in errors or incorrect predictions."))
    
    if (xeq is not None):
        if (type(xeq)!=pd.core.frame.DataFrame):
            sys.exit("xeq is not of type 'Pandas DataFrame'. Please input xeq as a Pandas DataFrame.")
        if (len(xeq) < 2):
            sys.exit("Insufficient data for the lags specified.")
        if (sum(xeq.columns.str.contains('^delta|Lag[0-9]$')) > 0):
            print(Warning("You have column name(s) in xeq that begin with 'delta' or end with 'Lag[0-9]'. It is strongly recommended that you change this, otherwise the function 'ecmpredict' may result in errors or incorrect predictions."))
    
    if (xeq is not None):
        xeqnames = xeq.columns
        xeqnames = xeq.columns + 'Lag1'
        xeqnames = xeqnames.tolist()
        xeq = xeq.shift()
    
    if (xtr is not None):
        xtrnames = xtr.columns
        xtrnames = 'delta' + xtr.columns
        xtrnames = xtrnames.tolist()
        xtr = xtr.diff().iloc[1:]
        
    if (type(y)==pd.core.frame.DataFrame):
        if (y.shape[1] > 1):
            print(Warning("You have more than one column in y, only the first will be used"))
        y = y.iloc[:, 0]
    yLag = y[:(len(y)-1)]
        
    if ((xtr is not None) & (xeq is not None)):
        x = pd.concat([xtr, xeq.dropna(axis=0, how='any')], 1).reset_index(drop=True)
        xnames = xtrnames + xeqnames
    elif ((xtr is not None) & (xeq is None)):
        x = xtr.reset_index(drop=True)
        xnames = xtrnames
    elif ((xtr is None) & (xeq is not None)):
        x = xeq.dropna(axis=0, how='any').reset_index(drop=True)
        xnames = xeqnames
    
    x = pd.concat([x, yLag], 1)
    x.columns = xnames + ['yLag1']
    dy = y.diff().dropna().reset_index(drop=True)
    
    if (weights is not None):
        if (type(weights)==pd.core.frame.DataFrame):
            print(Warning("weights is a data.frame, only the first column will be used"))
            weights = weights.iloc[:, 0]
        if (len(weights) > x.shape[0]):
            weights = weights[(len(weights)-x.shape[0]):len(weights)]
    else:
        weights = 1
    
    if (includeIntercept==True):
        x = sm.add_constant(x, has_constant='add')
    
    lr = sm.WLS(dy, x, weights=weights).fit()
        
    return(lr)