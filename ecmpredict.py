import pandas as pd
import sys
import re
import statsmodels.api as sm

def ecmpredict(model, newdata, init):
    if (len(newdata) < 2):
        sys.exit("Your input for 'newdata' has insufficient data.")
    
    form = model.params.index.tolist()
    
    if (sum(pd.Series(form).str.contains('^delta').tolist()) >= 1):
        xtrfctnames = [name for name in form if name.startswith('delta')]
        xtrfctnames = [re.sub(r'^delta', '', name) for name in xtrfctnames]
        xtrfct = newdata[xtrfctnames]
        xtrfct = xtrfct.diff().dropna().reset_index(drop=True)
        xtrfct = xtrfct.add_prefix('delta')
        xtrfctnames = xtrfct.columns.tolist()
    
    if (sum(pd.Series(form).str.contains('Lag1$').tolist()) >= 1):
        xeqfctnames = [name for name in form if not name.startswith('delta')]
        xeqfctnames = [re.sub(r'^delta', '', name) for name in xeqfctnames]
        xeqfctnames = [re.sub(r'Lag1$', '', name) for name in xeqfctnames]
        xeqfct = newdata.filter(xeqfctnames)
        xeqfct = xeqfct.add_suffix('Lag1')
        xeqfct = xeqfct.shift().dropna().reset_index(drop=True)
        xeqfctnames = xeqfct.columns.tolist()
        
    if (('xeqfct' in locals()) & ('xtrfct' in locals())):
        x = pd.concat([xtrfct, xeqfct], 1)
        x['yLag1'] = init
        x.columns = xtrfctnames + xeqfctnames + ['yLag1']
    elif (('xeqfct' not in locals()) & ('xtrfct' in locals())):
        x = xtrfct
        x['yLag1'] = init
        x.columns = xtrfctnames + ['yLag1']
    elif (('xeqfct' in locals()) & ('xtrfct' not in locals())):
        x = xeqfct
        x['yLag1'] = init
        x.columns = xeqfctnames + ['yLag1']
        
    if ('const' in form):
        x = sm.add_constant(x, has_constant='add')
        
    modelpred = model.predict(x.iloc[[0]])
    for i in range(1, x.shape[0]):
        x['yLag1'][i] = x['yLag1'][i] + modelpred
        modelpred = model.predict(x.iloc[[i]])
    modelpred = model.predict(x)
    modelpred = pd.concat([pd.Series([init]), modelpred]).reset_index(drop=True).cumsum()
    
    return(modelpred)