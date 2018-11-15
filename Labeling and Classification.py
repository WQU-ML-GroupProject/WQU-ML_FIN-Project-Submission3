#!/usr/bin/env python
# coding: utf-8

# <h1>Table of Contents<span class="tocSkip"></span></h1>
# <div class="toc"><ul class="toc-item"><li><span><a href="#Labeling-and-MetaLabeling" data-toc-modified-id="Labeling-and-MetaLabeling-1" data-vivaldi-spatnav-clickable="1"><span class="toc-item-num">1&nbsp;&nbsp;</span>Labeling and MetaLabeling</a></span><ul class="toc-item"><li><span><a href="#Overview" data-toc-modified-id="Overview-1.1" data-vivaldi-spatnav-clickable="1"><span class="toc-item-num">1.1&nbsp;&nbsp;</span>Overview</a></span></li><li><span><a href="#Code-Snippets" data-toc-modified-id="Code-Snippets-1.2" data-vivaldi-spatnav-clickable="1"><span class="toc-item-num">1.2&nbsp;&nbsp;</span>Code Snippets</a></span><ul class="toc-item"><li><span><a href="#Symmetric-CUSUM-Filter-[2.5.2.1]" data-toc-modified-id="Symmetric-CUSUM-Filter-[2.5.2.1]-1.2.1" data-vivaldi-spatnav-clickable="1"><span class="toc-item-num">1.2.1&nbsp;&nbsp;</span>Symmetric CUSUM Filter [2.5.2.1]</a></span></li><li><span><a href="#Daily-Volatility-Estimator-[3.1]" data-toc-modified-id="Daily-Volatility-Estimator-[3.1]-1.2.2" data-vivaldi-spatnav-clickable="1"><span class="toc-item-num">1.2.2&nbsp;&nbsp;</span>Daily Volatility Estimator [3.1]</a></span></li><li><span><a href="#Triple-Barrier-Labeling-Method-[3.2]" data-toc-modified-id="Triple-Barrier-Labeling-Method-[3.2]-1.2.3" data-vivaldi-spatnav-clickable="1"><span class="toc-item-num">1.2.3&nbsp;&nbsp;</span>Triple-Barrier Labeling Method [3.2]</a></span></li><li><span><a href="#Gettting-Time-of-First-Touch-(getEvents)-[3.3],-[3.6]" data-toc-modified-id="Gettting-Time-of-First-Touch-(getEvents)-[3.3],-[3.6]-1.2.4" data-vivaldi-spatnav-clickable="1"><span class="toc-item-num">1.2.4&nbsp;&nbsp;</span>Gettting Time of First Touch (getEvents) [3.3], [3.6]</a></span></li><li><span><a href="#Adding-Vertical-Barrier-[3.4]" data-toc-modified-id="Adding-Vertical-Barrier-[3.4]-1.2.5" data-vivaldi-spatnav-clickable="1"><span class="toc-item-num">1.2.5&nbsp;&nbsp;</span>Adding Vertical Barrier [3.4]</a></span></li><li><span><a href="#Labeling-for-side-and-size-[3.5]" data-toc-modified-id="Labeling-for-side-and-size-[3.5]-1.2.6" data-vivaldi-spatnav-clickable="1"><span class="toc-item-num">1.2.6&nbsp;&nbsp;</span>Labeling for side and size [3.5]</a></span></li><li><span><a href="#Expanding-getBins-to-Incorporate-Meta-Labeling-[3.7]" data-toc-modified-id="Expanding-getBins-to-Incorporate-Meta-Labeling-[3.7]-1.2.7" data-vivaldi-spatnav-clickable="1"><span class="toc-item-num">1.2.7&nbsp;&nbsp;</span>Expanding getBins to Incorporate Meta-Labeling [3.7]</a></span></li><li><span><a href="#Dropping-Unnecessary-Labels-[3.8]" data-toc-modified-id="Dropping-Unnecessary-Labels-[3.8]-1.2.8" data-vivaldi-spatnav-clickable="1"><span class="toc-item-num">1.2.8&nbsp;&nbsp;</span>Dropping Unnecessary Labels [3.8]</a></span></li><li><span><a href="#Linear-Partitions-[20.4.1]" data-toc-modified-id="Linear-Partitions-[20.4.1]-1.2.9" data-vivaldi-spatnav-clickable="1"><span class="toc-item-num">1.2.9&nbsp;&nbsp;</span>Linear Partitions [20.4.1]</a></span></li><li><span><a href="#multiprocessing-snippet-[20.7]" data-toc-modified-id="multiprocessing-snippet-[20.7]-1.2.10" data-vivaldi-spatnav-clickable="1"><span class="toc-item-num">1.2.10&nbsp;&nbsp;</span>multiprocessing snippet [20.7]</a></span></li><li><span><a href="#single-thread-execution-for-debugging-[20.8]" data-toc-modified-id="single-thread-execution-for-debugging-[20.8]-1.2.11" data-vivaldi-spatnav-clickable="1"><span class="toc-item-num">1.2.11&nbsp;&nbsp;</span>single-thread execution for debugging [20.8]</a></span></li><li><span><a href="#Example-of-async-call-to-multiprocessing-lib-[20.9]" data-toc-modified-id="Example-of-async-call-to-multiprocessing-lib-[20.9]-1.2.12" data-vivaldi-spatnav-clickable="1"><span class="toc-item-num">1.2.12&nbsp;&nbsp;</span>Example of async call to multiprocessing lib [20.9]</a></span></li><li><span><a href="#Unwrapping-the-Callback-[20.10]" data-toc-modified-id="Unwrapping-the-Callback-[20.10]-1.2.13" data-vivaldi-spatnav-clickable="1"><span class="toc-item-num">1.2.13&nbsp;&nbsp;</span>Unwrapping the Callback [20.10]</a></span></li><li><span><a href="#Pickle-Unpickling-Objects-[20.11]" data-toc-modified-id="Pickle-Unpickling-Objects-[20.11]-1.2.14" data-vivaldi-spatnav-clickable="1"><span class="toc-item-num">1.2.14&nbsp;&nbsp;</span>Pickle Unpickling Objects [20.11]</a></span></li></ul></li></ul></li><li><span><a href="#Exercises" data-toc-modified-id="Exercises-2" data-vivaldi-spatnav-clickable="1"><span class="toc-item-num">2&nbsp;&nbsp;</span>Exercises</a></span><ul class="toc-item"><li><span><a href="#Import-Dataset" data-toc-modified-id="Import-Dataset-2.1" data-vivaldi-spatnav-clickable="1"><span class="toc-item-num">2.1&nbsp;&nbsp;</span>Import Dataset</a></span></li><li><span><a href="#[3.1]-Form-Dollar-Bars" data-toc-modified-id="[3.1]-Form-Dollar-Bars-2.2" data-vivaldi-spatnav-clickable="1"><span class="toc-item-num">2.2&nbsp;&nbsp;</span>[3.1] Form Dollar Bars</a></span><ul class="toc-item"><li><span><a href="#(a)-Run-cusum-filter-with-threshold-equal-to-std-dev-of-daily-returns" data-toc-modified-id="(a)-Run-cusum-filter-with-threshold-equal-to-std-dev-of-daily-returns-2.2.1" data-vivaldi-spatnav-clickable="1"><span class="toc-item-num">2.2.1&nbsp;&nbsp;</span>(a) Run cusum filter with threshold equal to std dev of daily returns</a></span></li><li><span><a href="#(b)-Add-vertical-barrier" data-toc-modified-id="(b)-Add-vertical-barrier-2.2.2" data-vivaldi-spatnav-clickable="1"><span class="toc-item-num">2.2.2&nbsp;&nbsp;</span>(b) Add vertical barrier</a></span></li><li><span><a href="#(c)-Apply-triple-barrier-method-where-ptSl-=-[1,1]-and-t1-is-the-series-created-in-1.b" data-toc-modified-id="(c)-Apply-triple-barrier-method-where-ptSl-=-[1,1]-and-t1-is-the-series-created-in-1.b-2.2.3" data-vivaldi-spatnav-clickable="1"><span class="toc-item-num">2.2.3&nbsp;&nbsp;</span>(c) Apply triple-barrier method where <code>ptSl = [1,1]</code> and <code>t1</code> is the series created in <code>1.b</code></a></span></li><li><span><a href="#(d)-Apply-getBins-to-generate-labels" data-toc-modified-id="(d)-Apply-getBins-to-generate-labels-2.2.4" data-vivaldi-spatnav-clickable="1"><span class="toc-item-num">2.2.4&nbsp;&nbsp;</span>(d) Apply <code>getBins</code> to generate labels</a></span></li></ul></li><li><span><a href="#[3.2]-Use-snippet-3.8-to-drop-under-populated-labels" data-toc-modified-id="[3.2]-Use-snippet-3.8-to-drop-under-populated-labels-2.3" data-vivaldi-spatnav-clickable="1"><span class="toc-item-num">2.3&nbsp;&nbsp;</span>[3.2] Use snippet 3.8 to drop under-populated labels</a></span></li><li><span><a href="#[3.4]-Develop-moving-average-crossover-strategy.-For-each-obs.-the-model-suggests-a-side-but-not-size-of-the-bet" data-toc-modified-id="[3.4]-Develop-moving-average-crossover-strategy.-For-each-obs.-the-model-suggests-a-side-but-not-size-of-the-bet-2.4" data-vivaldi-spatnav-clickable="1"><span class="toc-item-num">2.4&nbsp;&nbsp;</span>[3.4] Develop moving average crossover strategy. For each obs. the model suggests a side but not size of the bet</a></span><ul class="toc-item"><li><span><a href="#(a)-Derive-meta-labels-for-ptSl-=-[1,2]-and-t1-where-numdays=1.-Use-as-trgt-dailyVol-computed-by-snippet-3.1-(get-events-with-sides)" data-toc-modified-id="(a)-Derive-meta-labels-for-ptSl-=-[1,2]-and-t1-where-numdays=1.-Use-as-trgt-dailyVol-computed-by-snippet-3.1-(get-events-with-sides)-2.4.1" data-vivaldi-spatnav-clickable="1"><span class="toc-item-num">2.4.1&nbsp;&nbsp;</span>(a) Derive meta-labels for <code>ptSl = [1,2]</code> and <code>t1</code> where <code>numdays=1</code>. Use as <code>trgt</code> dailyVol computed by snippet 3.1 (get events with sides)</a></span></li><li><span><a href="#(b)-Train-Random-Forest-to-decide-whether-to-trade-or-not-{0,1}-since-underlying-model-(crossing-m.a.)-has-decided-the-side,-{-1,1}" data-toc-modified-id="(b)-Train-Random-Forest-to-decide-whether-to-trade-or-not-{0,1}-since-underlying-model-(crossing-m.a.)-has-decided-the-side,-{-1,1}-2.4.2" data-vivaldi-spatnav-clickable="1"><span class="toc-item-num">2.4.2&nbsp;&nbsp;</span>(b) Train Random Forest to decide whether to trade or not <code>{0,1}</code> since underlying model (crossing m.a.) has decided the side, <code>{-1,1}</code></a></span></li></ul></li><li><span><a href="#[3.5]-Develop-mean-reverting-Bollinger-Band-Strategy.-For-each-obs.-model-suggests-a-side-but-not-size-of-the-bet." data-toc-modified-id="[3.5]-Develop-mean-reverting-Bollinger-Band-Strategy.-For-each-obs.-model-suggests-a-side-but-not-size-of-the-bet.-2.5" data-vivaldi-spatnav-clickable="1"><span class="toc-item-num">2.5&nbsp;&nbsp;</span>[3.5] Develop mean-reverting Bollinger Band Strategy. For each obs. model suggests a side but not size of the bet.</a></span><ul class="toc-item"><li><span><a href="#(a)-Derive-meta-labels-for-ptSl=[0,2]-and-t1-where-numdays=1.-Use-as-trgt-dailyVol." data-toc-modified-id="(a)-Derive-meta-labels-for-ptSl=[0,2]-and-t1-where-numdays=1.-Use-as-trgt-dailyVol.-2.5.1" data-vivaldi-spatnav-clickable="1"><span class="toc-item-num">2.5.1&nbsp;&nbsp;</span>(a) Derive meta-labels for <code>ptSl=[0,2]</code> and <code>t1</code> where <code>numdays=1</code>. Use as <code>trgt</code> dailyVol.</a></span></li><li><span><a href="#(b)-train-random-forest-to-decide-to-trade-or-not.-Use-features:-volatility,-serial-correlation,-and-the-crossing-moving-averages-from-exercise-2." data-toc-modified-id="(b)-train-random-forest-to-decide-to-trade-or-not.-Use-features:-volatility,-serial-correlation,-and-the-crossing-moving-averages-from-exercise-2.-2.5.2" data-vivaldi-spatnav-clickable="1"><span class="toc-item-num">2.5.2&nbsp;&nbsp;</span>(b) train random forest to decide to trade or not. Use features: volatility, serial correlation, and the crossing moving averages from exercise 2.</a></span></li><li><span><a href="#(c)-What-is-accuracy-of-predictions-from-primary-model-if-the-secondary-model-does-not-filter-bets?-What-is-classification-report?" data-toc-modified-id="(c)-What-is-accuracy-of-predictions-from-primary-model-if-the-secondary-model-does-not-filter-bets?-What-is-classification-report?-2.5.3" data-vivaldi-spatnav-clickable="1"><span class="toc-item-num">2.5.3&nbsp;&nbsp;</span>(c) What is accuracy of predictions from primary model if the secondary model does not filter bets? What is classification report?</a></span></li></ul></li></ul></li></ul></div>

# # Labeling and MetaLabeling

# ## Overview
# 
# In this chapter of the book AFML, De Prado introduces several novel techniques for labeling returns for the purposes of supervised machine learning. 
# 
# First he identifies the typical issues of fixed-time horizon labeling methods - primarily that it is easy to mislabel a return due to dynamic nature of volatility throughout a trading period.
# 
# More importantly he addresses a major overlooked aspect of the financial literature. He emphasizes that every investment strategy makes use of stop-loss limits of some kind, whether those are enforced by a margin call, risk department or self-imposed. He highlights how unrealistic it is to test/implement/propagate a strategy that profits from positions that would have been stopped out. 
# 
# > That virtually no publication accounts for that when labeling observations tells you something about the current state of financial literature.
# >
# > -De Prado, "Advances in Financial Machine Learning", pg.44
# 
# He also introduces a technique called metalabeling, which is used to augment a strategy by improving recall while also reducing the likelihood of overfitting.

# In[2]:


get_ipython().run_line_magic('load_ext', 'watermark')
get_ipython().run_line_magic('watermark', '')

get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')

# import standard libs
from IPython.display import display
from IPython.core.debugger import set_trace as bp
from pathlib import PurePath, Path
import sys
import time
from collections import OrderedDict as od
import re
import os
import json

# import python scientific stack
import pandas as pd
import pandas_datareader.data as web
pd.set_option('display.max_rows', 100)
from dask import dataframe as dd
from dask.diagnostics import ProgressBar
from multiprocessing import cpu_count
pbar = ProgressBar()
pbar.register()
import numpy as np
import scipy.stats as stats
import statsmodels.api as sm
from numba import jit
import math
import ffn

# import visual tools
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns

plt.style.use('seaborn-talk')
plt.style.use('bmh')
#plt.rcParams['font.family'] = 'DejaVu Sans Mono'
plt.rcParams['font.size'] = 9.5
plt.rcParams['font.weight'] = 'medium'
plt.rcParams['figure.figsize'] = 10,7
blue, green, red, purple, gold, teal = sns.color_palette('colorblind', 6)

# import util libs
from tqdm import tqdm, tqdm_notebook
import warnings
warnings.filterwarnings("ignore")
import missingno as msno

RANDOM_STATE = 777

print()
get_ipython().run_line_magic('watermark', '-p pandas,pandas_datareader,dask,numpy,sklearn,statsmodels,scipy,ffn,matplotlib,seaborn')


# ## Code Snippets
# 
# Below I reproduce all the relevant code snippets found in the book that are necessary to work through the excercises found at the end of chapter 3.
'''

from src.utils.utils import *
import src.features.bars as brs
import src.features.snippets as snp
'''
# ### Symmetric CUSUM Filter [2.5.2.1]

# In[2]:
from pathlib import PurePath, Path
import sys
import time
import os
import json
os.environ['THEANO_FLAGS'] = 'device=cpu'
import pymc3 as pm
import pandas as pd
import numpy as np
import dask 
import dask.dataframe
import decimal
import logzero
from logzero import logger
import matplotlib.pyplot as plt
import seaborn as sns
#from src.CONSTANTS import *
blue, green, red, purple, gold, teal = sns.color_palette('colorblind', 6)
#=============================================================================
## setup logger

#======================================================================
## setup logger

LOG_FORMAT='%(color)s[%(levelname)s %(asctime)s.%(msecs)03d %(module)s:%(lineno)d]%(end_color)s %(message)s'
LOG_DATE_FORMAT='%Y-%m-%d %I:%M:%S'

def setup_system_logger(out_log_fp, pdir, logger):
    """fn: setup logger for various package modules

    Params
    ------
    out_log_fp: str
        log file fp name doesn't include extension fn will add it
    logger: logzero logger object

    Returns
    -------
    logger: logzero logger instance
    """
    now = pd.to_datetime('now', utc=True)
    file_ = out_log_fp+f'_{now.date()}.log'
    logfile = Path(pdir/'logs'/file_).as_posix()
    check_path(logfile)
    formatter = logzero.LogFormatter(fmt=LOG_FORMAT, datefmt=LOG_DATE_FORMAT)
    logzero.setup_default_logger(logfile=logfile, formatter=formatter)
    return logger

#=============================================================================
# general utils

def get_relative_project_dir(project_repo_name=None):
    """helper fn to get local project directory"""
    current_working_directory = Path.cwd()
    cwd_parts = current_working_directory.parts
    while cwd_parts[-1] != project_repo_name:
        current_working_directory = current_working_directory.parent
        cwd_parts = current_working_directory.parts
    return current_working_directory

def check_path(fp):
    """fn: to create file directory if it doesn't exist"""
    if not Path(fp).exists():

        if len(Path(fp).suffix) > 0: # check if file
            Path(fp).parent.mkdir(exist_ok=True, parents=True)

        else: # or directory
            Path(fp).mkdir(exist_ok=True, parents=True)

def cprint(df):
    if not isinstance(df, (pd.DataFrame, dask.dataframe.DataFrame)):
        try:
            df = df.to_frame()
        except:
            raise ValueError('object cannot be coerced to df')

    print('-'*79)
    print('dataframe information')
    print('-'*79)
    print(df.tail(5))
    print('-'*50)
    print(df.info())
    print('-'*79)
    print()

get_range = lambda df, col: (df[col].min(), df[col].max())
#=============================================================================
# system utils
def decimal_round(val, prec=1e-4):
    """wrapper for rounding according to precision
    """
    DD = decimal.Decimal
    val_ = DD(val).quantize(DD(f'{prec}'), rounding=decimal.ROUND_DOWN)
    return float(val_)

#=============================================================================
# fn: code adapted from https://github.com/jonsedar/pymc3_vs_pystan/blob/master/convenience_functions.py
def custom_describe(df, nidx=3, nfeats=20):
    ''' Concat transposed topN rows, numerical desc & dtypes '''

    print(df.shape)
    nrows = df.shape[0]
    
    rndidx = np.random.randint(0,len(df),nidx)
    dfdesc = df.describe().T

    for col in ['mean','std']:
        dfdesc[col] = dfdesc[col].apply(lambda x: np.round(x,2))
 
    dfout = pd.concat((df.iloc[rndidx].T, dfdesc, df.dtypes), axis=1, join='outer')
    dfout = dfout.loc[df.columns.values]
    dfout.rename(columns={0:'dtype'}, inplace=True)
    
    # add count nonNAN, min, max for string cols
    nan_sum = df.isnull().sum()
    dfout['count'] = nrows - nan_sum
    dfout['min'] = df.min().apply(lambda x: x[:6] if type(x) == str else x)
    dfout['max'] = df.max().apply(lambda x: x[:6] if type(x) == str else x)
    dfout['nunique'] = df.apply(pd.Series.nunique)
    dfout['nan_count'] = nan_sum
    dfout['pct_nan'] = nan_sum / nrows
    
    return dfout.iloc[:nfeats, :]


def plot_tsne(dftsne, ft_num, ft_endog='is_vw'):
    ''' Convenience fn: scatterplot t-sne rep with cat or cont color'''

    pal = 'cubehelix'
    leg = True

    if ft_endog in ft_num:
        pal = 'BuPu'
        leg = False

    g = sns.lmplot('x', 'y', dftsne.sort(ft_endog), hue=ft_endog
           ,palette=pal, fit_reg=False, size=7, legend=leg
           ,scatter_kws={'alpha':0.7,'s':100, 'edgecolor':'w', 'lw':0.4})
    _ = g.axes.flat[0].set_title('t-SNE rep colored by {}'.format(ft_endog))


def trace_median(x):
    return pd.Series(np.median(x,0), name='median')

def plot_traces(trcs, retain=2500, varnames=None):
    ''' Convenience fn: plot traces with overlaid means and values '''
    df_smry = pm.summary(trcs[-retain:], varnames=varnames)

    if varnames: nrows = len(varnames)
    else: nrows = len(trcs.varnames)
    
    plt.style.use('seaborn-dark-palette')
    plt.rcParams['font.family'] = 'DejaVu Sans Mono'
    line_cols = ['mean','hpd_2.5','hpd_97.5']
    ax = pm.traceplot(trcs[-retain:], varnames=varnames, figsize=(12, nrows*1.5), 
                      lines={k: v[line_cols[0]] for k,v in df_smry.iterrows()})

    for i,var in enumerate(df_smry.index):
        ax[i,0].axvline(df_smry.loc[var,line_cols[1]],color=red)
        ax[i,0].axvline(df_smry.loc[var,line_cols[2]], color=blue)

    for i, mn in enumerate(df_smry['mean']):
        try:
            ax[i,0].annotate('{:.2f}'.format(mn), xy=(mn,0), xycoords='data',
                             xytext=(5,10), textcoords='offset points', rotation=90,
                             va='bottom', fontsize='large', color='#AA0022')
        except: 
            pass
        
    for i, mn in enumerate(df_smry['hpd_2.5']):
        try:
            ax[i,0].annotate('{:.2f}'.format(mn), xy=(mn,15), xycoords='data',
                             xytext=(5,10), textcoords='offset points', rotation=90,
                             va='top', fontsize='medium', color='#AA0022')
        except: 
            pass
        
    for i, mn in enumerate(df_smry['hpd_97.5']):
        try:
            ax[i,0].annotate('{:.2f}'.format(mn), xy=(mn,15), xycoords='data',
                             xytext=(5,10), textcoords='offset points', rotation=90,
                             va='top', fontsize='medium', color=blue)#'#AA0022')
        except: 
            pass  
        
        import pandas as pd
import numpy as np
from numba import jit
from tqdm import tqdm

#========================================================
def returns(s):
    arr = np.diff(np.log(s))
    return (pd.Series(arr, index=s.index[1:]))
#========================================================
def tick_bars(df, price_column, m):
    '''
    compute tick bars

    # args
        df: pd.DataFrame()
        column: name for price data
        m: int(), threshold value for ticks
    # returns
        idx: list of indices
    '''
    t = df[price_column]
    ts = 0
    idx = []
    for i, x in enumerate(tqdm(t)):
        ts += 1
        if ts >= m:
            idx.append(i)
            ts = 0
            continue
    return idx

def tick_bar_df(df, price_column, m):
    idx = tick_bars(df, price_column, m)
    return df.iloc[idx]
#========================================================
def volume_bars(df, volume_column, m):
    '''
    compute volume bars

    # args
        df: pd.DataFrame()
        column: name for volume data
        m: int(), threshold value for volume
    # returns
        idx: list of indices
    '''
    t = df[volume_column]
    ts = 0
    idx = []
    for i, x in enumerate(tqdm(t)):
        ts += x
        if ts >= m:
            idx.append(i)
            ts = 0
            continue
    return idx

def volume_bar_df(df, volume_column, m):
    idx = volume_bars(df, volume_column, m)
    return df.iloc[idx]
#========================================================
def dollar_bars(df, dv_column, m):
    '''
    compute dollar bars

    # args
        df: pd.DataFrame()
        column: name for dollar volume data
        m: int(), threshold value for dollars
    # returns
        idx: list of indices
    '''
    t = df[dv_column]
    ts = 0
    idx = []
    for i, x in enumerate(tqdm(t)):
        ts += x
        if ts >= m:
            idx.append(i)
            ts = 0
            continue
    return idx

def dollar_bar_df(df, dv_column, m):
    idx = dollar_bars(df, dv_column, m)
    return df.iloc[idx]
#========================================================

@jit(nopython=True)
def numba_isclose(a,b,rel_tol=1e-09,abs_tol=0.0):
    return np.fabs(a-b) <= np.fmax(rel_tol*np.fmax(np.fabs(a), np.fabs(b)), abs_tol)

@jit(nopython=True)
def bt(p0, p1, bs):
    #if math.isclose((p1 - p0), 0.0, abs_tol=0.001):
    if numba_isclose((p1-p0),0.0,abs_tol=0.001):
        b = bs[-1]
        return b
    else:
        b = np.abs(p1-p0)/(p1-p0)
        return b

@jit(nopython=True)
def get_imbalance(t):
    bs = np.zeros_like(t)
    for i in np.arange(1, bs.shape[0]):
        t_bt = bt(t[i-1], t[i], bs[:i-1])
        bs[i-1] = t_bt
    return bs[:-1] # remove last value
import sys
import time
import pandas as pd
import numpy as np
import copyreg, types
from tqdm import tqdm

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use('seaborn-talk')
plt.style.use('bmh')
#plt.rcParams['font.family'] = 'DejaVu Sans Mono'
plt.rcParams['font.size'] = 9.5
plt.rcParams['font.weight'] = 'medium'

# =======================================================
# Symmetric CUSUM Filter [2.5.2.1]
def getTEvents(gRaw, h):
    """cusum filter

    args
    ----
        gRaw: array-like
        h: int() or float()

    returns
    -------
        pd.DatetimeIndex()
    """
    tEvents, sPos, sNeg = [], 0, 0
    diff = np.log(gRaw).diff().dropna().abs()
    for i in tqdm(diff.index[1:]):
        try:
            pos, neg = float(sPos+diff.loc[i]), float(sNeg+diff.loc[i])
        except Exception as e:
            print(e)
            print(sPos+diff.loc[i], type(sPos+diff.loc[i]))
            print(sNeg+diff.loc[i], type(sNeg+diff.loc[i]))
            break

        sPos, sNeg=max(0., pos), min(0., neg)
        if sNeg<-h:
            sNeg=0;tEvents.append(i)
        elif sPos>h:
            sPos=0;tEvents.append(i)
    return pd.DatetimeIndex(tEvents)



# =======================================================
# Triple-Barrier Labeling Method [3.2]
def applyPtSlOnT1(close,events,ptSl,molecule):
    # apply stop loss/profit taking, if it takes place before t1 (end of event)
    events_=events.loc[molecule]
    out=events_[['t1']].copy(deep=True)
    if ptSl[0]>0: pt=ptSl[0]*events_['trgt']
    else: pt=pd.Series(index=events.index) # NaNs
    if ptSl[1]>0: sl=-ptSl[1]*events_['trgt']
    else: sl=pd.Series(index=events.index) # NaNs
    for loc,t1 in events_['t1'].fillna(close.index[-1]).iteritems():
        df0=close[loc:t1] # path prices
        df0=(df0/close[loc]-1)*events_.at[loc,'side'] # path returns
        out.loc[loc,'sl']=df0[df0<sl[loc]].index.min() # earliest stop loss
        out.loc[loc,'pt']=df0[df0>pt[loc]].index.min() # earliest profit taking
    return out
# =======================================================
# Gettting Time of First Touch (getEvents) [3.3]
def getEvents(close, tEvents, ptSl, trgt, minRet, numThreads,t1=False, side=None):
    #1) get target
    trgt=trgt.loc[tEvents]
    trgt=trgt[trgt>minRet] # minRet
    #2) get t1 (max holding period)
    if t1 is False:t1=pd.Series(pd.NaT, index=tEvents)
    #3) form events object, apply stop loss on t1
    if side is None:side_,ptSl_=pd.Series(1.,index=trgt.index), [ptSl[0],ptSl[0]]
    else: side_,ptSl_=side.loc[trgt.index],ptSl[:2]
    events=(pd.concat({'t1':t1,'trgt':trgt,'side':side_}, axis=1)
            .dropna(subset=['trgt']))
    df0=mpPandasObj(func=applyPtSlOnT1,pdObj=('molecule',events.index),
                    numThreads=numThreads,close=close,events=events,
                    ptSl=ptSl_)
    events['t1']=df0.dropna(how='all').min(axis=1) #pd.min ignores nan
    if side is None:events=events.drop('side',axis=1)
    return events
# =======================================================
# Adding Vertical Barrier [3.4]
def addVerticalBarrier(tEvents, close, numDays=1):
    t1=close.index.searchsorted(tEvents+pd.Timedelta(days=numDays))
    t1=t1[t1<close.shape[0]]
    t1=(pd.Series(close.index[t1],index=tEvents[:t1.shape[0]]))
    return t1
# =======================================================
# Labeling for side and size [3.5]
def _getBins(events,close,t1=None):
    #1) prices aligned with events
    events_=events.dropna(subset=['t1'])
    px=events_.index.union(events_['t1'].values).drop_duplicates()
    px=close.reindex(px,method='bfill')
    #2) create out object
    out=pd.DataFrame(index=events_.index)
    out['ret']=px.loc[events_['t1'].values].values/px.loc[events_.index]-1
    out['bin']=np.sign(out['ret'])
    # where out index and t1 (vertical barrier) intersect label 0
    try:
        locs = out.query('index in @t1').index
        out.loc[locs, 'bin'] = 0
    except:
        pass
    return out
# =======================================================
# Expanding getBins to Incorporate Meta-Labeling [3.7]
def getBins(events, close):
    '''
    Compute event's outcome (including side information, if provided).
    events is a DataFrame where:
    -events.index is event's starttime
    -events['t1'] is event's endtime
    -events['trgt'] is event's target
    -events['side'] (optional) implies the algo's position side
    Case 1: ('side' not in events): bin in (-1,1) <-label by price action
    Case 2: ('side' in events): bin in (0,1) <-label by pnl (meta-labeling)
    '''
    #1) prices aligned with events
    events_=events.dropna(subset=['t1'])
    px=events_.index.union(events_['t1'].values).drop_duplicates()
    px=close.reindex(px,method='bfill')
    #2) create out object
    out=pd.DataFrame(index=events_.index)
    out['ret']=px.loc[events_['t1'].values].values/px.loc[events_.index]-1
    if 'side' in events_:out['ret']*=events_['side'] # meta-labeling
    out['bin']=np.sign(out['ret'])
    if 'side' in events_:out.loc[out['ret']<=0,'bin']=0 # meta-labeling
    return out
# =======================================================
# Dropping Unnecessary Labels [3.8]
def dropLabels(events, minPct=.05):
    # apply weights, drop labels with insufficient examples
    while True:
        df0=events['bin'].value_counts(normalize=True)
        if df0.min()>minPct or df0.shape[0]<3:break
        print('dropped label: ', df0.argmin(),df0.min())
        events=events[events['bin']!=df0.argmin()]
    return events
# =======================================================
# Linear Partitions [20.4.1]
def linParts(numAtoms,numThreads):
    # partition of atoms with a single loop
    parts=np.linspace(0,numAtoms,min(numThreads,numAtoms)+1)
    parts=np.ceil(parts).astype(int)
    return parts

def nestedParts(numAtoms,numThreads,upperTriang=False):
    # partition of atoms with an inner loop
    parts,numThreads_=[0],min(numThreads,numAtoms)
    for num in range(numThreads_):
        part=1+4*(parts[-1]**2+parts[-1]+numAtoms*(numAtoms+1.)/numThreads_)
        part=(-1+part**.5)/2.
        parts.append(part)
    parts=np.round(parts).astype(int)
    if upperTriang: # the first rows are heaviest
        parts=np.cumsum(np.diff(parts)[::-1])
        parts=np.append(np.array([0]),parts)
    return parts
# =======================================================
# multiprocessing snippet [20.7]
def mpPandasObj(func,pdObj,numThreads=24,mpBatches=1,linMols=True,**kargs):
    '''
    Parallelize jobs, return a dataframe or series
    + func: function to be parallelized. Returns a DataFrame
    + pdObj[0]: Name of argument used to pass the molecule
    + pdObj[1]: List of atoms that will be grouped into molecules
    + kwds: any other argument needed by func

    Example: df1=mpPandasObj(func,('molecule',df0.index),24,**kwds)
    '''
    import pandas as pd
    #if linMols:parts=linParts(len(argList[1]),numThreads*mpBatches)
    #else:parts=nestedParts(len(argList[1]),numThreads*mpBatches)
    if linMols:parts=linParts(len(pdObj[1]),numThreads*mpBatches)
    else:parts=nestedParts(len(pdObj[1]),numThreads*mpBatches)

    jobs=[]
    for i in range(1,len(parts)):
        job={pdObj[0]:pdObj[1][parts[i-1]:parts[i]],'func':func}
        job.update(kargs)
        jobs.append(job)
    if numThreads==1:out=processJobs_(jobs)
    else: out=processJobs(jobs,numThreads=numThreads)
    if isinstance(out[0],pd.DataFrame):df0=pd.DataFrame()
    elif isinstance(out[0],pd.Series):df0=pd.Series()
    else:return out
    for i in out:df0=df0.append(i)
    df0=df0.sort_index()
    return df0
# =======================================================
# single-thread execution for debugging [20.8]
def processJobs_(jobs):
    # Run jobs sequentially, for debugging
    out=[]
    for job in jobs:
        out_=expandCall(job)
        out.append(out_)
    return out
# =======================================================
# Example of async call to multiprocessing lib [20.9]
import multiprocessing as mp
import datetime as dt

#________________________________
def reportProgress(jobNum,numJobs,time0,task):
    # Report progress as asynch jobs are completed
    msg=[float(jobNum)/numJobs, (time.time()-time0)/60.]
    msg.append(msg[1]*(1/msg[0]-1))
    timeStamp=str(dt.datetime.fromtimestamp(time.time()))
    msg=timeStamp+' '+str(round(msg[0]*100,2))+'% '+task+' done after '+ \
        str(round(msg[1],2))+' minutes. Remaining '+str(round(msg[2],2))+' minutes.'
    if jobNum<numJobs:sys.stderr.write(msg+'\r')
    else:sys.stderr.write(msg+'\n')
    return
#________________________________
def processJobs(jobs,task=None,numThreads=24):
    # Run in parallel.
    # jobs must contain a 'func' callback, for expandCall
    if task is None:task=jobs[0]['func'].__name__
    pool=mp.Pool(processes=numThreads)
    outputs,out,time0=pool.imap_unordered(expandCall,jobs),[],time.time()
    # Process asyn output, report progress
    for i,out_ in enumerate(outputs,1):
        out.append(out_)
        reportProgress(i,len(jobs),time0,task)
    pool.close();pool.join() # this is needed to prevent memory leaks
    return out
# =======================================================
# Unwrapping the Callback [20.10]
def expandCall(kargs):
    # Expand the arguments of a callback function, kargs['func']
    func=kargs['func']
    del kargs['func']
    out=func(**kargs)
    return out
# =======================================================
# Pickle Unpickling Objects [20.11]
def _pickle_method(method):
    func_name=method.im_func.__name__
    obj=method.im_self
    cls=method.im_class
    return _unpickle_method, (func_name,obj,cls)
#________________________________
def _unpickle_method(func_name,obj,cls):
    for cls in cls.mro():
        try:func=cls.__dict__[func_name]
        except KeyError:pass
        else:break
    return func.__get__(obj,cls)
#________________________________

# =======================================================
# Estimating uniqueness of a label [4.1]
def mpNumCoEvents(closeIdx,t1,molecule):
    '''
    Compute the number of concurrent events per bar.
    +molecule[0] is the date of the first event on which the weight will be computed
    +molecule[-1] is the date of the last event on which the weight will be computed

    Any event that starts before t1[modelcule].max() impacts the count.
    '''
    #1) find events that span the period [molecule[0],molecule[-1]]
    t1=t1.fillna(closeIdx[-1]) # unclosed events still must impact other weights
    t1=t1[t1>=molecule[0]] # events that end at or after molecule[0]
    t1=t1.loc[:t1[molecule].max()] # events that start at or before t1[molecule].max()
    #2) count events spanning a bar
    iloc=closeIdx.searchsorted(np.array([t1.index[0],t1.max()]))
    count=pd.Series(0,index=closeIdx[iloc[0]:iloc[1]+1])
    for tIn,tOut in t1.iteritems():count.loc[tIn:tOut]+=1.
    return count.loc[molecule[0]:t1[molecule].max()]
# =======================================================
# Estimating the average uniqueness of a label [4.2]
def mpSampleTW(t1,numCoEvents,molecule):
    # Derive avg. uniqueness over the events lifespan
    wght=pd.Series(index=molecule)
    for tIn,tOut in t1.loc[wght.index].iteritems():
        wght.loc[tIn]=(1./numCoEvents.loc[tIn:tOut]).mean()
    return wght
# =======================================================
# Sequential Bootstrap [4.5.2]
## Build Indicator Matrix [4.3]
def getIndMatrix(barIx,t1):
    # Get Indicator matrix
    indM=(pd.DataFrame(0,index=barIx,columns=range(t1.shape[0])))
    for i,(t0,t1) in enumerate(t1.iteritems()):indM.loc[t0:t1,i]=1.
    return indM
# =======================================================
# Compute average uniqueness [4.4]
def getAvgUniqueness(indM):
    # Average uniqueness from indicator matrix
    c=indM.sum(axis=1) # concurrency
    u=indM.div(c,axis=0) # uniqueness
    avgU=u[u>0].mean() # avg. uniqueness
    return avgU
# =======================================================
# return sample from sequential bootstrap [4.5]
def seqBootstrap(indM,sLength=None):
    # Generate a sample via sequential bootstrap
    if sLength is None:sLength=indM.shape[1]
    phi=[]
    while len(phi)<sLength:
        avgU=pd.Series()
        for i in indM:
            indM_=indM[phi+[i]] # reduce indM
            avgU.loc[i]=getAvgUniqueness(indM_).iloc[-1]
        prob=avgU/avgU.sum() # draw prob
        phi+=[np.random.choice(indM.columns,p=prob)]
    return phi
# =======================================================
# Determination of sample weight by absolute return attribution [4.10]
def mpSampleW(t1,numCoEvents,close,molecule):
    # Derive sample weight by return attribution
    ret=np.log(close).diff() # log-returns, so that they are additive
    wght=pd.Series(index=molecule)
    for tIn,tOut in t1.loc[wght.index].iteritems():
        wght.loc[tIn]=(ret.loc[tIn:tOut]/numCoEvents.loc[tIn:tOut]).sum()
    return wght.abs()

# =======================================================
# fractionally differentiated features snippets
# =======================================================

# get weights
def getWeights(d,size):
    # thres>0 drops insignificant weights
    w=[1.]
    for k in range(1,size):
        w_ = -w[-1]/k*(d-k+1)
        w.append(w_)
    w=np.array(w[::-1]).reshape(-1,1)
    return w

def getWeights_FFD(d,thres):

    w,k=[1.],1

    while True:

        w_=-w[-1]/k*(d-k+1)

        if abs(w_)<thres:break

        w.append(w_);k+=1

    return np.array(w[::-1]).reshape(-1,1)

# =======================================================
# expanding window fractional differentiation

def fracDiff(series, d, thres=0.01):
    '''
    Increasing width window, with treatment of NaNs
    Note 1: For thres=1, nothing is skipped
    Note 2: d can be any positive fractional, not necessarily
        bounded between [0,1]
    '''
    #1) Compute weights for the longest series
    w=getWeights(d, series.shape[0])
    #2) Determine initial calcs to be skipped based on weight-loss threshold
    w_=np.cumsum(abs(w))
    w_ /= w_[-1]
    skip = w_[w_>thres].shape[0]
    #3) Apply weights to values
    df={}
    for name in series.columns:
        seriesF, df_=series[[name]].fillna(method='ffill').dropna(), pd.Series()
        for iloc in range(skip, seriesF.shape[0]):
            loc=seriesF.index[iloc]
            if not np.isfinite(series.loc[loc,name]).any():continue # exclude NAs
            try:
                df_.loc[loc]=np.dot(w[-(iloc+1):,:].T, seriesF.loc[:loc])[0,0]
            except:
                continue
        df[name]=df_.copy(deep=True)
    df=pd.concat(df,axis=1)
    return df

# =======================================================
# fixed-width window fractional differentiation

def fracDiff_FFD(series,d,thres=1e-5):
    # Constant width window (new solution)
    w = getWeights_FFD(d,thres)
    width = len(w)-1
    df={}
    for name in series.columns:
        seriesF, df_=series[[name]].fillna(method='ffill').dropna(), pd.Series()
        for iloc1 in range(width,seriesF.shape[0]):
            loc0,loc1=seriesF.index[iloc1-width], seriesF.index[iloc1]
            test_val = series.loc[loc1,name] # must resample if duplicate index
            if isinstance(test_val, (pd.Series, pd.DataFrame)):
                test_val = test_val.resample('1m').mean()
            if not np.isfinite(test_val).any(): continue # exclude NAs
            try:
                df_.loc[loc1]=np.dot(w.T, seriesF.loc[loc0:loc1])[0,0]
            except:
                continue
        df[name]=df_.copy(deep=True)
    df=pd.concat(df,axis=1)
    return df

"""
def fracDiff_FFD(series,d,thres=1e-5):
    '''
    Constant width window (new solution)
    Note 1: thres determines the cut-off weight for the window
    Note 2: d can be any positive fractional, not necessarily
        bounded [0,1].
    '''
    #1) Compute weights for the longest series
    w=getWeights_FFD(d, thres) ## WHERE IS THIS FUNCTION IN THE BOOK
    width=len(w)-1
    #2) Apply weights to values
    df={}
    for name in series.columns:
        seriesF, df_=series[[name]].fillna(method='ffill').dropna(), pd.Series()
        for iloc1 in range(width,seriesF.shape[0]):
            loc0,loc1=seriesF.index[iloc1-width], seriesF.index[iloc1]
            if not np.isfinite(series.loc[loc1,name]): continue # exclude NAs
            df_.loc[loc1]=np.dot(w.T, seriesF.loc[loc0:loc1])[0,0]
        df[name]=df_.copy(deep=True)
    df=pd.concat(df,axis=1)
    return df
"""
# =======================================================
# finding the min. D value that passes ADF test

def plotMinFFD(df0, thres=1e-5):
    # pg. 85
    from statsmodels.tsa.stattools import adfuller
    import matplotlib.pyplot as plt

    out=pd.DataFrame(columns=['adfStat','pVal','lags','nObs','95% conf','corr'])
    for d in np.linspace(0,1,11):
        df1=np.log(df0[['close']]).resample('1D').last() # downcast to daily obs
        df2=fracDiff_FFD(df1,d,thres=thres)
        corr=np.corrcoef(df1.loc[df2.index,'close'],df2['close'])[0,1]
        df2=adfuller(df2['close'],maxlag=1,regression='c',autolag=None)
        out.loc[d]=list(df2[:4])+[df2[4]['5%']]+[corr] # with critical value
    f,ax=plt.subplots(figsize=(9,5))
    out[['adfStat','corr']].plot(ax=ax, secondary_y='adfStat')
    plt.axhline(out['95% conf'].mean(),linewidth=1,color='r',linestyle='dotted')
    return out

# =======================================================
# Modeling snippets
# =======================================================

# =======================================================
# Purging observations in the training set (7.1)

def getTrainTimes(t1,testTimes):
    """
    Given testTimes, find the times of the training observations
    -t1.index: Time when the observation started
    -t1.value: Time when the observation ended
    -testTimes: Times of testing observations
    """
    trn=t1.copy(deep=True)
    for i,j in testTimes.iteritems():
        df0=trn[(i<=trn.index)&(trn.index<=j)].index # train starts within test
        df1=trn[(i<=trn)&(trn<=j)].index # train ends within test
        df2=trn[(trn.index<=i)&(j<=trn)].index # train envelops test
        trn=trn.drop(df0.union(df1).union(df2))
    return trn

# =======================================================
# Embargo on Training Observations (7.2)

def getEmbargoTimes(times,pctEmbargo):
    # Get embargo time for each bar
    step=int(times.shape[0]*pctEmbargo)
    if step==0:
        mbrg=pd.Series(times,index=times)
    else:
        mbrg=pd.Series(times[step:],index=times[:-step])
        mbrg=mbrg.append(pd.Series(times[-1],index=times[-step:]))
    return mbrg

## Examples

# testtimes=pd.Series(mbrg[dt1],index=[dt0]) # include embargo before purge
# trainTimes=getTrainTimes(t1,testTimes)
# testTimes=t1.loc[dt0:dt1].index

# =======================================================
# Cross-validation class when observations overlap (7.3)

from sklearn.model_selection._split import _BaseKFold
class PurgedKFold(_BaseKFold):
    """
    Extend KFold class to work with labels that span intervals
    The train is purged of observations overlapping test-label intervals
    Test set is assumed contiguous (shuffle=False), w/o training samples in between
    """
    def __init__(self,n_splits=3,t1=None,pctEmbargo=0.):
        if not isinstance(t1,pd.Series):
            raise ValueError('Label Through Dates must be a pd.Series')
        super(PurgedKFold,self).__init__(n_splits,shuffle=False,random_state=None)
        self.t1=t1
        self.pctEmbargo=pctEmbargo

    def split(self,X,y=None,groups=None):
        if (X.index==self.t1.index).sum()!=len(self.t1):
            raise ValueError('X and ThruDateValues must have the same index')

        # TODO: grouping function combinations insert here??
        # manage groups by using label in dataframe?
        # use combinations + group label to split into chunks??

        indices=np.arange(X.shape[0])
        mbrg=int(X.shape[0]*self.pctEmbargo)
        test_starts=[
            (i[0],i[-1]+1) for i in np.array_split(np.arange(X.shape[0]),
                                                   self.n_splits)
        ]
        for i,j in test_starts:
            t0=self.t1.index[i] # start of test set
            test_indices=indices[i:j]
            maxT1Idx=self.t1.index.searchsorted(self.t1[test_indices].max())
            train_indices=self.t1.index.searchsorted(self.t1[self.t1<=t0].index)
            if maxT1Idx<X.shape[0]: # right train ( with embargo)
                train_indices=np.concatenate((train_indices, indices[maxT1Idx+mbrg:]))
            yield train_indices,test_indices

# =======================================================
# CV score implements purgedKfold & embargo (7.4)

def cvScore(clf,X,y,sample_weight,scoring='neg_log_loss',
            t1=None,cv=None,cvGen=None,pctEmbargo=None):
    if scoring not in ['neg_log_loss','accuracy']:
        raise Exception('wrong scoring method.')
    from sklearn.metrics import log_loss,accuracy_score
    idx = pd.IndexSlice
    if cvGen is None:
        cvGen=PurgedKFold(n_splits=cv,t1=t1,pctEmbargo=pctEmbargo) # purged
    score=[]
    for train,test in cvGen.split(X=X):
        fit=clf.fit(X=X.iloc[idx[train],:],y=y.iloc[idx[train]],
                    sample_weight=sample_weight.iloc[idx[train]].values)
        if scoring=='neg_log_loss':
            prob=fit.predict_proba(X.iloc[idx[test],:])
            score_=-log_loss(y.iloc[idx[test]], prob,
                             sample_weight=sample_weight.iloc[idx[test]].values,
                             labels=clf.classes_)
        else:
            pred=fit.predict(X.iloc[idx[test],:])
            score_=accuracy_score(y.iloc[idx[test]],pred,
                                  sample_weight=sample_weight.iloc[idx[test]].values)
        score.append(score_)
    return np.array(score)

# =======================================================
# Plot ROC-AUC for purgedKFold

def crossValPlot(skf,classifier,X,y):
    """Code adapted from:
        sklearn crossval example
    """
    from itertools import cycle
    from sklearn.metrics import roc_curve, auc
    from scipy import interp

    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)
    idx = pd.IndexSlice
    f,ax = plt.subplots(figsize=(10,7))
    i = 0
    for train, test in skf.split(X, y):
        probas_ = (classifier.fit(X.iloc[idx[train]], y.iloc[idx[train]])
                   .predict_proba(X.iloc[idx[test]]))
        # Compute ROC curve and area the curve
        fpr, tpr, thresholds = roc_curve(y.iloc[idx[test]], probas_[:, 1])
        tprs.append(interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)
        ax.plot(fpr, tpr, lw=1, alpha=0.3,
                 label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc))

        i += 1

    ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
             label='Luck', alpha=.8)

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    ax.plot(mean_fpr, mean_tpr, color='b',
             label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
             lw=2, alpha=.8)

    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    ax.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                     label=r'$\pm$ 1 std. dev.')

    ax.set_xlim([-0.05, 1.05])
    ax.set_ylim([-0.05, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Receiver operating characteristic example')
    ax.legend(bbox_to_anchor=(1,1))

#=======================================================
# Feature Importance snippets
#=======================================================

#=======================================================
# 8.2 Mean Decrease Impurity (MDI)

def featImpMDI(fit,featNames):
    # feat importance based on IS mean impurity reduction
    # only works with tree based classifiers
    df0={i:tree.feature_importances_ for i,tree
         in enumerate(fit.estimators_)}
    df0=pd.DataFrame.from_dict(df0,orient='index')
    df0.columns=featNames
    df0=df0.replace(0,np.nan) # b/c max_features=1
    imp=(pd.concat({'mean':df0.mean(),
                    'std':df0.std()*df0.shape[0]**-0.5},
                   axis=1))
    imp/=imp['mean'].sum()
    return imp

#=======================================================
# 8.3 Mean Decrease Accuracy (MDA)

def featImpMDA(clf,X,y,cv,sample_weight,t1,pctEmbargo,scoring='neg_log_loss'):
    # feat imporant based on OOS score reduction
    if scoring not in ['neg_log_loss','accuracy']:
        raise ValueError('wrong scoring method.')
    from sklearn.metrics import log_loss, accuracy_score
    cvGen=PurgedKFold(n_splits=cv,t1=t1,pctEmbargo=pctEmbargo) # purged cv
    scr0,scr1=pd.SEries(), pd.DataFrame(columns=X.columns)

    for i,(train,test) in enumerate(cvGen.split(X=X)):
        X0,y0,w0=X.iloc[train,:],y.iloc[train],sample_weight.iloc[train]
        X1,y1,w1=X.iloc[test,:],y.iloc[test],sample_weight.iloc[test]
        fit=clf.fit(X=X0,y=y0,sample_weight=w0.values)
        if scoring=='neg_log_loss':
            prob=fit.predict_proba(X1)
            scr0.loc[i]=-log_loss(y1,prob,sample_weight=w1.values,
                                  labels=clf.classes_)
        else:
            pred=fit.predict(X1)
            scr0.loc[i]=accuracy_score(y1,pred,sample_weight=w1.values)

    for j in X.columns:
        X1_=X1.copy(deep=True)
        np.random.shuffle(X1_[j].values) # permutation of a single column
        if scoring=='neg_log_loss':
            prob=fit.predict_proba(X1_)
            scr1.loc[i,j]=-log_loss(y1,prob,sample_weight=w1.values,
                                    labels=clf.classes_)
        else:
            pred=fit.predict(X1_)
            scr1.loc[i,j]=accuracy_score(y1,pred,sample_weight=w1.values)
    imp=(-scr1).add(scr0,axis=0)
    if scoring=='neg_log_loss':imp=imp/-scr1
    else: imp=imp/(1.-scr1)
    imp=(pd.concat({'mean':imp.mean(),
                    'std':imp.std()*imp.shape[0]**-0.5},
                   axis=1))
    return imp,scr0.mean()

#=======================================================
# 8.4 Single Feature Importance (SFI)

def auxFeatImpSFI(featNames,clf,trnsX,cont,scoring,cvGen):
    imp=pd.DataFrame(columns=['mean','std'])
    for featName in featNames:
        df0=cvScore(clf,X=trnsX[[featName]],y=cont['bin'],
                    sample_weight=cont['w'],scoring=scoring,cvGen=cvGen)
        imp.loc[featName,'mean']=df0.mean()
        imp.loc[featName,'std']=df0.std()*df0.shape[0]**-0.5
    return imp

#=======================================================
# 8.5 Computation of Orthogonal Features

def get_eVec(dot,varThres):
    # compute eVec from dot proc matrix, reduce dimension
    eVal,eVec=np.linalg.eigh(dot)
    idx=eVal.argsort()[::-1] # arugments for sorting eVal desc.
    eVal,eVec=eVal[idx],eVec[:,idx]
    #2) only positive eVals
    eVal=(pd.Series(eVal,index=['PC_'+str(i+1)
                                for i in range(eVal.shape[0])]))
    eVec=(pd.DataFrame(eVec,index=dot.index,columns=eVal.index))
    eVec=eVec.loc[:,eVal.index]
    #3) reduce dimension, form PCs
    cumVar=eVal.cumsum()/eVal.sum()
    dim=cumVar.values.searchsorted(varThres)
    eVal,eVec=eVal.iloc[:dim+1],eVec.iloc[:,:dim+1]
    return eVal,eVec

def orthoFeats(dfx,varThres=0.95):
    # given a DataFrame, dfx, of features, compute orthofeatures dfP
    dfZ=dfx.sub(dfx.mean(),axis=1).div(dfx.std(),axis=1) # standardize
    dot=(pd.DataFrame(np.dot(dfZ.T,dfZ),
                      index=dfx.columns,
                      columns=dfx.columns))
    eVal,eVec=get_eVec(dot,varThres)
    dfP=np.dot(dfZ,eVec)
    return dfP

#=======================================================
# 8.6 Computation of weighted kendall's tau between feature importance and inverse PCA ranking

#from scipy.stats import weightedtau
#featImp=np.array([0.55,0.33,0.07,0.05]) # feature importance
#pcRank=np.array([1,2,3,4],dtype=np.float) # PCA rank
#weightedtau(featImp,pcRank**-1)[0]

#=======================================================
# 8.7 Creating a Synthetic Dataset

def getTestData(n_features=40,n_informative=10,n_redundant=10,n_samples=10_000):
    # generate a random dataset for a classification problem
    from sklearn.datasets import make_classification
    kwds=dict(n_samples=n_samples,n_features=n_feautres,
              n_informative=n_informative,n_redundant=n_redundant,
              random_state=0,shuffle=False)
    trnsX,cont=make_classification(**kwds)
    df0=(pd.DatetimeIndex(periods=n_samples, freq=pd.tseries.offsets.BDay(),
                          end=pd.datetime.today()))
    trnsX,cont=(pd.DataFrame(trnsX,index=df0),
                pd.Series(cont,index=df0).to_frame('bin'))
    df0=['I_'+str(i) for i in range(n_informative)]+['R_'+str(i) for i in range(n_redundant)]
    df0+=['N_'+str(i) for i in range(n_features-len(df0))]
    trnsX.columns=df0
    cont['w']=1./cont.shape[0]
    cont['t1']=pd.Series(cont.index,index=cont.index)
    return trnsX,cont

#=======================================================
# 8.8 Calling Feature Importance for Any Method

def featImportances(trnsX,cont,n_estimators=1000,cv=10,
                    max_samples=1.,numThreads=11,pctEmbargo=0,
                    scoring='accuracy',method='SFI',minWLeaf=0.,**kargs):
    # feature importance from a random forest
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import BaggingClassifier
    #from mpEngine import mpPandasObj
    n_jobs=(-1 if numThreads>1 else 1) # run 1 thread w/ ht_helper in dirac1
    #1) prepare classifier,cv. max_features=1, to prevent masking
    clf=DecisionTreeClassifier(criterion='entropy',max_features=1,
                               class_weight='balanced',
                               min_weight_fraction_leaf=minWLeaf)
    clf=BaggingClassifier(base_estimator=clf,n_estimators=n_estimators,
                          max_features=1.,max_samples=max_samples,
                          oob_score=True,n_jobs=n_jobs)
    fit=clf.fit(X=trnsX,y=cont['bin'],sample_weight=cont['w'].values)
    oob=fit.oob_score_
    if method=='MDI':
        imp=featImpMDI(fit,featNames=trnsX.columns)
        oos=cvScore(clf,X=trnsX,y=cont['bin'],cv=cv,sample_weight=cont['w'],
                    t1=cont['t1'],pctEmbargo=pctEmbargo,scoring=scoring).mean()
    elif method=='MDA':
        imp,oos=featImpMDA(clf,X=trnsX,y=cont['bin'],cv=cv,
                           sample_weight=cont['w'],t1=cont['t1'],
                           pctEmbargo=pctEmbargo,scoring=scoring)
    elif method=='SFI':
        cvGen=PurgedKFold(n_splits=cv,t1=cont['t1'],pctEmbargo=pctEmbargo)
        oos=cvScore(clf,X=trnsX,y=cont['bin'],sample_weight=cont['w'],
                    scoring=scoring,cvGen=cvGen).mean()
        clf.n_jobs=1 # parallelize auxFeatImpSFI rather than clf
        imp=pmPandasObj(auxFeatImpSFI,('featNames',trnsX.columns),numThreads,
                        clf=clf,trnsX=trnsX,cont=cont,scoring=scoring,cvGen=cvGen)
    return imp,oob,oos

#=======================================================
# 8.9 Calling All Components

def testFunc(n_features=40,n_informative=10,n_redundant=10,n_estimators=1000,
             n_samples=10000,cv=10):
    # test the performance of the feat importance functions on artificial data
    # Nr noise features = n_featurs-n_informative-n_redundant
    trnsX,cont=getTestData(n_features,n_informative,n_redundant,n_samples)
    dict0={'minWLeaf':[0.],'scoring':['accuracy'],'method':['MDI','MDA','SFI'],
           'max_samples':[1.]}
    jobs,out=(dict(zip(dict0,i))for i in product(*dict0.values())),[]
    kargs={'pathOut':PurePath(pdir/'testFunc').as_posix(),
           'n_estimators':n_estimators,'tag':'testFunc','cv':cv}
    for job in jobs:
        job['simNum']=job['method']+'_'+job['scoring']+'_'+'%.2f'%job['minWLeaf']+\
        '_'+str(job['max_samples'])
        print(job['simNum'])
        kargs.update(job)
        imp,oob,oos=featImportance(trnsX=trnsX,cont=cont,**kargs)
        plotFeatImportance(imp=imp,oob=oob,oos=oos,**kargs)
        df0=imp[['mean']]/imp['mean'].abs().sum()
        df0['type']=[i[0] for i in df0.index]
        df0=df0.groupby('type')['mean'].abs().sum()
        df0.update({'oob':oob,'oos':oos});df0.update(job)
        out.append(df0)
    out=(pd.DataFrame(out).sort_values(['method','scoring','minWLeaf','max_samples']))
    out=out['method','scoring','minWLeaf','max_samples','I','R','N','oob','oos']
    out.to_csv(kargs['pathOut']+'stats.csv')
    return

#=======================================================
# 8.10 Feature Importance Plotting Function

def plotFeatImportance(pathOut,imp,oob,oos,method,tag=0,simNum=0,**kargs):
    # plot mean imp bars with std
    mpl.figure(figsize=(10,imp.shape[0]/5.))
    imp=imp.sort_values('mean',ascending=True)
    ax=imp['mean'].plot(kind='barh',color='b',alpha=0.25,xerr=imp['std'],
                        error_kw={'ecolor':'r'})
    if method=='MDI':
        mpl.xlim([0,imp.sum(axis=1).max()])
        mpl.axvline(1./imp.shape[0],lw=1.,color='r',ls='dotted')
    ax.get_yaxis().set_visible(False)
    for i,j in zip(ax.patches,imp.index):
        ax.text(i.get_width()/2, i.get_y()+i.get_height()/2,
                j,ha='center',va='center',color='k')
    mpl.title('tag='+tag+' | simNUm='+str(simNum)+' | oob='+str(round(oob,4))+' | oos='+str(round(oos,4)))
    mpl.savefig(pathOut+'featImportance_'+str(simNum)+'.png',dpi=100)
    mpl.clf();mpl.close()
    return


def getTEvents(gRaw, h):
    tEvents, sPos, sNeg = [], 0, 0
    diff = np.log(gRaw).diff().dropna()
    for i in tqdm(diff.index[1:]):
        try:
            pos, neg = float(sPos+diff.loc[i]), float(sNeg+diff.loc[i])
        except Exception as e:
            print(e)
            print(sPos+diff.loc[i], type(sPos+diff.loc[i]))
            print(sNeg+diff.loc[i], type(sNeg+diff.loc[i]))
            break
        sPos, sNeg=max(0., pos), min(0., neg)
        if sNeg<-h:
            sNeg=0;tEvents.append(i)
        elif sPos>h:
            sPos=0;tEvents.append(i)
    return pd.DatetimeIndex(tEvents)




# ### Daily Volatility Estimator [3.1]

# In[3]:
# =======================================================
# Daily Volatility Estimator [3.1]
## for wtvr reason dates are not aligned for return calculation
## must account for it for computation

# daily vol reindexed to close of each bar
def getDailyVol(close,span0=100):
    
    # substract one day from each datetime timestamp index field
    dftemp=close.index-pd.Timedelta(days=1) 
    #search for new dates in original timestamp index field. writes 0 if not found, row number if found
    df0=close.index.searchsorted(dftemp) 
    #remove entries that have 0, i.e. rows that didn't have a day earlier match
    df0=df0[df0>0]
    #create a new dataframe that has the original dates as index, and the new dates as second column
    df0=(pd.Series(close.index[df0-1],
                   index=close.index[close.shape[0]-df0.shape[0]:]))
    #bp()
    try:
        df0=close.loc[df0.index]/close.loc[df0.values].values-1 # calculate daily returns for each bar
    except Exception as e:
        print(e)
        print('adjusting shape of close.loc[df0.index]')
        cut = close.loc[df0.index].shape[0] - close.loc[df0.values].shape[0]
        df0=close.loc[df0.index].iloc[:-cut]/close.loc[df0.values].values-1
        
    #calculate volatility for each bar based on std deviation of the daily return of that bar
    df0=df0.ewm(span=span0).std().rename('dailyVol')
    return df0
'''
def getDailyVol(close,span0=100):
    # daily vol reindexed to close
    df0=close.index.searchsorted(close.index-pd.Timedelta(days=1))
    df0=df0[df0>0]   
    df0=(pd.Series(close.index[df0-1], 
                   index=close.index[close.shape[0]-df0.shape[0]:]))   
    
       
    try:
        print('\n df0 index\n', df0.index)
        print('\n df0 values\n', df0.values)
        dfidxs=close.loc[df0.index].drop_duplicates()
        dfvals=close.loc[df0.values]#.iloc[:dfidxs.size] #.drop_duplicates()
        print('\n close.loc[df0.index]\n', dfidxs)
        print('\n close.loc[df0.values]\n', dfvals)
        #df1=(dfidxs.values/dfvals.values) -1 # daily rets
        df0=(dfidxs/dfvals.values) -1 # daily rets
        print('\n df1\n',df1)
    except Exception as e:
        print(e)
        print('adjusting shape of close.loc[df0.index]')
        cut = close.loc[df0.index].shape[0] - close.loc[df0.values].shape[0]
        df0=close.loc[df0.index].iloc[:-cut]/close.loc[df0.values].values-1
        
        #df4=pd.Series(df1).ewm(span=span0).std().dropna()
        #df0=pd.Series(df1).ewm(span=span0).std().dropna()
        df0=(df0).ewm(span=span0).std().dropna()
    return df0

'''
# ### Triple-Barrier Labeling Method [3.2]

# In[4]:


def applyPtSlOnT1(close,events,ptSl,molecule):
    # apply stop loss/profit taking, if it takes place before t1 (end of event)
    events_=events.loc[molecule]
    out=events_[['t1']].copy(deep=True)
    if ptSl[0]>0: pt=ptSl[0]*events_['trgt']
    else: pt=pd.Series(index=events.index) # NaNs
    if ptSl[1]>0: sl=-ptSl[1]*events_['trgt']
    else: sl=pd.Series(index=events.index) # NaNs
    for loc,t1 in events_['t1'].fillna(close.index[-1]).iteritems():
        df0=close[loc:t1] # path prices
        df0=(df0/close[loc]-1)*events_.at[loc,'side'] # path returns
        out.loc[loc,'sl']=df0[df0<sl[loc]].index.min() # earliest stop loss
        out.loc[loc,'pt']=df0[df0>pt[loc]].index.min() # earliest profit taking
    return out


# ### Gettting Time of First Touch (getEvents) [3.3], [3.6]

# In[5]:


def getEvents(close, tEvents, ptSl, trgt, minRet, numThreads, t1=False, side=None):
    #1) get target
    trgt=trgt.loc[tEvents]
    trgt=trgt[trgt>minRet] # minRet
    #2) get t1 (max holding period)
    if t1 is False:t1=pd.Series(pd.NaT, index=tEvents)
    #3) form events object, apply stop loss on t1
    if side is None:side_,ptSl_=pd.Series(1.,index=trgt.index), [ptSl[0],ptSl[0]]
    else: side_,ptSl_=side.loc[trgt.index],ptSl[:2]
    events=(pd.concat({'t1':t1,'trgt':trgt,'side':side_}, axis=1)
            .dropna(subset=['trgt']))
    df0=mpPandasObj(func=applyPtSlOnT1,pdObj=('molecule',events.index),
                    numThreads=numThreads,close=close,events=events,
                    ptSl=ptSl_)
    events['t1']=df0.dropna(how='all').min(axis=1) # pd.min ignores nan
    if side is None:events=events.drop('side',axis=1)
    return events


# ### Adding Vertical Barrier [3.4]

# In[6]:


def addVerticalBarrier(tEvents, close, numDays=1):
    t1=close.index.searchsorted(tEvents+pd.Timedelta(days=numDays))
    t1=t1[t1<close.shape[0]]
    t1=(pd.Series(close.index[t1],index=tEvents[:t1.shape[0]]))
    return t1


# ### Labeling for side and size [3.5]

# In[7]:


def getBinsOld(events,close):
    #1) prices aligned with events
    events_=events.dropna(subset=['t1'])
    px=events_.index.union(events_['t1'].values).drop_duplicates()
    px=close.reindex(px,method='bfill')
    #2) create out object
    out=pd.DataFrame(index=events_.index)
    out['ret']=px.loc[events_['t1'].values].values/px.loc[events_.index]-1
    out['bin']=np.sign(out['ret'])
    # where out index and t1 (vertical barrier) intersect label 0
    try:
        locs = out.query('index in @t1').index
        out.loc[locs, 'bin'] = 0
    except:
        pass
    return out


# ### Expanding getBins to Incorporate Meta-Labeling [3.7]

# In[8]:


def getBins(events, close):
    '''
    Compute event's outcome (including side information, if provided).
    events is a DataFrame where:
    -events.index is event's starttime
    -events['t1'] is event's endtime
    -events['trgt'] is event's target
    -events['side'] (optional) implies the algo's position side
    Case 1: ('side' not in events): bin in (-1,1) <-label by price action
    Case 2: ('side' in events): bin in (0,1) <-label by pnl (meta-labeling)
    '''
    #1) prices aligned with events
    events_=events.dropna(subset=['t1'])
    px=events_.index.union(events_['t1'].values).drop_duplicates()
    px=close.reindex(px,method='bfill')
    #2) create out object
    out=pd.DataFrame(index=events_.index)
    out['ret']=px.loc[events_['t1'].values].values/px.loc[events_.index]-1
    if 'side' in events_:out['ret']*=events_['side'] # meta-labeling
    out['bin']=np.sign(out['ret'])
    if 'side' in events_:out.loc[out['ret']<=0,'bin']=0 # meta-labeling
    return out


# ### Dropping Unnecessary Labels [3.8]

# In[9]:


def dropLabels(events, minPct=.05):
    # apply weights, drop labels with insufficient examples
    while True:
        df0=events['bin'].value_counts(normalize=True)
        if df0.min()>minPct or df0.shape[0]<3:break
        print('dropped label: ', df0.argmin(),df0.min())
        events=events[events['bin']!=df0.argmin()]
    return events


# ### Linear Partitions [20.4.1]

# In[10]:


def linParts(numAtoms,numThreads):
    # partition of atoms with a single loop
    parts=np.linspace(0,numAtoms,min(numThreads,numAtoms)+1)
    parts=np.ceil(parts).astype(int)
    return parts


# In[11]:


def nestedParts(numAtoms,numThreads,upperTriang=False):
    # partition of atoms with an inner loop
    parts,numThreads_=[0],min(numThreads,numAtoms)
    for num in range(numThreads_):
        part=1+4*(parts[-1]**2+parts[-1]+numAtoms*(numAtoms+1.)/numThreads_)
        part=(-1+part**.5)/2.
        parts.append(part)
    parts=np.round(parts).astype(int)
    if upperTriang: # the first rows are heaviest
        parts=np.cumsum(np.diff(parts)[::-1])
        parts=np.append(np.array([0]),parts)
    return parts


# ### multiprocessing snippet [20.7]

# In[12]:


def mpPandasObj(func,pdObj,numThreads=24,mpBatches=1,linMols=True,**kargs):
    '''
    Parallelize jobs, return a dataframe or series
    + func: function to be parallelized. Returns a DataFrame
    + pdObj[0]: Name of argument used to pass the molecule
    + pdObj[1]: List of atoms that will be grouped into molecules
    + kwds: any other argument needed by func
    
    Example: df1=mpPandasObj(func,('molecule',df0.index),24,**kwds)
    '''
    import pandas as pd
    #if linMols:parts=linParts(len(argList[1]),numThreads*mpBatches)
    #else:parts=nestedParts(len(argList[1]),numThreads*mpBatches)
    if linMols:parts=linParts(len(pdObj[1]),numThreads*mpBatches)
    else:parts=nestedParts(len(pdObj[1]),numThreads*mpBatches)
    
    jobs=[]
    for i in range(1,len(parts)):
        job={pdObj[0]:pdObj[1][parts[i-1]:parts[i]],'func':func}
        job.update(kargs)
        jobs.append(job)
    if numThreads==1:out=processJobs_(jobs)
    else: out=processJobs(jobs,numThreads=numThreads)
    if isinstance(out[0],pd.DataFrame):df0=pd.DataFrame()
    elif isinstance(out[0],pd.Series):df0=pd.Series()
    else:return out
    for i in out:df0=df0.append(i)
    df0=df0.sort_index()
    return df0


# ### single-thread execution for debugging [20.8]

# In[13]:


def processJobs_(jobs):
    # Run jobs sequentially, for debugging
    out=[]
    for job in jobs:
        out_=expandCall(job)
        out.append(out_)
    return out


# ### Example of async call to multiprocessing lib [20.9]

# In[14]:


import multiprocessing as mp
import datetime as dt

#________________________________
def reportProgress(jobNum,numJobs,time0,task):
    # Report progress as asynch jobs are completed
    msg=[float(jobNum)/numJobs, (time.time()-time0)/60.]
    msg.append(msg[1]*(1/msg[0]-1))
    timeStamp=str(dt.datetime.fromtimestamp(time.time()))
    msg=timeStamp+' '+str(round(msg[0]*100,2))+'% '+task+' done after '+ str(round(msg[1],2))+' minutes. Remaining '+str(round(msg[2],2))+' minutes.'
    if jobNum<numJobs:sys.stderr.write(msg+'\r')
    else:sys.stderr.write(msg+'\n')
    return
#________________________________
def processJobs(jobs,task=None,numThreads=24):
    # Run in parallel.
    # jobs must contain a 'func' callback, for expandCall
    if task is None:task=jobs[0]['func'].__name__
    pool=mp.Pool(processes=numThreads)
    outputs,out,time0=pool.imap_unordered(expandCall,jobs),[],time.time()
    # Process asyn output, report progress
    for i,out_ in enumerate(outputs,1):
        out.append(out_)
        reportProgress(i,len(jobs),time0,task)
    pool.close();pool.join() # this is needed to prevent memory leaks
    return out


# ### Unwrapping the Callback [20.10]

# In[15]:


def expandCall(kargs):
    # Expand the arguments of a callback function, kargs['func']
    func=kargs['func']
    del kargs['func']
    out=func(**kargs)
    return out


# ### Pickle Unpickling Objects [20.11]

# In[16]:


def _pickle_method(method):
    func_name=method.im_func.__name__
    obj=method.im_self
    cls=method.im_class
    return _unpickle_method, (func_name,obj,cls)
#________________________________
def _unpickle_method(func_name,obj,cls):
    for cls in cls.mro():
        try:func=cls.__dict__[func_name]
        except KeyError:pass
        else:break
    return func.__get__(obj,cls)
#________________________________
import copyreg,types, multiprocessing as mp
copyreg.pickle(types.MethodType,_pickle_method,_unpickle_method)


# # Exercises

# ## Import Dataset
# 
# Note this dataset below has been resampled to `1s` and then `NaNs` removed. This was done to remove any duplicate indices not accounted for in a simple call to `pd.DataFrame.drop_duplicates()`. 

# In[17]:

path = os.getcwd()
path
df = pd.read_csv(path+'/bitstampUSD_21.csv', index_col=0)
print(df)

# ## [3.1] Form Dollar Bars

# In[18]:


dbars = dollar_bar_df(df, 'dv', 500_000).drop_duplicates().dropna()
print(dbars)
# run: pip install https://github.com/matplotlib/mpl_finance/archive/master.zip
from mpl_finance import candlestick_ochl as candlestick
from mpl_finance import volume_overlay3
from matplotlib.dates import num2date
from matplotlib.dates import date2num
import matplotlib.mlab as mlab
import datetime

datafile = path+"/dollar_bars.csv"
r = mlab.csv2rec(datafile, delimiter=',')
print(r)
# ### (a) Run cusum filter with threshold equal to std dev of daily returns

# In[19]:

dbars.index=pd.to_datetime(dbars.index)
close = dbars.price.copy()
close.index=pd.to_datetime(close.index)
dailyVol = getDailyVol(close)
print(dailyVol)


# In[20]:


f,ax=plt.subplots()
dailyVol.plot(ax=ax)
ax.axhline(dailyVol.mean(),ls='--',color=red)


# In[21]:


tEvents = getTEvents(close,h=dailyVol.mean())
print('\n tEvents \n',tEvents)


# ### (b) Add vertical barrier

# In[22]:


t1 = addVerticalBarrier(tEvents, close)
print('\n t1 \n',t1)
t1


# ### (c) Apply triple-barrier method where `ptSl = [1,1]` and `t1` is the series created in `1.b`

# In[23]:


# create target series
ptsl = [1,1]
target=dailyVol
# select minRet
minRet = 0.01

# Run in single-threaded mode on Windows
import platform
if platform.system() == "Windows":
    cpus = 1
else:
    cpus = cpu_count() - 1
    
events = getEvents(close,tEvents,ptsl,target,minRet,cpus,t1=t1)


# In[24]:


cprint(events)


# ### (d) Apply `getBins` to generate labels

# In[25]:


labels = getBins(events, close)
cprint(labels)


# In[26]:


labels.bin.value_counts()


# ## [3.2] Use snippet 3.8 to drop under-populated labels

# In[27]:


clean_labels = dropLabels(labels)
cprint(clean_labels)


# In[28]:


clean_labels.bin.value_counts()


# ## [3.4] Develop moving average crossover strategy. For each obs. the model suggests a side but not size of the bet

# In[29]:


fast_window = 3
slow_window = 7

close_df = (pd.DataFrame()
            .assign(price=close)
            .assign(fast=close.ewm(fast_window).mean())
            .assign(slow=close.ewm(slow_window).mean()))
cprint(close_df)


# In[30]:


def get_up_cross(df):
    crit1 = df.fast.shift(1) < df.slow.shift(1)
    crit2 = df.fast > df.slow
    return df.fast[(crit1) & (crit2)]

def get_down_cross(df):
    crit1 = df.fast.shift(1) > df.slow.shift(1)
    crit2 = df.fast < df.slow
    return df.fast[(crit1) & (crit2)]

up = get_up_cross(close_df)
down = get_down_cross(close_df)

f, ax = plt.subplots(figsize=(11,8))

close_df.loc['2014':].plot(ax=ax, alpha=.5)
up.loc['2014':].plot(ax=ax,ls='',marker='^', markersize=7,
                     alpha=0.75, label='upcross', color='g')
down.loc['2014':].plot(ax=ax,ls='',marker='v', markersize=7, 
                       alpha=0.75, label='downcross', color='r')

ax.legend()


# ### (a) Derive meta-labels for `ptSl = [1,2]` and `t1` where `numdays=1`. Use as `trgt` dailyVol computed by snippet 3.1 (get events with sides)

# In[31]:


side_up = pd.Series(1, index=up.index)
side_down = pd.Series(-1, index=down.index)
side = pd.concat([side_up,side_down]).sort_index()
cprint(side)


# In[32]:


minRet = .01 
ptsl=[1,2]
ma_events = getEvents(close,tEvents,ptsl,target,minRet,cpus,t1=t1,side=side)
cprint(ma_events)


# In[33]:


ma_events.side.value_counts()


# In[34]:


ma_side = ma_events.dropna().side


# In[35]:


ma_bins = getBins(ma_events,close).dropna()
cprint(ma_bins)


# In[36]:


Xx = pd.merge_asof(ma_bins, side.to_frame().rename(columns={0:'side'}),
                   left_index=True, right_index=True, direction='forward')
cprint(Xx)


# ### (b) Train Random Forest to decide whether to trade or not `{0,1}` since underlying model (crossing m.a.) has decided the side, `{-1,1}`

# In[37]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, classification_report


# In[38]:


X = ma_side.values.reshape(-1,1)
#X = Xx.side.values.reshape(-1,1)
y = ma_bins.bin.values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)

n_estimator = 10000
rf = RandomForestClassifier(max_depth=2, n_estimators=n_estimator,
                            criterion='entropy', random_state=RANDOM_STATE)
rf.fit(X_train, y_train)

# The random forest model by itself
y_pred_rf = rf.predict_proba(X_test)[:, 1]
y_pred = rf.predict(X_test)
fpr_rf, tpr_rf, _ = roc_curve(y_test, y_pred_rf)
print(classification_report(y_test, y_pred))

plt.figure(1)
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr_rf, tpr_rf, label='RF')
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve')
plt.legend(loc='best')
plt.show()


# ## [3.5] Develop mean-reverting Bollinger Band Strategy. For each obs. model suggests a side but not size of the bet.

# In[39]:


def bbands(price, window=None, width=None, numsd=None):
    """ returns average, upper band, and lower band"""
    ave = price.rolling(window).mean()
    sd = price.rolling(window).std(ddof=0)
    if width:
        upband = ave * (1+width)
        dnband = ave * (1-width)
        return price, np.round(ave,3), np.round(upband,3), np.round(dnband,3)        
    if numsd:
        upband = ave + (sd*numsd)
        dnband = ave - (sd*numsd)
        return price, np.round(ave,3), np.round(upband,3), np.round(dnband,3)


# In[40]:


window=50
bb_df = pd.DataFrame()
bb_df['price'],bb_df['ave'],bb_df['upper'],bb_df['lower']=bbands(close, window=window, numsd=1)
bb_df.dropna(inplace=True)
cprint(bb_df)


# In[41]:


f,ax=plt.subplots(figsize=(11,8))
bb_df.loc['2014'].plot(ax=ax)


# In[42]:


def get_up_cross(df, col):
    # col is price column
    crit1 = df[col].shift(1) < df.upper.shift(1)  
    crit2 = df[col] > df.upper
    return df[col][(crit1) & (crit2)]

def get_down_cross(df, col):
    # col is price column    
    crit1 = df[col].shift(1) > df.lower.shift(1) 
    crit2 = df[col] < df.lower
    return df[col][(crit1) & (crit2)]

bb_down = get_down_cross(bb_df, 'price')
bb_up = get_up_cross(bb_df, 'price') 

f, ax = plt.subplots(figsize=(11,8))

bb_df.loc['2014':].plot(ax=ax, alpha=.5)
bb_up.loc['2014':].plot(ax=ax, ls='', marker='^', markersize=7,
                        alpha=0.75, label='upcross', color='g')
bb_down.loc['2014':].plot(ax=ax, ls='', marker='v', markersize=7, 
                          alpha=0.75, label='downcross', color='r')
ax.legend()


# ### (a) Derive meta-labels for `ptSl=[0,2]` and `t1` where `numdays=1`. Use as `trgt` dailyVol.

# In[43]:


bb_side_up = pd.Series(-1, index=bb_up.index) # sell on up cross for mean reversion
bb_side_down = pd.Series(1, index=bb_down.index) # buy on down cross for mean reversion
bb_side_raw = pd.concat([bb_side_up,bb_side_down]).sort_index()
cprint(bb_side_raw)

minRet = .01 
ptsl=[0,2]
bb_events = getEvents(close,tEvents,ptsl,target,minRet,cpus,t1=t1,side=bb_side_raw)
cprint(bb_events)

bb_side = bb_events.dropna().side
cprint(bb_side)


# In[44]:


bb_side.value_counts()


# In[45]:


bb_bins = getBins(bb_events,close).dropna()
cprint(bb_bins)


# In[46]:


bb_bins.bin.value_counts()


# ### (b) train random forest to decide to trade or not. Use features: volatility, serial correlation, and the crossing moving averages from exercise 2.

# In[47]:


def returns(s):
    arr = np.diff(np.log(s))
    return (pd.Series(arr, index=s.index[1:]))

def df_rolling_autocorr(df, window, lag=1):
    """Compute rolling column-wise autocorrelation for a DataFrame."""

    return (df.rolling(window=window)
            .corr(df.shift(lag))) # could .dropna() here

#df_rolling_autocorr(d1, window=21).dropna().head()


# In[48]:


srl_corr = df_rolling_autocorr(returns(close), window=window).rename('srl_corr')
cprint(srl_corr)


# In[49]:


features = (pd.DataFrame()
            .assign(vol=bb_events.trgt)
            .assign(ma_side=ma_side)
            .assign(srl_corr=srl_corr)
            .drop_duplicates()
            .dropna())
cprint(features)


# In[50]:


Xy = (pd.merge_asof(features, bb_bins[['bin']], 
                    left_index=True, right_index=True, 
                    direction='forward').dropna())
cprint(Xy)


# In[51]:


Xy.bin.value_counts()


# In[52]:


X = Xy.drop('bin',axis=1).values
y = Xy['bin'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, shuffle=False)

n_estimator = 10000
rf = RandomForestClassifier(max_depth=2, n_estimators=n_estimator,
                            criterion='entropy', random_state=RANDOM_STATE)
rf.fit(X_train, y_train)

# The random forest model by itself
y_pred_rf = rf.predict_proba(X_test)[:, 1]
y_pred = rf.predict(X_test)
fpr_rf, tpr_rf, _ = roc_curve(y_test, y_pred_rf)
print(classification_report(y_test, y_pred, target_names=['no_trade','trade']))

plt.figure(1)
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr_rf, tpr_rf, label='RF')
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve')
plt.legend(loc='best')
plt.show()


# ### (c) What is accuracy of predictions from primary model if the secondary model does not filter bets? What is classification report?

# In[53]:


minRet = .01 
ptsl=[0,2]
bb_events = getEvents(close,tEvents,ptsl,target,minRet,cpus,t1=t1)
cprint(bb_events)

bb_bins = getBins(bb_events,close).dropna()
cprint(bb_bins)

features = (pd.DataFrame()
            .assign(vol=bb_events.trgt)
            .assign(ma_side=ma_side)
            .assign(srl_corr=srl_corr)
            .drop_duplicates()
            .dropna())
cprint(features)

Xy = (pd.merge_asof(features, bb_bins[['bin']], 
                    left_index=True, right_index=True, 
                    direction='forward').dropna())
cprint(Xy)

### run model ###
X = Xy.drop('bin',axis=1).values
y = Xy['bin'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, shuffle=False)

n_estimator = 10000
rf = RandomForestClassifier(max_depth=2, n_estimators=n_estimator,
                            criterion='entropy', random_state=RANDOM_STATE)
rf.fit(X_train, y_train)

# The random forest model by itself
y_pred_rf = rf.predict_proba(X_test)[:, 1]
y_pred = rf.predict(X_test)
fpr_rf, tpr_rf, _ = roc_curve(y_test, y_pred_rf)
print(classification_report(y_test, y_pred))

plt.figure(1)
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr_rf, tpr_rf, label='RF')
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve')
plt.legend(loc='best')
plt.show()


# In[ ]:




