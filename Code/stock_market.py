# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

#!/usr/bin/env python3 
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  3 15:09:43 2020

@author: b.schnater
"""

# Als ie vastloop: pip install yfinance --upgrade --no-cache-dir
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime as dt
import re
import yfinance as yf
from datetime import date
from pandas_datareader.data import DataReader

from scipy import stats
import statsmodels.api as sm

path = r'C:\Users\bassc\OneDrive\Bas\Beurs\Output' # Savepath

# Timeframes bepalen
start = "2019-01-01"
today = str(dt.today().date())
end = today


#series_code = 'DGS10'
#data_source = 'fred'
#start = date(1962, 1, 1)

#data = DataReader(series_code, data_source, start)

###################
# Data downloaden # 
###################
# AEX
aex = yf.download("^AEX", start=start, end=today)
aex.index = aex.index.tz_localize(None)
aex = aex['Adj Close']
#nasdaq = yf.download("^IXIC", start=start, end=today)
#nasdaq = nasdaq['Adj Close']

import pandas_datareader as pdr
vix = pdr.DataReader('VIXCLS', 'fred', start=start, end=end).dropna().squeeze()
lower, upper = 16.5, 19.5

# Each term inside parentheses is [False, True, ...]
# Both terms must be True element-wise for a trigger to occur
blue = (vix < upper) & (vix.shift() >= upper)
yellow = (vix < lower) & (vix.shift() >= lower)
green = (vix > upper) & (vix.shift() <= upper)
red = (vix > lower) & (vix.shift() <= lower)

mapping = {1: 'blue', 2: 'yellow', 3: 'green', 4: 'red'}

indicator = pd.Series(np.where(blue, 1., np.where(yellow, 2.,
                      np.where(green, 3., np.where(red, 4., np.nan)))),
                      index=vix.index).ffill().map(mapping).dropna()

vix = vix.reindex(indicator.index)
vix.plot(color='black')
plt.axhspan(ymin=0, ymax=20, color='green', alpha=0.2)
plt.axhspan(ymin=25, ymax=30, color='orange', alpha=0.2)
plt.axhspan(ymin=30, ymax=85, color='red', alpha=0.2)
plt.text(s='Graph by Bas Schnater', x=start, y=0)
plt.title('VIX index')
plt.ylabel('VIX')
plt.show()

plot = pd.concat([vix, aex], axis=1)

plot['VIXCLS'].plot()
plt.ylabel('VIX CLS')
plt.text(s='Graph by Bas Schnater', x=start, y=8)
plot['Adj Close'].plot(secondary_y=True)
plt.gca().invert_yaxis()
plt.ylabel('AEX-index (aflopend)')
plt.title('Relatie tussen VIX-index en AEX-index')
plt.show()

fig, axes = plt.subplots(nrows=2, sharex=True)
axes[0].plot(vix, color='black')
axes[0].set_title('VIX')
axes[1].plot(aex)
axes[1].set_title('AEX')
fig.autofmt_xdate()
plt.show()

df_corr = aex.corr(vix)
print("Correlatie tussen AEX en VIX: ", np.round(df_corr,3))

# AEX fondsen
aex_fondsen = ["URW.AS","INGA.AS","AD.AS","BESI.AS","PRX.AS","KPN.AS","ADYEN.AS","DSM.AS","WKL.AS","PHIA.AS","UNA.AS","GLPG.AS","ASM.AS","MT.AS","RAND.AS","ASML.AS","REN.AS","IMCD.AS","AKZA.AS","ASRNL.AS","NN.AS","AGN.AS","HEIA.AS"]
aex_fondsen = yf.download(aex_fondsen, start=start, end=today)
aex_fondsen = aex_fondsen['Adj Close']
aex_fondsen.index = aex_fondsen.index.tz_localize(None)

# Correlatie-matrix
df_corr = aex_fondsen.corr()
print(df_corr)
plt.figure(figsize=(12,10))
plt.title('Correlatie AEX-aandelen')
sns.heatmap(df_corr, cmap='RdYlGn', annot=True, cbar=False, fmt='.2g')
plt.savefig('AEX_Correlatie-matrix.png', format='png', dpi=100, bbox_inches="tight")
plt.show()

# Correlatie-matrix chips
corr_chips = df_corr[['ASM.AS','ASML.AS','BESI.AS']]
plt.figure(figsize=(3,12))
plt.title('Correlatie chip-aandelen')
sns.heatmap(corr_chips, cmap='RdYlGn', annot=True, cbar=False, fmt='.2g')
plt.savefig('AEX_Chips-Correlatie-matrix.png', format='png', dpi=100, bbox_inches="tight")
plt.show()

# Bitcoin
bitcoin = yf.download("BTC-USD", start=start, end=today)
#bitcoin = bitcoin.div(bitcoin.iloc[0]).mul(100)
bitcoin = bitcoin['Adj Close']
bitcoin.plot(title='Koers Bitcoin')
plt.show()

### ETF Trackers ###
etfs = [
       "ECAR.MI","IQQH.DE",'QDVE.DE','TRET.MI','VWRL.AS', "SMH.MI", "REMX.MI"
#        'REMX.MI'#,
#       'SMH.MI'#,
#       'HTMW.DE'
#       ,'HDRO.MI'
       ]
etf_list = []
start = '2019-07-01'
for etf in etfs:
    etf = yf.download(etf, start=start, end=today)
    etf.index = etf.index.tz_localize(None)
    etf = etf['Adj Close'] # Volume is wel interessant
    etf_list.append(etf)
etf_list = pd.concat(etf_list, axis=1)
etf_list.columns = etfs

# Visualisatie ETFs
etf_list = etf_list.div(etf_list.iloc[1]).mul(100)
etf_list = etf_list.iloc[1:,:]

etf_list.bfill().rolling(14).mean().plot(title='ETFs vanaf ' + str(etf_list.index.min()), figsize=(15,10))
plt.legend(bbox_to_anchor=(1,1))
plt.xlim(start,today)
plt.savefig('ETF-koersen.png', format='png', dpi=100, bbox_inches="tight")
plt.show()

# Correlatie ETF's
fig, ax = plt.subplots(figsize=(10,7))
sns.heatmap(etf_list.corr(), annot=True, cmap='RdYlGn', fmt='.2g')
plt.savefig('ETF-correlaties.png', format='png', dpi=100, bbox_inches="tight")
plt.show()

######################
# Data visualisaties # 
######################

sns.set_style('darkgrid')
sns.set_context('paper')
sns.set_palette('Dark2')
boxspecs = dict(boxstyle='round', facecolor='white', alpha=0.3)

#######################
### BUBBEL-ANALYSES ###
#######################

# AEX fondsen
aex_fondsen = ["BESI.AS","URW.AS","INGA.AS","AD.AS","PRX.AS","KPN.AS","ADYEN.AS","DSM.AS","WKL.AS","PHIA.AS","UNA.AS","GLPG.AS","ASM.AS","MT.AS","RAND.AS","ASML.AS","REN.AS","IMCD.AS","AKZA.AS","ASRNL.AS","NN.AS","AGN.AS","HEIA.AS"]
amx_fondsen = ['AALB.AS', 'AF.PA','ALFEN.AS',"APAM.AS","ARCAD.AS","BAMNB.AS","BFIT.AS","CRBN.AS","ECMPA.AS","FAGR.BR","FLOW.AS","FUR.AS","JDEP.AS","NSI.AS", "OCI.AS", "PHARM.AS","PNL.AS","SBMO.AS","LIGHT.AS","ABN.AS","TWEKA.AS","VPK.AS","WDP.BR"]
ascx_fondsen = ['AXS.AS','AJAX.AS','ACOMO.AS','AVTX.AS','BSGR.AS','BAMNB.AS','CMCOM.AS','FFARM.AS','HEIJM.AS','KENDR.AS','BOLS.AS','NEDAP.AS','NSI.AS','ORDI.AS','SIFG.AS','SLIGR.AS','TOM2.AS','VASTN.AS','WHA.AS']
nl_aandelen = aex_fondsen + amx_fondsen + ascx_fondsen

damrak = yf.download(nl_aandelen, start=start, end=today)
damrak = damrak['Adj Close']
damrak.index = pd.to_datetime(damrak.index.astype(str).str.slice(start=0, stop=11))

damrak_lijst = []
for aandeel in nl_aandelen:
    stock = pd.DataFrame(damrak[aandeel])

    ticker = yf.Ticker(aandeel) # or pdr.get_data_yahoo(... 
    #from pandas_datareader import data as pdr
    #import yfinance as yf
    #yf.pdr_override() # <== that's all it takes :-)
    stock = ticker.history(period='max')
    stock = stock[['Close','Dividends']].loc[start:end]
    stock.index = pd.to_datetime(stock.index.astype(str).str.slice(start=0, stop=11))
    stock['ticker'] = ticker.ticker
    stock['name'] = ticker.info['shortName']
    stock['sector'] = ticker.info['sector']
    stock['industry'] = ticker.info['industry']
    stock['exchange'] = ticker.fast_info.exchange
    
    stock['P/E ratio'] = ticker.info['forwardPE']
    stock['52WeekChange'] = ticker.info['52WeekChange']
    stock['shares'] = ticker.fast_info['shares']
    stock['revenuePerShare'] = ticker.info['revenuePerShare']
    stock['forwardEps'] = ticker.info['forwardEps']
    
    # Analyst opinions
    analists = ticker.analyst_price_target
    stock['currentPrice'] = analists.iloc[1,0]
    stock['targetMeanPrice'] = analists.iloc[2,0]
    stock['avggrowth'] = stock['targetMeanPrice']/stock['currentPrice']
    stock['targetHighPrice'] = analists.iloc[3,0]
    stock['maxgrowth'] = stock['targetHighPrice']/stock['currentPrice']
    
    # Technische analyse
    stock['20dSTD'] = stock['Close'].rolling(window=20).std()
    stock['MA20'] = stock['Close'].rolling(window=20).mean()
    stock['MA200'] = stock['Close'].rolling(window=200).mean()
    stock['MA365'] = stock['Close'].rolling(window=365).mean()
    stock['Upper'] = stock['MA20'] + (stock['20dSTD'] * 2)
    stock['Lower'] = stock['MA20'] - (stock['20dSTD'] * 2)
    # advies
    #stock['advies'] = np.where(stock['Close'] < stock['MA200'], 'kooptip', 'neutraal')
    #stock['advies'] = np.where((stock['Close'] > stock['MA200']) & (stock['Close'] < stock['20dSTD']), 'hoog, niet kopen', stock['advies'])
    #stock['advies'] = np.where((stock['Close'] > stock['MA200']) & (stock['Close'] > stock['20dSTD']), 'uitgebroken, verkopen', stock['advies'])
    
    # Plots
    stock['Close'].plot(label='Dagkoers', color='black')
    stock['Upper'].plot(label='Upper', color='red',linestyle='-', alpha=0.3)
    stock['Lower'].plot(label='Lower', color='red',linestyle='-', alpha=0.3)
    stock['MA365'].plot(label='MA365', color='orange',linestyle='dotted')
    stock['MA200'].plot(label='MA200', color='blue', linestyle='dotted')
    plt.title(stock['name'].iloc[-1])
    plt.xlim(start,today)
    plt.legend(bbox_to_anchor=(1,1))
    plt.text(x=stock.index.min(), y=stock['Close'].max(), s=str('Sector: ' + stock['industry'].iloc[0]), bbox=dict(facecolor='white', edgecolor='none'))
    plt.text(x=stock.index.min(), y=stock['Close'].max()/1.1, s=str('Groei gem.: ' + str(round(stock['avggrowth'].iloc[0],2))), bbox=dict(facecolor='white', edgecolor='none'))
    plt.text(x=stock.index.min(), y=stock['Close'].max()/1.2, s=str('Groei max.: ') + str(round(stock['maxgrowth'].iloc[0],2)), bbox=dict(facecolor='white', edgecolor='none'))
    plt.text(x=stock.index.min(), y=stock['Close'].max()/1.3, s=str('WPA: ') + str(stock['revenuePerShare'].iloc[0]), bbox=dict(facecolor='white', edgecolor='none'))
    plt.text(x=stock.index.min(), y=stock['Close'].max()/1.4, s=str('P/E ratio: ') + str(stock['P/E ratio'].iloc[0]), bbox=dict(facecolor='white', edgecolor='red'))    
    #    plt.text(start,float(stock.min()), 'Source: Yahoo Finance. Graph by Bas Schnater (@BasSchnater)')
    plt.savefig(r"C:\Users\bassc\OneDrive\Bas\Beurs\Output" + "_" + str(stock['name'].iloc[-1]) + '_bollinger_bands.png', format='png', dpi=100, bbox_inches="tight")
#    plt.savefig(path + redactie + '_productie.png', format='png', dpi=100, bbox_inches="tight")
    plt.show()





# Huidige portfolio aandelen en ETF's
start = '2015-01-01'
aandelen = etfs # ['ALFEN.AS','MT.AS','ASM.AS','ASML.AS','AVTX.AS','BESI.AS','DSM.AS','RAND.AS','CVLC.F','WKL.AS']
#aandelen = ['PICK', 'REMX', 'DBB','XME']
#aandelen = aex
for aandeel in aandelen:
    bubbel = yf.download(aandeel, start=start, end=today) 
    bubbel = bubbel['Adj Close']
    bubbel.plot(label='Dagkoers')
    bubbel.rolling('200D', closed='both').mean().div(1.3).plot(label='-30% 200D', color='black',linestyle='dotted')
    bubbel.rolling('200D', closed='both').mean().plot(label='200-dagen-gemiddelde')
    bubbel.rolling('200D').mean().mul(1.3).plot(label='Bovengrens 30% 200D', color='black',linestyle='dotted')
    bubbel.rolling('200D').mean().mul(2).plot(label='Bijzondere markt 100% 200D', color='red',linestyle='dotted')
    plt.legend()
    plt.title(aandeel + ' koersontwikkeling')
    plt.savefig(aandeel + '_bubbel.png', format='png', dpi=100)
    plt.show()


# Testen van bubbels | Moving Averages
bubbel_aandeel = 'SMH.MI' #ticker
start = '2020-01-01'
bubbel  = yf.download(bubbel_aandeel, start=start, end=today) 
bubbel = bubbel['Adj Close']
bubbel.plot(label='Dagkoers')
bubbel.rolling('200D', closed='both').mean().div(1.3).plot(label='-30% 200D', color='black',linestyle='dotted')
bubbel.rolling('200D', closed='both').mean().plot(label='200-dagen-gemiddelde')
bubbel.rolling('200D').mean().mul(1.3).plot(label='Bovengrens 30% 200D', color='black',linestyle='dotted')
bubbel.rolling('200D').mean().mul(2).plot(label='Bijzondere markt 100% 200D', color='red',linestyle='dotted')
plt.legend()
plt.title(bubbel_aandeel + ' koersontwikkeling')
plt.savefig(bubbel_aandeel + '_bubbel.png', format='png', dpi=100, bbox_inches="tight")
plt.show()


#=============== INDIVIDUELE AANDELEN BEOORDELEN ================#
import pandas as pd
import numpy as np

# Pyfolio documentatie: https://nbviewer.org/format/slides/github/quantopian/pyfolio/blob/master/pyfolio/examples/pyfolio_talk_slides.ipynb#/
# Pyfolio index error: https://github.com/quantopian/pyfolio/issues/661
portfolio = aex_namen


    start = start
    stock_rets = yf.download('ASM.AS', start='2019-01-01', end=today)
    pyf = pyf['Adj Close']

    ### Portfolio analysis using Pyfolio ### <== Iets werkt niet in het package
    import pyfolio as pf
    # -> data moet Pandas Series zijn met datetimeindex
    pyf_returns = pyf.pct_change().dropna()
    pyf_returns.index=pd.to_datetime(pyf_returns.index)
    pyf_returns = pyf_returns.dropna()

    # Ensure the returns are a series
#    pyf_returns=pyf_returns['Adj Close']
    print(type(pyf_returns))

    # Create the returns tear sheet    
    pf.create_returns_tear_sheet(pyf_returns, return_fig=True)
#    fig.suptitle(aandeel + ' vanaf ' + start)
    plt.savefig(aandeel + '_tearsheet.png', format='png', dpi=100, bbox_inches="tight")



#### Quantstats Portfolio Analytics ####
# Documentation: https://pypi.org/project/QuantStats/

# -- Setup
import quantstats as qs
qs.extend_pandas()

# fetch the daily returns for a stock
stock = pd.Series(damrak['ASM.AS'])

### Stats
[f for f in dir(qs.stats) if f[0] != '_'] # <-- alle beschikbare statistieken
qs.reports.metrics(mode='full', returns=stock)
qs.stats.sharpe(stock)
qs.stats.smart_sharpe(stock)

### Plots
qs.plots.snapshot(stock, title='Stock Performance')
qs.plots.to_plotly() # <-- werkt nog niet

### Reports
qs.reports.full(stock)
qs.reports.html(stock, "AEX") # <-- werkt nog niet


###### Pyfolio portfolio analytics ######
# Meer voorbeelden: https://nbviewer.org/github/quantopian/pyfolio/tree/master/pyfolio/examples/
# Hulp bij interpretaties: https://www.fmz.com/digest-topic/5798
# Volledige lessen: https://www.quantrocket.com/codeload/quant-finance-lectures/quant_finance_lectures/Lecture33-Portfolio-Analysis-with-pyfolio.ipynb.html
# Uitlezen grafieken: https://towardsdatascience.com/the-easiest-way-to-evaluate-the-performance-of-trading-strategies-in-python-4959fd798bb3

# ERROR: oplossing bij nieuwe installatie https://github.com/quantopian/pyfolio/issues/652

# Pyfolio werkt vooral met returns, dus aantal voorwaarden:
# 1) Moet een series zijn
# 2) Moeten returns zijn (dus % change))

start = 
for aandeel in portfolio:    
    # Pyfolio stock
    start = start
    pyf = yf.download(aandeel, start=start, end=today)
    pyf = pyf['Adj Close']

    ### Portfolio analysis using Pyfolio ###
    import pyfolio as pf
    pyf_returns = stock.pct_change().dropna()
    pyf_returns.index=pd.to_datetime(pyf_returns.index)
    pyf_returns = pyf_returns.dropna()

    # Ensure the returns are a series
    #pyf_returns=pyf_returns['Adj Close']
    print(type(pyf_returns))

    # Create the returns tear sheet    
    pf.create_returns_tear_sheet(pyf_returns, return_fig=True)
#    fig.suptitle(aandeel + ' vanaf ' + start)
    plt.savefig(aandeel + '_tearsheet.png', format='png', dpi=100, bbox_inches="tight")
#    fig.show()

# Breakdown portfolio performance
#sect_map = {'COST':'Consumer Goods',
#            'INTC':'Technology',
#            'CERN':'Healthcare',
#            'GPS':'Technology',
#            'MMM':'Construction',
#            'DELL':'Technology',
#            'AMD':'Technology'}
#pf.create_position_tear_sheet(pyf, positions,sector_mappings=sect_map)
#display_tear_sheet()

"""Andere pyfolio toepassingen"""
# Voorbereiding
oos_date = pd.Timestamp('2020-07-01').tz_localize('CET')

aex_returns = aex_fondsen.pct_change()
aex_returns.index = pd.to_datetime(aex_returns.index)
aex_returns = aex_returns.iloc[1:,:]
aex_returns = aex_returns.drop(columns='UNA.AS')
aex_returns.index = aex_returns.index.tz_localize('CET')

# AEX rets voor benchmark
aex_rets = aex.pct_change().dropna()
#aex_rets.index = pd.to_datetime(aex_rets)
aex_rets.index = aex_rets.index.tz_localize('CET')
aex_rets.columns = ['AEX']

"""Sharpe ratios per aandeel"""
[f for f in dir(pf.plotting) if 'plot_' in f]
aex_sharpe = pd.DataFrame(pf.timeseries.sharpe_ratio(aex_returns))
# Berekent sharpe ratio per aandeel
aex_sharpe.index = aex_returns.columns
print(aex_sharpe.sort_values(by=0))

"""Grafieken"""
# Rolling returns => alles in groen
pf.plotting.plot_rolling_returns(aex_returns)
# werkt (nog) niet => pf.create_bayesian_tear_sheet(aex_returns['ASM.AS'], live_start_date=oos_date)

# Tear sheet per aandeel met weergave sinds start traden
pf.create_returns_tear_sheet(aex_returns['ASM.AS'], benchmark_rets=aex_rets, live_start_date=oos_date)


mei = aex_fondsen.loc['2020-03-10':'2020-05-20']
mei = mei.div(mei.iloc[0]).mul(100)
mei.plot()
plt.legend(bbox_to_anchor=(1,1))
plt.show()

print()
















######################
# Portfolio analysis #
######################
start = '2018-01-01'
#=================#### NIEUWE INVESTERING? ####====================#
# 1a) Data downloaden
test = yf.download("IDVY.AS", start=start, end=today) 
test_returns = test['Adj Close']
test_returns = test_returns.pct_change().dropna()

# 1b) indexed groei bekijken
test_indexed = test.div(test.iloc[0]).mul(100)
test_indexed.plot()
plt.show()

# 2) Volatiliteit bekijken
test_returns.plot(title='Volatiliteit')
plt.xlim('2020-07-01', today)
plt.show()

# 3) Spreiding van returns bekijken
test_returns.hist(bins=75, density='False') # Rechts van 0 is hoofdzakelijk groei
plt.axvline(x=0, color='black')
plt.show()

# 4) Kern statistieken bekijken
from scipy.stats import skew, kurtosis # Let op ivm ongelijke wegingen ook ongelijk antwoord
stats = []
stats.append(np.std(test_returns))
stats.append(skew(test_returns))
stats.append(kurtosis(test_returns)+3)
print(stats)

print("Mean : ", np.mean(test_returns))
print("Std/vol  : ", np.std(test_returns))
#print("Portfolio var  : ", pf_returns_mean.std()**2)
print("Skew : ", skew(test_returns)) # 0 is normaal. <1 / >1 is highly skewed. Negative skewness is bump to right of middle, which is good
print("Kurt : ", kurtosis(test_returns)+3) # Normal = 3. >3 is fat-tailed and implies undesirable kurtosis. Quite often in extreme ends in the distribution

### Tail analysis ###
# Calculate historical VaR(95)
var_95 = np.percentile(test_returns, 5)
var_90 = np.percentile(test_returns, 10)
print(var_95)
cvar_95 = test_returns[test_returns <= var_95].mean()
print(cvar_95)

# Sort the returns for plotting
sorted_rets = test_returns.sort_values()

# Plot the probability of each sorted return quantile
plt.hist(sorted_rets, bins=50)
plt.axvline(x=var_95, color='r', linestyle='-', label="VaR 95: {0:.3f}%".format(var_95))
plt.axvline(x=var_90, color='r', linestyle='dotted', label="VaR 90: {0:.3f}%".format(var_90))
plt.axvline(x=cvar_95, color='b', linestyle='-', label='CVaR 95: {0:.3f}%'.format(cvar_95))
plt.legend()
plt.show()
# Mocht je een algoritme-trader bouwen dan kun je hiermee de bandbreedtes uitrekenen

# MEer wetenschappelijk (want gem en std van returns)
# Import norm from scipy.stats
from scipy.stats import norm
# Estimate the average daily return
mu = np.mean(test_returns)
# Estimate the daily volatility
vol = np.std(test_returns)
# Set the VaR confidence level
confidence_level = 0.05
# Calculate Parametric VaR
var_95 = norm.ppf(confidence_level, mu, vol)
print('Mean: ', str(mu), '\nVolatility: ', str(vol), '\nVaR(95): ', str(var_95))

### Monte Carlo simulation ###
# Set the simulation parameters
mu = np.mean(test_returns)
vol = np.std(test_returns)
T = 252
S0 = 442 # Initial stock price
# Add one to the random returns
for i in range(50):
    rand_rets = np.random.normal(mu, vol, T) + 1
    # Forecasted random walk
    forecasted_values = S0*(rand_rets.cumprod())
    # Plot the random walk
    plt.plot(range(T), forecasted_values)
plt.show()

# Wat gebeurd er de dag na grote verliezen?
day_after = pd.DataFrame(test_returns)
for i in range(10):
    day_after[i] = test_returns.shift(-i)
day_neg = day_after[day_after['Adj Close'] < -0.035]
day_neg.index = day_neg.index.strftime("%d/%m/%Y")
day_neg_transposed = day_neg.T
day_neg_transposed = day_neg_transposed.iloc[1:-1,:]
#day_neg_transposed = day_neg_transposed.div(day_neg_transposed.iloc[0]).mul(100)
plt.plot(day_neg_transposed)
plt.legend()
plt.show()


import numpy as np
from scipy import stats
#from mat4py import loadmat
import statsmodels.api as sm
import pandas as pd
import matplotlib.pyplot as plt

## 1. Load data
dfX  = pd.DataFrame(test_returns) # in this file the time series data is labeled "x"

#x = loadmat('ts_data.mat')
#dfX =  pd.DataFrame({'x': np.array(x['x'][:])[:,0]})
x = dfX['Adj Close'].values
T = len(x)

dfX['xlag1'] = dfX['Adj Close'].shift(1)
dfX['xlag2'] = dfX['Adj Close'].shift(2)
dfX['xlag3'] = dfX['Adj Close'].shift(3)
dfX['xlag4'] = dfX['Adj Close'].shift(4)
dfX['xlag5'] = dfX['Adj Close'].shift(5)
dfX['const'] = 1

## 2. Run AR regression using sm.ols
y = dfX['Adj Close']
X = dfX[['const','xlag1']]
model = sm.OLS(y,X,missing='drop')
resultsAR = model.fit()                     # resultsARcontains all the estimation information

print(resultsAR.summary()) #prints detailed version of regression results
# Filteren P-values >0.05
b = resultsAR.params # save coefficients

## 3. Forward iterate AR(2) model to obtain forecast
h = 2 # define number of forecast periods
xforecast = np.zeros((T+h-1,1))  #define length of forecast vector (including observed data)
xforecast[0:T,] = dfX['Adj Close'].values.reshape(T,1);     #set first T periods equal to observed data

for t in range((T),(T+h-1)):  # start forward iteration loop
    xforecast[t,] = b[0] + b[1]*xforecast[(t-1),:]# + b[2]*xforecast[(t-2),:] # !!!!aanpassen aan aantal lags!!!!

## 4. Plot observed data and forecast
plt.plot(xforecast,'r')
plt.plot(x,'k')
#plt.title('Toekomstige huizenprijzen Amsterdam')
plt.show()

###### /Nieuwe investering?/ ######
#==========================================================================================================#

#
##
###
#### PORTFOLIO PERFORMANCE ####

# Returns dataset maken
pf_returns = pf.pct_change()
pf_returns = pf_returns.dropna()

# MEAN RETURNS
mean_dailyreturns = pf_returns.mean()
port_return = np.sum(mean_dailyreturns*pf_weights) # average daily return
# Daily returns over time
pf_returns['Portfolio'] = pf_returns.dot(pf_weights)
# Cumulative daily returns
daily_cum_return = (1+pf_returns).cumprod()
#PLot van daily cumulative returns
daily_cum_return['Portfolio'].plot(title='Portfolio ontwikkeling')

### MEASURING RISK OF A PORTFOLIO
# Maken van covariance matrix
numstocks = len(fondsen)
cov_matrix_annual = (pf_returns.iloc[:,0:numstocks].cov())*250
print(cov_matrix_annual)
sns.heatmap(cov_matrix_annual)

# Portfolio variance
port_variance = np.dot(pf_weights.T, np.dot(cov_matrix_annual, pf_weights))
print(port_variance)
print('Portfolio annual variance: ' + str(np.round(port_variance, 5))+'%')
#Portfolio volatility
post_stddev = np.sqrt(np.dot(pf_weights.T, np.dot(cov_matrix_annual, pf_weights)))
print('Portfolio annual volatility: ' + str(np.round(post_stddev, 5))+'%')

### ANNUALIZED RETURNS
pf_AUM = pf.iloc[:,0:numstocks].dot(pf_weights)
# Calculate the total return from the S&P500 value series
total_return = (pf_AUM[-1] - pf_AUM[0]) / pf_AUM[0]
print(total_return)

months = 12
# Over 1 jaar
annualized_return = ((1 + total_return)**(12/months))-1
print (annualized_return)

### ANNUALIZED VOLATILITY
annualized_vol = pf_returns['Portfolio'].std()*np.sqrt(250)

### SHARPE RATIO
risk_free = 0.01
sharpe_ratio = (annualized_return-risk_free)/annualized_vol
print(sharpe_ratio)

### PORTFOLIO STATS
print("Portfolio mean : ", np.mean(pf_returns['Portfolio']))
print("Portfolio std/vol  : ", np.std(pf_returns['Portfolio']))
#print("Portfolio var  : ", pf_returns_mean.std()**2)
print("Portfolio skew : ", stats.skew(pf_returns['Portfolio'])) # 0 is normaal. <1 / >1 is highly skewed. Positive skewness is bump to left of middle, which is good
print("Portfolio kurt : ", stats.kurtosis(pf_returns['Portfolio'])+3) # Normal = 3. >3 is fat-tailed and implies undesirable kurtosis. Quite often in extreme ends in the distribution

### PORTFOLIO VERGELIJKING MET BENCHMARK EW
# Equally weighted portfolio opzetten
pf_weights_eq = np.repeat(1/numstocks, numstocks)
df = pd.DataFrame()
df['pf_equal'] = pf_returns.iloc[:,0:numstocks].mul(pf_weights_eq, axis=1).sum(axis=1)
df['pf_myweights'] = pf_returns['Portfolio']

# Visualizeren van prestaties eigen portfolio t.o.v. equally weighted
cum_returns = ((1+df).cumprod()-1)
cum_returns[['pf_myweights','pf_equal']].plot()
plt.show()

### RELATIES TUSSEN AANDELEN
#Correlatiematrix | Zegt iets over lineaire relaties maar niets over variantie
pf_corr = pf.iloc[:,0:numstocks].corr() 
print(pf_corr)
sns.heatmap(pf_corr, annot=True, cmap="YlGnBu", linewidths=0.3, annot_kws={"size": 8})
plt.xticks(rotation=90)
plt.yticks(rotation=0) 
plt.show()
# Hoge en lage correlaties gewenst om marktrisico af te dekken


# https://towardsdatascience.com/time-series-forecasting-predicting-stock-prices-using-facebooks-prophet-model-9ee1657132b5
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# Load the dataset using pandas
data = yf.download("GOOG", start='2015-06-08', end='2020-12-28')
data.head(5)
data = data.reset_index()

data.describe()
data = data[["Date","Close"]] # select Date and Price
# Rename the features: These names are NEEDED for the model fitting
data = data.rename(columns = {"Date":"ds","Close":"y"}) #renaming the columns of the dataset
data.head(5)

from fbprophet import Prophet
m = Prophet(daily_seasonality = True) # the Prophet class (model)
m.fit(data) # fit the model using all data

future = m.make_future_dataframe(periods=365) #we need to specify the number of days in future
prediction = m.predict(future)
m.plot(prediction)
plt.title("Prediction of the Google Stock Price using the Prophet")
plt.xlabel("Date")
plt.ylabel("Close Stock Price")
plt.show()








#=============== OPTIMALE PORTFOLIO SAMENSTELLEN ================#
### PyPortfolioOpt ### ==> Modern portfolio theory
from pypfopt import risk_models
from pypfopt import expected_returns
from pypfopt.efficient_frontier import EfficientFrontier

start = '2018-01-01'
end = str(dt.today().date())

# ETFs
etf_espo = yf.download("ESPO", start=start, end=end) #https://www.vaneck.com/nl/en/?country=nl&audience=retail&lang=en
etf_ecar = yf.download("ECAR.L", start=start, end=end) #https://www.ishares.com/nl/particuliere-belegger/nl/producten/307130/ishares-electric-vehicles-and-driving-technology-ucits-etf-usd-acc-fund?switchLocale=y&siteEntryPassthrough=true
etf_eem = yf.download("EEM", start=start, end=today)
etf_iwrd = yf.download("IWRD.AS", start=start, end=today)
etf_tret = yf.download("TRET.MI", start=start, end=today) # Global real estate
etf_eurodividends = yf.download("IDVY.AS", start=start, end=today) # iShares EuroStoxx dividend
etf_globaldividends = yf.download("ISPA.DE", start=start, end=today) #
etf_eurodividend_groei = yf.download("IDJV.AS", start=start, end=today) #https://www.ishares.com/nl/particuliere-belegger/nl/producten/251793/ishares-euro-total-market-value-large-ucits-etf?switchLocale=y&siteEntryPassthrough=true
etf_sp500 = yf.download("SPY", start=start, end=today)
#etf_bjk = yf.download("BJK", start=start, end=today) #https://www.vaneck.com/nl/en/?country=nl&audience=retail&lang=en
etf_wcss = yf.download("WCSS.AS", start=start, end=today) #https://www.ishares.com/nl/professionele-belegger/nl/producten/308902/ishares-msci-world-consumer-staples-sector-ucits-etf-usd-dist-fund
etf_wtai = yf.download("WTAI.MI", start=start, end=today) #https://www.wisdomtree.eu/nl-nl/etfs/thematic/wtai---wisdomtree-artificial-intelligence-ucits-etf---usd-acc
etf_wcld = yf.download("WCLD.MI", start=start, end=today) #https://www.wisdomtree.eu/nl-nl/etfs/thematic/wcld---wisdomtree-cloud-computing-ucits-etf---usd-acc
etf_qqq = yf.download("QQQ", start=start, end=today) #https://www.wisdomtree.eu/nl-nl/etfs/thematic/wcld---wisdomtree-cloud-computing-ucits-etf---usd-acc
etf_smh_it = yf.download("SMH", start=start, end=today) #https://www.vaneck.com/nl/en/etf/equity/smh/overview/
etf_vanguardworld = yf.download("VWRL.AS", start=start, end=today)
#etf_smallcap = yf.download("IUSN.DE", start=start, end=today) 
etf_cleanenergy = yf.download("ICLN", start=start, end=today)
etf_emqq = yf.download("EMQQ", start=start, end=today)
#etf_inrg = yf.download("INRG.MI", start=start, end=today)

fondsen = [etf_smh_it,etf_espo, etf_ecar, etf_eem, etf_iwrd, etf_tret, etf_eurodividends,
           etf_eurodividend_groei, etf_sp500, etf_wcss, etf_wtai,
           etf_wcld, etf_qqq, etf_vanguardworld, etf_cleanenergy, etf_emqq]
fondsnamen = ['SMH','ESPO', 'ECAR','EEM','IWRD','TRET','EuroDividends','EuroDividendGroei',
              'SP500','WCSS','WTAI','WCLD','QQQ','VanguardWorld','CleanEnergy','EMQQ']
#pf_weights =  np.array([0.08,0.08,0.08,0.08,0.08,0.08,0.08,0.08,0.08,0.08,0.08,0.08]) # Aanpassen naarmate verhoudinge wijzigen
fondsen_lijst = []
for fonds in fondsen:
    fonds = fonds['Adj Close']
#    fonds['returns'] = fonds.pct_change()
#    fonds = fonds.div(fonds.iloc[0]).mul(100)
    fondsen_lijst.append(fonds)
pf = pd.concat(fondsen_lijst, axis=1)
#pf = pf.dropna()
pf.columns = fondsnamen
pf.info()


# ETFs
pf = df_etf
# Aandelen alle in NL
pf = df_nle

# 2) Mu en S instellen  
mu = expected_returns.mean_historical_return(pf) # <- expected return portfolio
Sigma = risk_models.sample_cov(pf) # <- measure of risk
print("Expected return portfolio: ", mu)
print("Measure of risk: ", Sigma)
# Obtain the efficient frontier

# Print portfolios:
weight_min = 0
weight_max = 0.2
target_return = 0.4
target_risk = 2.0
risk_free_rate = 0 # <- huidige rente
ef = EfficientFrontier(mu, Sigma, weight_bounds=(weight_min, weight_max)) # <- nooit meer dan 25% in een portfolio

"""Plotting options"""
#from pypfopt import plotting
#risk_range = np.linspace(0.10, 0.40, 100)
#plotting.plot_efficient_frontier(ef, ef_param='risk', ef_param_range=risk_range, show_assets=True, showfig=True)
#plt.show()

"""Portfolio options"""
# Nu kun je met verschillende functies portfolios uit de collectie van portfolios halen. Bijv .max_sharpe()
# Max sharpe portfolio
pf_optimal = pd.DataFrame()
ef = EfficientFrontier(mu, Sigma, weight_bounds=(weight_min, weight_max))
print(ef.max_sharpe()) # geeft optimaal gewogen portfolio tegen max sharpe
pf_optimal['norm_maxsharpe'] = ef.portfolio_performance(verbose=True, risk_free_rate=risk_free_rate)
pf_optimal.index = ['Expected annual return','Annual volatility','Sharpe ratio']
# Min volatility portfolio
ef = EfficientFrontier(mu, Sigma, weight_bounds=(weight_min, weight_max))
print(ef.min_volatility())
pf_optimal['norm_min_volatility'] = ef.portfolio_performance(verbose=True, risk_free_rate=risk_free_rate)
# Efficient risk portfolio
ef = EfficientFrontier(mu, Sigma, weight_bounds=(weight_min, weight_max))
print(ef.efficient_risk(target_risk)) # geeft portfolio optimaal return tegen gegeven target risk
pf_optimal['norm_efficientrisk'] = ef.portfolio_performance(verbose=True, risk_free_rate=risk_free_rate)
# Efficient return portfolio
ef = EfficientFrontier(mu, Sigma, weight_bounds=(weight_min, weight_max))
print(ef.efficient_return(target_return=target_return)) # geeft portfolio tegen minimaal risico en gegeven target return
pf_optimal['norm_efficientreturn'] = ef.portfolio_performance(verbose=True, risk_free_rate=risk_free_rate)

# Heatmap ideale 
sns.heatmap(pf_optimal, cmap='Greens', fmt='.2f', annot=True, cbar=False)
plt.title('Different portfolios and expected performance')
plt.show()

"""Code om metrics van de verwachte portfolio performance te meten"""
print(ef.portfolio_performance(verbose=True, risk_free_rate=risk_free_rate)) # Geeft optimale portfolio performance |  risk-free-rate is huidige rente
ef = EfficientFrontier(mu, Sigma, weight_bounds=(weight_min, weight_max))
sample_portfolio = ef.portfolio_performance()

# Maximum sharpe portfolio
ef = EfficientFrontier(mu, Sigma, weight_bounds=(weight_min, weight_max))
maxsharpe_rawweights = ef.max_sharpe()
maxsharpe_cleanedweights = ef.clean_weights()
print(maxsharpe_cleanedweights)
perf_max_sharpe_regulier = ef.portfolio_performance(verbose=True, risk_free_rate=0.00)
pf_regulier_maxsharpe_weights = maxsharpe_cleanedweights

# Minimum volatility portfolio
ef = EfficientFrontier(mu, Sigma, weight_bounds=(weight_min, weight_max))
minvol_rawweights = ef.min_volatility()
minvol_cleanweights = ef.clean_weights()
ef.portfolio_performance(verbose=True, risk_free_rate=0.00)
print('Min vol: ', minvol_cleanweights, 'Max sharpe: ', maxsharpe_cleanedweights, sep="\n")

# Target return against minimum risk
returns=pf.pct_change()
covMatrix = returns.cov()*252
Sigma = risk_models.sample_cov(pf)
print (covMatrix, Sigma)

ef = EfficientFrontier(mu, Sigma, weight_bounds=(weight_min, weight_max))
targetreturn_weights = ef.efficient_return(0.3) # = 10% winst bovenop rente; uitkomst laat zien welke aandelen je daarvoor nodig hebt
print (targetreturn_weights)
ef.portfolio_performance(verbose=True)

#####################################
##### RECENTERE DATA MEEGENOMEN #####
#####################################
# Expected returns waar meest recente data zwaarder wordt meegewogen
from pypfopt import expected_returns
mu_ema = expected_returns.ema_historical_return(pf, span=100 ,frequency=252)
Sigma_ew = risk_models.exp_cov(pf, span=100, frequency=252)
ef_recent = EfficientFrontier(mu_ema, Sigma_ew, weight_bounds=(weight_min, weight_max))

pf_recent = pd.DataFrame()
ef_recent = EfficientFrontier(mu_ema, Sigma_ew, weight_bounds=(weight_min, weight_max))
print(ef_recent.max_sharpe()) # geeft optimaal gewogen portfolio tegen max sharpe
pf_recent['recent_maxsharpe'] = ef_recent.portfolio_performance(verbose=True, risk_free_rate=risk_free_rate)
pf_recent.index = ['Expected annual return','Annual volatility','Sharpe ratio']
# Min volatility portfolio
ef_recent = EfficientFrontier(mu_ema, Sigma_ew, weight_bounds=(weight_min, weight_max))
print(ef_recent.min_volatility())
pf_recent['recent_min_volatility'] = ef_recent.portfolio_performance(verbose=True, risk_free_rate=risk_free_rate)
# Efficient risk portfolio
ef_recent = EfficientFrontier(mu_ema, Sigma_ew, weight_bounds=(weight_min, weight_max))
print(ef_recent.efficient_risk(target_risk)) # geeft portfolio optimaal return tegen gegeven target risk
pf_recent['recent_efficientrisk'] = ef_recent.portfolio_performance(verbose=True, risk_free_rate=risk_free_rate)
# Efficient return portfolio
ef_recent = EfficientFrontier(mu_ema, Sigma_ew, weight_bounds=(weight_min, weight_max))
print(ef_recent.efficient_return(target_return=target_return)) # geeft portfolio tegen minimaal risico en gegeven target return
pf_recent['recent_efficientreturn'] = ef_recent.portfolio_performance(verbose=True, risk_free_rate=risk_free_rate)

###################
##### SORTINO #####
###################
# Use the variance of the negative returns only -> semicovariance
# Sortino ratio (using only negative covariance to calculate downwards risk)
# Define exponentially weightedSigma and mu using stock_prices
Sigma_semi = risk_models.semicovariance(pf, benchmark=risk_free_rate, frequency=252)
mu_2 = expected_returns.ema_historical_return(pf, frequency=252, span=252)
ef_sortino = EfficientFrontier(mu_2, Sigma_semi, weight_bounds=(weight_min, weight_max))

pf_sortino = pd.DataFrame()
ef_sortino = EfficientFrontier(mu_2, Sigma_semi, weight_bounds=(weight_min, weight_max))
print(ef_sortino.max_sharpe()) # geeft optimaal gewogen portfolio tegen max sharpe
pf_sortino['sort_maxsharpe'] = ef_sortino.portfolio_performance(verbose=True, risk_free_rate=risk_free_rate)
pf_sortino.index = ['Expected annual return','Annual volatility','Sharpe ratio']
# Min volatility portfolio
ef_sortino = EfficientFrontier(mu_2, Sigma_semi, weight_bounds=(weight_min, weight_max))
print(ef_sortino.min_volatility())
pf_sortino['sort_min_volatility'] = ef_sortino.portfolio_performance(verbose=True, risk_free_rate=risk_free_rate)
# Efficient risk portfolio
ef_sortino = EfficientFrontier(mu_2, Sigma_semi, weight_bounds=(weight_min, weight_max))
print(ef_sortino.efficient_risk(target_risk)) # geeft portfolio optimaal return tegen gegeven target risk
pf_sortino['sort_efficientrisk'] = ef_sortino.portfolio_performance(verbose=True, risk_free_rate=risk_free_rate)
# Efficient return portfolio
ef_sortino = EfficientFrontier(mu_2, Sigma_semi, weight_bounds=(weight_min, weight_max))
print(ef_sortino.efficient_return(target_return=target_return)) # geeft portfolio tegen minimaal risico en gegeven target return
pf_sortino['sort_efficientreturn'] = ef_sortino.portfolio_performance(verbose=True, risk_free_rate=risk_free_rate)

# Samenbrengen en plotten
portfolios = pd.concat([pf_optimal, pf_recent, pf_sortino], axis=1).T
# Heatmap ideale 
sns.heatmap(portfolios, cmap='Greens', fmt='.2f', annot=True, cbar=False)
plt.title('Different portfolios and expected performance: sortino')
plt.show()


#####################################################################################################################################








#############################################################
# DATACAMP: Introduction to managing finance data in Python # 
#############################################################

import pandas as pd
from pandas_datareader.data import DataReader
from datetime import date

start = date(2010,1,1)
brent = DataReader('DCOILBRENTEU', 'fred', start)
wti = DataReader('DCOILWTICO', 'fred', start)

brent.info()
brent.plot()

ticker = 'RDSA.AS'
data_source = 'yahoo'

shell = DataReader(ticker, data_source, start) 
shell = shell['Adj Close']

df = pd.concat([brent, wti, shell], axis=1)
df = df.dropna().rename(columns={'Adj Close':'Shell'})

# Coronacijfers inladen
import pandas as pd
rivm_totaal = pd.read_csv('https://raw.githubusercontent.com/J535D165/CoronaWatchNL/master/data-geo/data-municipal/RIVM_NL_municipal.csv', index_col=None, parse_dates=True)
rivm_totaal['Datum'] = pd.to_datetime(rivm_totaal['Datum'])
rivm_types = rivm_totaal.pivot_table(index='Datum', columns='Type',values='AantalCumulatief',aggfunc='sum')
rivm_types = rivm_types.diff()

# Samenvoegen
#df = pd.concat([df, shell], axis=1)
#df = df.dropna().rename(columns={'Adj Close':'AEX'})
df1 = df.loc['2013-04-01':today]
df1.plot()

# Cijfers schalen
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
df_scaled = pd.DataFrame(scaler.fit_transform(df1))
df_scaled.columns = df1.columns
df_scaled.index = df1.index

# Visualiseren
df_scaled.loc['2015-01-01':today].plot()
plt.show()
df_scaled.loc['2020-01-01':today].plot()
plt.show()

# Correlatiematrix
df_corr = df_scaled.loc['2015-01-01':today].corr()
print(df_corr)



############################################################
# DATACAMP: Introduction to Portfolio Management in Python # 
############################################################
pf = aex_fondsen[['ASM.AS','ASML.AS','RDSA.AS']].loc['2018-01-01':'2020-01-01']
pf_AUM = pf.pct_change()
pf_AUM_mean = pd.DataFrame(pf_AUM.mean(axis=1)).dropna()
rfr = 0
months = 60
pf_AUM_mean.plot()

sns.distplot(pf_AUM_mean)

### Sharpe ratio berekenen ###
# Calculate total return and annualized return from price data 
total_return = (pf_AUM_mean.iloc[-1] - pf_AUM_mean.iloc[0]) / pf_AUM_mean.iloc[0]
# Annualize the total return over 4 year 
annualized_return = ((1 + total_return)**(12/months))-1
# Create the returns data 
pf_returns = pf_AUM_mean.pct_change()
# Calculate annualized volatility from the standard deviation
vol_pf = pf_AUM_mean.std()*np.sqrt(250)
# Calculate the Sharpe ratio 
sharpe_ratio = ((annualized_return - rfr) /vol_pf)
print (sharpe_ratio)

# Verschillende cijfers om distributie van de data te bepalen
print("mean : ", pf_AUM_mean.mean()*100)
print("Std. dev  : ", pf_AUM_mean.std()*100)
print("skew : ", pf_AUM_mean.skew())
print("kurt : ", pf_AUM_mean.kurtosis())

# Vergelijken distributies
# Print skewness and kurtosis of the stocks
aex_aum = aex_fondsen.loc['2015-01-01':'2020-01-01']
aex_AUM = aex_aum.pct_change()+100
aex_AUM_mean = aex_AUM.mean(axis=1)

aex_fondsen_skew = aex_AUM.skew()
print(aex_fondsen_skew.smallest(5)) # want negatief is met de bult naar rechts, dus vaker hogere uitslagen boven gemiddelde 
aex_fondsen_kurt = aex_AUM.kurtosis()
print(aex_fondsen_kurt.nsmallest(5))

print ("skew AEX : ", aex_AUM_mean.skew())
print ("kurt AEX : ", aex_AUM_mean.kurtosis())
aex_AUM.hist(bins=50)
aex_AUM_mean.hist(bins=50)

# Print the histogram of the portfolio
pf_returns.hist()
plt.show()

# Print skewness and kurtosis of the portfolio
print ("skew pf : ", pf_returns.skew())
print ("kurt pf : ", pf_returns.kurtosis())

# /==============================/ #


################################
# Machine learning - speeltuin #
################################

# AR model om koersen te voorspellen
ar_data = pd.DataFrame(df['Adj Close'].resample('M').mean())
ar_data = ar_data.dropna()

#ACF
from statsmodels.graphics.tsaplots import plot_acf
plot_acf(ar_data, alpha=1, lags=10)
plt.show()

# ARMA Model
from statsmodels.tsa.arima_model import ARMA
mod = ARMA(ar_data, order=(1,0)) # 1,0 = AR-model; 0,1 = ARMA-model
res = mod.fit()
results = res.summary()
print(results)

#Visuele voorspelling
res.plot_predict(start='2015', end='2021-01-01')
plt.legend(fontsize=8)
plt.ylabel('Koerswaarde Shell')
plt.title('Verloop koers Shell en voorspelling')
plt.show()

# Shell % wisselingen
shell_pct = df['Adj Close'].pct_change()
shell_pct.loc['2020-01-01':today].resample('W').mean().plot()

from statsmodels.graphics.tsaplots import plot_acf
plot_acf(df['Adj Close'], lags=1)


# Clustering
x = aex_fondsen.dropna().transpose()
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
x = scaler.fit_transform(x)

#Using the elbow method to determine the optimal number of cluster
from sklearn.cluster import KMeans #importing K-means clustering class
wcss = [] #within cluster sum of squares
for i in range(1,11):
    kmeans = KMeans(n_clusters=i, init='k-means++',max_iter=300,n_init=10,random_state=0)
    kmeans.fit(x)
    wcss.append(kmeans.inertia_)
    
# Create a plot in order to observe the elbow 
plt.plot(range(1,11), wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS') # Within Clusters Sum of Squares
plt.show()

# n_clusters aanpassen 
kmeans = KMeans(n_clusters=3, init='k-means++',max_iter=300, n_init=10)

# Predict the cluster for each data point
y_kmeans = kmeans.fit_predict(x)

# Example prediction
#prediction = kmeans.predict([['40','40000']])
"""to find the correct predicted cluster add 1 because clusters numbering in python starts from 0"""

# Make centroids for each cluster
centroids = kmeans.cluster_centers_

#Silhouette score
from sklearn import metrics
score = metrics.silhouette_score(x, y_kmeans)
print(score)

# Instantiate Silhouette Visualizer
from yellowbrick.cluster import SilhouetteVisualizer
plt2 = plt.figure(2)
visualizer = SilhouetteVisualizer(KMeans(3))
visualizer.fit(x) # fit the data to the visualizer
visualizer.poof()
plt2.show()

#Visualizing the cluster
plt3=plt.figure(3)
plt.scatter(x[y_kmeans==0,0],x[y_kmeans==0,1], s=50,c='blue',label='cluster 1')
plt.scatter(x[y_kmeans==1,0],x[y_kmeans==1,1], s=50,c='black',label='cluster 2')
#plt.scatter(x[y_kmeans==2,0],x[y_kmeans==2,1], s=50,c='orange',label='cluster 3')
#plt.scatter(x[y_kmeans==3,0],x[y_kmeans==3,1], s=50,c='yellow',label='cluster 4')
#plt.scatter(x[y_kmeans==4,0],x[y_kmeans==4,1], s=50,c='green',label='cluster 5')
#plt.scatter(centroids[:,0], centroids[:,1], marker='*', s=300, c='red',label='Centroids')
plt.title('Clusters van AEX-fondsen')
#plt.xlabel('Annual Income in $')
#plt.ylabel('Spending Score')
plt.legend()
plt3.show()

plt.clf()




#######################
# DEGIRO Kernselectie #
import tabula
import pandas as pd
import numpy as np

# DEGIRO Kernselectie
file = 'D:/OneDrive/Data projects/Beurs data/DEGIRO_Trackers_Kernselectie.pdf'

degiro = tabula.read_pdf(file, pages='3-7', multiple_tables=False, stream=True, guess=False)
degiro = degiro[0]
degiro.columns = map(str.lower, degiro.columns)
degiro.columns = ['ISIN naam','Valuta','Beurs']
degiro = degiro[['ISIN naam','Beurs']]
degiro = degiro.dropna()
degiro = degiro.iloc[1:,:].reset_index(drop=True)
degiro['ISIN code'] = degiro['ISIN naam'].str.split(' ').str[0]



#### DE GIRO TEST ####


import degiroapi
from degiroapi.product import Product
from degiroapi.order import Order
from degiroapi.utils import pretty_json

# Login account
degiro = degiroapi.DeGiro()
degiro.login("username", "password")

import degiroapi
from degiroapi.product import Product
from degiroapi.order import Order
from degiroapi.utils import pretty_json

from datetime import datetime, timedelta

# login
degiro = degiroapi.DeGiro()
degiro.login("username", "password")

# logout
degiro.logout()

# print the current cash funds
cashfunds = degiro.getdata(degiroapi.Data.Type.CASHFUNDS)
for data in cashfunds:
    print(data)

# print the current portfolio (True to filter Products with size 0, False to show all)
portfolio = degiro.getdata(degiroapi.Data.Type.PORTFOLIO, True)
for data in portfolio:
    print(data)

# output one search result
products = degiro.search_products('Pfizer')
print(Product(products[0]).id)
print(Product(products[0]).name)
print(Product(products[0]).symbol)
print(Product(products[0]).isin)
print(Product(products[0]).currency)
print(Product(products[0]).product_type)
print(Product(products[0]).tradable)
print(Product(products[0]).close_price)
print(Product(products[0]).close_price_date)

# output multiple search result
products = degiro.search_products('Pfizer', 3)
print(Product(products[0]).id)
print(Product(products[1]).id)
print(Product(products[2]).id)

# printing info for a specified product ID:
info = degiro.product_info(5322419)
print(info["id"], info["name"], info["currency"], info["closePrice"])

# print transactions
transactions = degiro.transactions(datetime(2019, 1, 1), datetime.now())
print(pretty_json(transactions))

# print order history (maximum timespan 90 days)
orders = degiro.orders(datetime.now() - timedelta(days=90), datetime.now())
print(pretty_json(orders))

# printing order history (maximum timespan 90 days), with argument True return only open orders
orders = degiro.orders(datetime.now() - timedelta(days=90), datetime.now(), True)
print(pretty_json(orders))

# deleting an open order
orders = degiro.orders(datetime.now() - timedelta(days=1), datetime.now(), True)
degiro.delete_order(orders[0]['orderId'])

degiro.delete_order("f278d56f-eaa0-4dc7-b067-45c6b4b3d74f")

# getting realtime and historical data from a stock
products = degiro.search_products('nrz')

# Interval can be set to One_Day, One_Week, One_Month, Three_Months, Six_Months, One_Year, Three_Years, Five_Years, Max
realprice = degiro.real_time_price(Product(products[0]).id, degiroapi.Interval.Type.One_Day)

# reatime data
print(realprice[0]['data']['lastPrice'])
print(pretty_json(realprice[0]['data']))

# historical data
print(realprice[1]['data'])

# get s&p 500 stock list
sp5symbols = []
products = degiro.get_stock_list(14, 846)
for product in products:
    sp5symbols.append(Product(product).symbol)

# get german30 stock list
daxsymbols = []
products = degiro.get_stock_list(6, 906)
for product in products:
    daxsymbols.append(Product(product).symbol)

# placing an order(dependent on the order type)
# set a limit order price to which the order gets executed
# order type, product id, execution time type (either 1 for "valid on a daily basis", or 3 for unlimited, size, limit(the limit price)
degiro.buyorder(Order.Type.LIMIT, Product(products[0]).id, 3, 1, 30)
# sets a limit order when the stoploss price is reached(not bought for more than the limit at the stop loss price)
# order type, product id, execution time type (either 1 for "valid on a daily basis", or 3 for "unlimited"), size, limit(the limit price), stop_loss(stop loss price)
degiro.buyorder(Order.Type.STOPLIMIT, Product(products[0]).id, 3, 1, 38, 38)
# order type, product id, execution time type (either 1 for "valid on a daily basis", or 3 for "unlimited"), size
degiro.buyorder(Order.Type.MARKET, Product(products[0]).id, 3, 1)
# the stop loss price has to be higher than the current price, when current price reaches the stoploss price the order is placed
# order type, product id, execution time type (either 1 for "valid on a daily basis", or 3 for "unlimited"), size, don't change none, stop_loss(stop loss price)
degiro.buyorder(Order.Type.STOPLOSS, Product(products[0]).id, 3, 1, None, 38)
# selling a product
# order type, product id, execution time type (either 1 for "valid on a daily basis", or 3 for unlimited, size, limit(the limit price)
degiro.sellorder(Order.Type.LIMIT, Product(products[0]).id, 3, 1, 40)
# order type, product id, execution time type (either 1 for "valid on a daily basis", or 3 for "unlimited"), size, limit(the limit price), stop_loss(stop loss price)
degiro.sellorder(Order.Type.STOPLIMIT, Product(products[0]).id, 3, 1, 37, 38)
# order type, product id, execution time type (either 1 for "valid on a daily basis", or 3 for "unlimited"), size
degiro.sellorder(Order.Type.MARKET, Product(products[0]).id, 3, 1)
# order type, product id, execution time type (either 1 for "valid on a daily basis", or 3 for "unlimited"), size, don't change none, stop_loss(stop loss price)
degiro.sellorder(Order.Type.STOPLOSS, Product(products[0]).id, 3, 1, None, 38)

