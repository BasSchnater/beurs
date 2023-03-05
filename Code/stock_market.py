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
start = "2015-01-01"
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
start = start
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

sns.set_style('darkgrid')
sns.set_context('paper')
sns.set_palette('Dark2')
boxspecs = dict(boxstyle='round', facecolor='white', alpha=0.3)

# AEX fondsen
aex_fondsen = ["BESI.AS","URW.AS","INGA.AS","AD.AS","PRX.AS","KPN.AS","ADYEN.AS","DSM.AS","WKL.AS","PHIA.AS","UNA.AS","GLPG.AS","ASM.AS","MT.AS","RAND.AS","ASML.AS","REN.AS","IMCD.AS","AKZA.AS","ASRNL.AS","NN.AS","AGN.AS","HEIA.AS"]
amx_fondsen = ['AALB.AS', 'AF.PA','ALFEN.AS',"APAM.AS","ARCAD.AS","BAMNB.AS","BFIT.AS","CRBN.AS","ECMPA.AS","FAGR.BR","FLOW.AS","FUR.AS","JDEP.AS","NSI.AS", "OCI.AS", "PHARM.AS","PNL.AS","SBMO.AS","LIGHT.AS","ABN.AS","TWEKA.AS","VPK.AS","WDP.BR"]
ascx_fondsen = ['AXS.AS','AJAX.AS','ACOMO.AS','AVTX.AS','BSGR.AS','BAMNB.AS','CMCOM.AS','HEIJM.AS','KENDR.AS','BOLS.AS','NEDAP.AS','NSI.AS','ORDI.AS','SIFG.AS','SLIGR.AS','TOM2.AS','VASTN.AS','WHA.AS']
nl_aandelen = aex_fondsen + amx_fondsen + ascx_fondsen

damrak = yf.download(nl_aandelen, start=start, end=today)
damrak = damrak['Adj Close']
damrak.index = pd.to_datetime(damrak.index.astype(str).str.slice(start=0, stop=11))

import quantstats as qs
for aandeel in nl_aandelen:
    stock = pd.DataFrame(damrak[aandeel])

    ticker = yf.Ticker(aandeel) # or pdr.get_data_yahoo(... 
    #from pandas_datareader import data as pdr
    #import yfinance as yf
    #yf.pdr_override() # <== that's all it takes :-)
    stock = pd.DataFrame(stock[aandeel].loc[start:end])
    stock.columns = ['Close']
    stock['Close'] = stock['Close'].bfill()
    # Technische analyse
    stock['20dSTD'] = stock['Close'].rolling(window=20).std()
    stock['MA20'] = stock['Close'].rolling(window=20).mean()
    stock['MA50'] = stock['Close'].rolling(window=50).mean()
    stock['MA200'] = stock['Close'].rolling(window=200).mean()
    stock['MA365'] = stock['Close'].rolling(window=365).mean()
    stock['Upper'] = stock['MA20'] + (stock['20dSTD'] * 2)
    stock['Lower'] = stock['MA20'] - (stock['20dSTD'] * 2)
    # advies
    #stock['advies'] = np.where(stock['Close'] < stock['MA200'], 'kooptip', 'neutraal')
    #stock['advies'] = np.where((stock['Close'] > stock['MA200']) & (stock['Close'] < stock['20dSTD']), 'hoog, niet kopen', stock['advies'])
    #stock['advies'] = np.where((stock['Close'] > stock['MA200']) & (stock['Close'] > stock['20dSTD']), 'uitgebroken, verkopen', stock['advies'])
    
    # Plots
    stock['Close'].plot(label='Dagkoers', color='black', linewidth=2)
    stock['Upper'].plot(label='Bollinger 20 2STD', color='red',linestyle='-', alpha=0.3)
    stock['Lower'].plot(label='', color='red',linestyle='-', alpha=0.3)
    stock['MA365'].plot(label='MA365', color='orange',linestyle='dotted')
    stock['MA200'].plot(label='MA200', color='blue', linestyle='dotted')
    stock['MA50'].plot(label='MA50', color='blue',)
    stock['MA200'].plot(label='MA200', color='orange')
    plt.title(aandeel)
    plt.xlim(start,today)
    plt.legend(bbox_to_anchor=(1,1))
#    plt.text(x=stock.index.min(), y=stock['Close'].max(), s=str('Sector: ' + stock['industry'].iloc[0]), bbox=dict(facecolor='white', edgecolor='none'))
#    plt.text(x=stock.index.min(), y=stock['Close'].max()/1.1, s=str('Groei gem.: ' + str(round(stock['avggrowth'].iloc[0],2))), bbox=dict(facecolor='white', edgecolor='none'))
#    plt.text(x=stock.index.min(), y=stock['Close'].max()/1.2, s=str('Groei max.: ') + str(round(stock['maxgrowth'].iloc[0],2)), bbox=dict(facecolor='white', edgecolor='none'))
#    plt.text(x=stock.index.min(), y=stock['Close'].max()/1.3, s=str('WPA: ') + str(stock['revenuePerShare'].iloc[0]), bbox=dict(facecolor='white', edgecolor='none'))
#    plt.text(x=stock.index.min(), y=stock['Close'].max()/1.4, s=str('P/E ratio: ') + str(stock['P/E ratio'].iloc[0]), bbox=dict(facecolor='white', edgecolor='red'))    
    #    plt.text(start,float(stock.min()), 'Source: Yahoo Finance. Graph by Bas Schnater (@BasSchnater)')
    plt.savefig(aandeel + '.png', format='png', dpi=100, bbox_inches="tight")
#    plt.savefig(path + redactie + '_productie.png', format='png', dpi=100, bbox_inches="tight")
    plt.show()
    
    
"""    # -- Quantstats
    qs.extend_pandas()
    # fetch the daily returns for a stock
    stock = pd.Series(damrak[aandeel])

    ### Plots
    qs.plots.snapshot(stock, title='Stock Performance')
    qs.reports.basic(stock)
    
    qs.reports.html(stock, benchmark='AEX.AS')
    stock.to_plotly()

    qs.reports.html(stock, output=str(stock) + '_stats.html', title='BTC Sentiment')
    stock.plot_monthly_heatmap(savefig='output/fb_monthly_heatmap.png')
    stock.plot_earnings(savefig='output/fb_earnings.png', start_balance=100000)""" 
    
"""    # Technische analyse
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

# Yahoo data werkt nu niet, later weer proberen 
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
"""


#### Quantstats Portfolio Analytics ####
# Documentation: https://pypi.org/project/QuantStats/

# -- Setup
import pandas as pd
import numpy as np
import quantstats as qs
qs.extend_pandas()

### Stocks en comparison ###
stock_name = 'ALFEN.AS'
stock = qs.utils.download_returns(stock_name, period="10y")
stock = stock.rename(stock_name)
stock.index = stock.index.tz_localize(None)

stock_benchmark = 'AEX'
benchmark = qs.utils.download_returns(stock_benchmark, period="10y")
benchmark = benchmark.rename(stock_benchmark)
benchmark.index = benchmark.index.tz_localize(None)

### Stats
[f for f in dir(qs.stats) if f[0] != '_'] # <-- alle beschikbare statistieken
qs.reports.metrics(mode='full', returns=stock)
qs.stats.sharpe(stock)
qs.stats.smart_sharpe(stock)

### Plots
[f for f in dir(qs.plots) if f[0] != '_'] # <-- alle beschikbare statistieken
qs.plots.snapshot(stock, title='Stock Performance')
#qs.plots.to_plotly(fig) # <-- werkt nog niet
qs.plots.rolling_sharpe(stock)

### Reports
qs.reports.full(stock) # Produces all plots
stock.plot_distribution()#savefig='fb_earnings.png')#, start_balance=100)

### HTML tearsheet
qs.reports.html(stock, benchmark=(benchmark), output='output.html', download_filename=stock_name + '.html', title=stock_name)



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
