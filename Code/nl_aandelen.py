import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf

sns.set_style('darkgrid')
sns.set_context('paper')
sns.set_palette('Dark2')
boxspecs = dict(boxstyle='round', facecolor='white', alpha=0.3)

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
