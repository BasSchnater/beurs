
#=============== INDIVIDUELE AANDELEN BEOORDELEN ================#
import pandas as pd
import numpy as np
import pyfolio

# Pyfolio documentatie: https://nbviewer.org/format/slides/github/quantopian/pyfolio/blob/master/pyfolio/examples/pyfolio_talk_slides.ipynb#/
# Pyfolio index error: https://github.com/quantopian/pyfolio/issues/661
portfolio = aex_namen

    start = start
    stock_rets = yf.download('ASM.AS', start='2019-01-01', end=today)
    pyf = pyf['Adj Close']

    ### Portfolio analysis using Pyfolio ### 
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



###### Pyfolio portfolio analytics ######
# Meer voorbeelden: https://nbviewer.org/github/quantopian/pyfolio/tree/master/pyfolio/examples/
# Hulp bij interpretaties: https://www.fmz.com/digest-topic/5798
# Volledige lessen: https://www.quantrocket.com/codeload/quant-finance-lectures/quant_finance_lectures/Lecture33-Portfolio-Analysis-with-pyfolio.ipynb.html
# Uitlezen grafieken: https://towardsdatascience.com/the-easiest-way-to-evaluate-the-performance-of-trading-strategies-in-python-4959fd798bb3

# ERROR: oplossing bij nieuwe installatie https://github.com/quantopian/pyfolio/issues/652

# Pyfolio werkt vooral met returns, dus aantal voorwaarden:
# 1) Moet een series zijn
# 2) Moeten returns zijn (dus % change))

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