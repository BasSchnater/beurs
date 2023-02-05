
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

