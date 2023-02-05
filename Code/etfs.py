import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf

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