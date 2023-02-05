import pandas as pd
import numpy as np


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
