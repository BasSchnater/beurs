# -*- coding: utf-8 -*-
"""
Created on Thu Dec 10 21:16:55 2020

@author: bassc
"""

import numpy_financial as np_fin
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Toekomstig aandelen
future_value = 10000 * (1+0.05)**(25)
print("Future Value of Investment: " + str(round(future_value, 2)))

# ===== Waardestijging ===== #
# Calculate stock returns
stock = np_fin.fv(rate=0.07, nper=20, pmt=-1800, pv=-10000)
print("Aandelen zullen stijgen tot €" + str(round(stock, 2)) + " in 20 jaar met jaarlijks €1800 inleg")
# Calculate etf returns
etf = np_fin.fv(rate=0.035, nper=20, pmt=0, pv=-5000)
print("ETFs zullen stijgen tot €" + str(round(etf, 2)) + " in 20 jaar zonder maandelijks €250 extra inleg")
# Calculate fund returns
fund = np_fin.fv(rate=0.02, nper=20, pmt=0, pv=-5000)
print("Fund ING zal stijgen tot €" + str(round(fund, 2)) + " in 20 jaar zonder maandelijks €250 extra inleg")

returns = pd.DataFrame()
returns['stock'] = np_fin.fv(rate=0.07/12, nper=np.arange(0, 121, 1), pv=-5839, pmt=-150)
returns['etf'] = np_fin.fv(rate=0.035/12, nper=np.arange(0, 121, 1), pv=-2121, pmt=-150)
returns['fund'] = np_fin.fv(rate=0.02/12, nper=np.arange(0, 121, 1), pv=-5414, pmt=-100)
returns['savings'] = np_fin.fv(rate=0.005/12, nper=np.arange(0, 121, 1), pv=-2000, pmt=-500)
returns['total'] = returns.sum(axis=1)

# Groei kapitaal
returns[['stock','etf','fund','savings']].plot(title='Vermogen')
plt.xlabel('Maanden')
plt.ylabel('Bedrag')
plt.show()

# Groei aandelen bij verschillende inleg 
returns_stock = pd.DataFrame()
returns_stock['150 per maand'] = np_fin.fv(rate=0.07/12, nper=np.arange(0, 121, 1), pv=-5839, pmt=-150)
returns_stock['250 per maand'] = np_fin.fv(rate=0.07/12, nper=np.arange(0, 121, 1), pv=-5839, pmt=-250)
returns_stock['500 per maand'] = np_fin.fv(rate=0.07/12, nper=np.arange(0, 121, 1), pv=-5839, pmt=-500)
returns_stock.plot()
  
# ===== Waardedaling ===== #
initial_investment = 10000
growth_rate = 0.05
growth_periods = 5
future_value = initial_investment * (1-growth_rate)**growth_periods
print("Future value: " + str(round(future_value, 2)))

