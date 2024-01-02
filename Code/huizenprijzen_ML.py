##### HUIZENPRIJZEN #####
#Huizenprijzen
import pandas as pd
import seaborn as sns
import cbsodata

https://opendata.cbs.nl/statline/portal.html?_la=nl&_catalog=CBS&tableId=37610

huizen = pd.DataFrame(cbsodata.get_data('83625NED'))
#huizen = pd.DataFrame(cbsodata.get_data('37610'))
huizen = huizen.rename(columns={'GemiddeldeVerkoopprijs_1':'prijs_gem','RegioS':'Regio'}).drop(columns=['ID'])
huizen_ams = huizen[huizen['Regio'] == 'Amsterdam']
huizen_ams.index = huizen_ams['Perioden']
huizen_ams['pct_change'] = huizen_ams['prijs_gem'].pct_change()
huizen_ams['prijs_gem'].plot(label='Amsterdam')
huizen_lelystad = huizen[huizen['Regio'] == 'Lelystad']
huizen_lelystad.index = huizen_lelystad['Perioden']
huizen_lelystad['pct_change'] = huizen_lelystad['prijs_gem'].pct_change()
huizen_lelystad['prijs_gem'].plot(label='Lelystad')
plt.title('Groei gem. huizenprijs')
plt.ylabel('€ gem. huizenprijs')
plt.legend()
plt.show()

huizen_pivot = huizen.pivot(index='Perioden', columns='Regio', values='prijs_gem')
huizen_pivot.index = pd.to_datetime(huizen_pivot.index)
huizen_pivot['gem_nl'] = huizen_pivot.mean(axis=1)
huizen_pivot[['Amsterdam','Rotterdam','Lelystad','Diemen','Weesp','Haarlemmermeer']].resample('3Y').mean().plot()
huizen_pivot['gem_nl'].rolling(5).mean().plot()
plt.show()
huizen_pivot_pct = huizen_pivot.pct_change()
huizen_pivot_pct[['Amsterdam','Rotterdam','Lelystad','Diemen','Weesp','gem_nl']].plot()
plt.figure()
huizen_pivot_pct[['gem_nl','Amsterdam']].hist(bins=5, sharex=True)
plt.ylabel('Aantal')
plt.xlabel('% stijging/daling WOZ per jaar')
plt.show()

print("Gem stijging huizenprijzen in Amsterdam sinds 1995: ", huizen_pivot_pct['Amsterdam'].mean()*100)
print("Gem stijging huizenprijzen in Nederland sinds 1995: ", huizen_pivot_pct['gem_nl'].mean()*100)

samen= pd.DataFrame()
samen['Amsterdam'] = huizen_ams['pct_change']
samen['lelystad'] = huizen_lelystad['pct_change']
samen = samen.dropna()

recent = samen.iloc[19:26,:]
print(recent.mean())

samen.plot.bar(title='Jaarlijkse huizenprijs stijging/daling')
plt.ylabel('% stijging/daling')
plt.show()

bbp = pd.DataFrame(cbsodata.get_data('84087NED'))
bbp = bbp[['Perioden','BrutoBinnenlandsProduct_184']].rename(columns={'BrutoBinnenlandsProduct_184':'BBP'})
bbp.index = bbp['Perioden']
bbp.head()

df_corr = bbp['BBP'].corr(markt_ams['prijs_gem'])
print(df_corr)
#inflatie = pd.DataFrame(cbsodata.get_data('83131NED'))
#inflatie = inflatie[inflatie['Bestedingscategorieen'] == '040000 Huisvesting, water en energie']
#inflatie = inflatie.drop(columns=['MaandmutatieCPI_3','MaandmutatieCPIAfgeleid_4','Wegingscoefficient_7'])
#inflatie['Perioden'] = pd.to_datetime(inflatie['Perioden'])

markt_ams = pd.concat([huizen_ams,bbp], axis=1)
markt_ams['prijs_gem'].plot(label='Amsterdam')
plt.legend(loc='upper left')
markt_ams['BBP'].plot(secondary_y=True)
plt.title('Huizenprijzen Amsterdam t.o.v. BBP')
plt.legend(loc='lower right')
plt.show()

huizen_pivot = huizen.pivot_table(index='Perioden', columns='Regio', values='prijs_gem')
#huizen_pivot_pct = huizen_pivot.pct_change()
plt.plot()
huizen_pivot[['Amsterdam','Almere','Hoorn','Haarlem','Weesp']].plot()
huizen_pivot[['Noord-Holland (PV)','Flevoland (PV)']].plot(linestyle='dotted')
huizen_pivot['Lelystad'].plot(label='Lelystad', linewidth=3, color='black')
huizen_pivot['Amsterdam'].plot(label='Amsterdam', linewidth=3, color='darkred', marker='X', linestyle='')
plt.title('Gemiddelde huizenprijs per gemeente')
plt.ylabel('Gem. huizenprijs')
plt.xlabel('Jaar')
plt.legend()
plt.show()

huizen_pivot_pct = huizen_pivot.pct_change()
huizen_pivot_pct['gem'] = huizen_pivot_pct.mean(axis=1)
huizen_pivot_pct[['Amsterdam','Weesp','Hoorn','Haarlem']].plot.bar(title='% verandering huizenprijzen t.o.v. jaar ervoor')
#huizen_pivot_pct['Lelystad'].plot.bar(linewidth=2,color='black', alpha=0.7)
plt.legend()
plt.show()

print(huizen_pivot_pct.iloc[-1].nlargest(200))
print(huizen_pivot_pct.iloc[-1].nsmallest(20))

huizen_pivot_pct.iloc[-1].nlargest(250).plot.barh('% huizenprijsverschil 2019 t.o.v. 2018')
huizen_pivot_pct['Lelystad'].iloc[-1].plot.bar(color='black')
plt.ylabel('% huizenprijs-verschil (100% = 1)')
plt.show()

huizen_pct_volgorde = huizen_pivot_pct.iloc[-1].sort_values(ascending=False).reset_index()
huizen_pct_volgorde.dropna()

###################
# Machine learing #
###################

# ===== Voorspellen groei huizenprijzen ===== #
import pandas as pd
import seaborn as sns
import cbsodata

huizen = pd.DataFrame(cbsodata.get_data('83625NED'))
huizen = huizen.rename(columns={'GemiddeldeVerkoopprijs_1':'prijs_gem','RegioS':'Regio'}).drop(columns=['ID'])
huizen_pivot = huizen.pivot_table(index='Perioden', columns='Regio', values='prijs_gem')


import numpy as np
from scipy import stats
#from mat4py import loadmat
import statsmodels.api as sm
import pandas as pd
import matplotlib.pyplot as plt

## 1. Load data
dfX  = pd.DataFrame(huizen_pivot['Amsterdam']) # in this file the time series data is labeled "x"

#x = loadmat('ts_data.mat')
#dfX =  pd.DataFrame({'x': np.array(x['x'][:])[:,0]})
x = dfX['Amsterdam'].values
T = len(x)

dfX['xlag'] = dfX['Amsterdam'].shift(1)
dfX['xlag2'] = dfX['Amsterdam'].shift(2)
dfX['xlag3'] = dfX['Amsterdam'].shift(3)
dfX['xlag4'] = dfX['Amsterdam'].shift(4)
dfX['xlag5'] = dfX['Amsterdam'].shift(5)
dfX['xlag6'] = dfX['Amsterdam'].shift(6)
dfX['xlag7'] = dfX['Amsterdam'].shift(7)
dfX['xlag8'] = dfX['Amsterdam'].shift(8)
dfX['xlag9'] = dfX['Amsterdam'].shift(9)
dfX['xlag10'] = dfX['Amsterdam'].shift(10)
dfX['xlag11'] = dfX['Amsterdam'].shift(11)
dfX['xlag12'] = dfX['Amsterdam'].shift(12)
dfX['xlag13'] = dfX['Amsterdam'].shift(13)
dfX['xlag14'] = dfX['Amsterdam'].shift(14)
dfX['xlag15'] = dfX['Amsterdam'].shift(15)
dfX['const'] = 1

## 2. Run AR regression using sm.ols
y = dfX['Amsterdam']
X = dfX[['const','xlag']]
model = sm.OLS(y,X,missing='drop')
resultsAR = model.fit()                     # resultsARcontains all the estimation information

print(resultsAR.summary()) #prints detailed version of regression results
# Filteren P-values >0.05

b = resultsAR.params # save coefficients

## 3. Forward iterate AR(2) model to obtain forecast
h = 10 # define number of forecast periods
xforecast = np.zeros((T+h-1,1))  #define length of forecast vector (including observed data)
xforecast[0:T,] = dfX['Amsterdam'].values.reshape(T,1);     #set first T periods equal to observed data

for t in range((T),(T+h-1)):  # start forward iteration loop
    xforecast[t,] = b[0] + b[1]*xforecast[(t-1),:]# + b[2]*xforecast[(t-2),:] # !!!!aanpassen aan aantal lags!!!!

## 4. Plot observed data and forecast
plt.plot(xforecast,'r')
plt.plot(x,'k')
plt.title('Toekomstige huizenprijzen Amsterdam')
plt.show()
