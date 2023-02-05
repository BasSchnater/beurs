


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