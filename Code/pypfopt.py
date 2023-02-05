

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

