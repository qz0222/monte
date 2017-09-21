sns.set_style("whitegrid")
from pandas.io.data import DataReader
from datetime import datetime
from __future__ import division
tech_list = ["AAPL","GOOG","MSFT","AMZN"]
end = datetime.now()
start = datetime(end.year-1,end.month,end.day)

for stock in tech_list:
    globals()[stock] = DataReader(stock,"yahoo",start,end)

AAPL.head()
AAPL.describe()
AAPL["Adj Close"].plot(legend = True,figsize=(10,4))
AAPL["Volume"].plot(legend = True,figsize = (10,4))
ma_day = [10,20,50]
for ma in ma_day:
    column_name = "MAfor %s days" %(str(ma))
    AAPL[column_name] = pd.rolling_mean(AAPL["Adj Close"],ma)

AAPL[["Adj Close","MAfor 10 days","MAfor 20 days","MAfor 50 days"]].plot(subplots = False, figsize=(10,4))
AAPL["Daily Return"] = AAPL["Adj Close"].pct_change()
AAPL["Daily Return"].plot(legend = True, figsize= (10,4),marker = "o",linestyle = "--")
sns.distplot(AAPL["Daily Return"].dropna(),bins = 100,color = "purple")
AAPL["Daily Return"].hist(bins=100)

closing_df = DataReader(tech_list,"yahoo",start,end)["Adj Close"]
tech_rets = closing_df.pct_change()
sns.jointplot("GOOG","GOOG",tech_rets,kind = "scatter",color = "seagreen")


sns.pairplot(tech_rets.dropna())

returns_fig = sns.PairGrid(closing_df)
returns_fig.map_upper(plt.scatter,color = "purple")
returns_fig.map_lower(sns.kdeplot,color = "cool_d")
returns_fig.map_diag(plt.hist,bins=30)


sns.corrplot(tech_rets.dropna(),annot = True)

rets = tech_rets.dropna()
area = np.pi*20
plt.scatter(rets.mean(),rets.std(),s=area)
plt.xlabel("Expected Return")
plt.ylabel("Risk")

sns.distplot(AAPL["Daily Return"].dropna(),bins=100,color="purple")
rets["AAPL"].quantile(0.05)

#Monte Carlo Method
days = 365
dt = 1/days
mu = rets.mean()["GOOG"]
sigma = rets.std()["GOOG"]

def stock_monte_carlo(start_price,days,mu,sigma):
    price = np.zeros(days)
    price[0] = start_price
    shock = np.zeros(days)
    drift = np.zeros(days)
    for x in xrange(1,days):
        shock[x] = np.random.normal(loc = mu*dt,scale = sigma*np.sqrt(dt))
        drift[x] = mu*dt
        price[x] = price[x-1]+(price[x-1]*(shock[x]+drift[x]))
    return price

start_price = 540.74
for x in xrange(100):
    plt.plot(stock_monte_carlo(start_price,days,mu,sigma))
plt.xlabel("Days")
plt.ylabel("Price")
plt.title("Monte Carlo Analysis for Google")

runs = 10000
simulations = np.zeros(runs)
for run in xrange(runs):
    simulations[run] = stock_monte_carlo(start_price,days,mu,sigma)[days-1]

q = np.percentile(simulations,1)
plt.hist(simulations,bins=200)
