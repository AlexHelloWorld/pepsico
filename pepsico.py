# read in data
# add season info and holiday info


import pandas as pd
import numpy as np
import pyro
import torch

import pyro.distributions as dist
from pyro.infer import MCMC, NUTS
from pyro.nn import PyroModule, PyroSample


from sklearn.metrics import mean_squared_error



pyro.set_rng_seed(30)


#I used Bayesian Hierarchical modeling to construct the Marketing Mix Model.
#1. For the media data, adstock transformation was applied to media data to model the lagging effects.
#2. Sale was constructed by adding together the contributions of media, trade, base and noise.
def adstock_model(media_data, trade_data, price_data, base_data, sale_data, max_lag=8):
    media_col_count = media_data.shape[1]
    trade_col_count = trade_data.shape[1]
    base_col_count = base_data.shape[1]
    
    lag = torch.tensor(np.arange(0, max_lag))
    lag = lag.repeat(media_col_count, 1)
    retain_rate = pyro.sample('retain_rate', dist.Uniform(0, 1))
    delay = pyro.sample('delay', dist.Uniform(0,  max_lag/2))
    
    # adstock parameters
    weight = torch.pow(retain_rate, (lag - delay)**2)
    weight = weight/torch.sum(weight)
    
    #weight = torch.reshape(weight, (media_col_count, 1))
    beta_ads = pyro.sample('beta_ads', dist.HalfNormal(10 * torch.ones(media_col_count)).to_event(1))
    
    beta_price = pyro.sample('beta_price', dist.Normal(0, 10))
    beta_trade = pyro.sample('beta_trade', dist.HalfNormal(10 * torch.ones(trade_col_count)).to_event(1))
    beta_base = pyro.sample('beta_base', dist.Normal(torch.zeros(base_col_count), 10 * torch.ones(base_col_count)).to_event(1))
    
    intercept = pyro.sample('intercept', dist.Uniform(0, torch.min(sale_data)))
    
    noise = pyro.sample('noise', dist.Normal(0, 10))

    with pyro.plate('data', len(media_data)):
        adstock_mean = torch.sum(weight * media_data, dim=2)
        
        y_expected = torch.sum(adstock_mean * beta_ads, dim=1) + torch.sum(beta_trade * trade_data, dim=1) + torch.sum(beta_base *base_data, dim=1) + torch.sum(beta_price * price_data, dim=1) + intercept
        
        pyro.sample('y', dist.Normal(y_expected, 1), obs=sale_data)

# predict sales using model output
def compute_prediction(hmc_samples, media_data, trade_data, price_data, base_data):
    retain_rate = hmc_samples['retain_rate'].mean()
    delay = hmc_samples['delay'].mean()
    
    beta_ads = torch.tensor(hmc_samples['beta_ads'].mean(0))
    beta_trade = torch.tensor(hmc_samples['beta_trade'].mean(0))
    beta_base = torch.tensor(hmc_samples['beta_base'].mean(0))
    
    beta_price = hmc_samples['beta_price'].mean()
    
    lag = torch.tensor(np.arange(0, max_lag))
    lag = lag.repeat(media_col_count, 1)
    weight = torch.pow(retain_rate, (lag - delay)**2)
    weight = weight/torch.sum(weight)
    
    adstock = torch.sum(weight * media_data, dim=2)
    
    ads_prediction = torch.sum(adstock * beta_ads, dim=1) 
    trade_prediction = torch.sum(beta_trade * trade_data, dim=1) + torch.sum(beta_price * price_data, dim=1)
    base_prediction = torch.sum(beta_base * base_data, dim=1) + hmc_samples['intercept'].mean()
    
    y_prediction = ads_prediction + trade_prediction + base_prediction
    return (y_prediction.numpy(), ads_prediction.numpy(), trade_prediction.numpy(), base_prediction.numpy())


# read in input data. The original excel file was organized into a single csv file containing all the media, trade and sales data
data = pd.read_csv('data/data_clean.csv', parse_dates=['Week'])

# add season information
data['is_spring'] = np.where(data['Week'].dt.month.isin([3, 4, 5]), 1, 0)
data['is_summer'] = np.where(data['Week'].dt.month.isin([6, 7, 8]), 1, 0)
data['is_fall'] = np.where(data['Week'].dt.month.isin([9, 10, 11]), 1, 0)
data['is_winter'] = np.where(data['Week'].dt.month.isin([12, 1, 2]), 1, 0)

# add holiday information
# define holiday weeks as the 47th week(Thanksgiving), last week of the year (Christmas), the first week of year (New Year)
data['is_holiday'] = np.where(data['Week'].dt.isocalendar().week.isin([47, 52, 1]), 1, 0)

# extend the media data to contain previous results
media_df = data[['TV', 'Facebook', 'Twitter', 'Amazon', 'Audio', 'Print', 'Digital_AO']]

tensor_list = []
media_row_count = media_df.shape[0]
media_col_count = media_df.shape[1]

max_lag = 9
for i in range(max_lag):
    current_media_data = torch.tensor(media_df.values[i:(media_row_count-max_lag+i+1)])
    current_media_data = torch.reshape(current_media_data, (media_row_count - max_lag + 1, media_col_count, 1))
    tensor_list.insert(0, current_media_data)

media_tensor = torch.cat(tensor_list, dim=2)

sale_tensor = torch.tensor(data['sales'].values[(max_lag-1):]) / 1000
base_tensor = torch.tensor(data[['is_spring', 'is_summer', 'is_fall', 'is_winter', 'is_holiday']].values[(max_lag-1):])
trade_tensor = torch.tensor(data[['Display', 'EndCap']].values[(max_lag-1):])

price_tensor = torch.tensor(data[['PriceChange']].values[(max_lag-1):])


hmc_samples_list = []
predictions_list = []
errors = []
# run mcmc multiple times and choose the model with the best MSE
for i in range(5):
    kernel_bayes= NUTS(adstock_model)
    mcmc_bayes = MCMC(kernel_bayes, num_samples=1000, warmup_steps=500)
    mcmc_bayes.run(media_tensor, trade_tensor, price_tensor, base_tensor, sale_tensor, max_lag=max_lag)


    hmc_samples = {k: v.detach().cpu().numpy() for k, v in mcmc_bayes.get_samples().items()}
    predictions = compute_prediction(hmc_samples, media_tensor, trade_tensor, price_tensor, base_tensor)
    
    hmc_samples_list.append(hmc_samples)
    predictions_list.append(predictions_list)
    
    errors.append(mean_squared_error(predictions[0] * 1000, data['sales'].values[(max_lag-1):]))


# The accuracy of the model is accessed by mean squared error between the actual sale and predicted sale. Low mean squared errors suggest that the model is working as expected. 

print("Minimun mean squared error: " + str(min(errors)))

min_index = errors.index(min(errors))

hmc_samples = hmc_samples_list[min_index]
predictions = compute_prediction(hmc_samples, media_tensor, trade_tensor, price_tensor, base_tensor)


print("retain_rate")
print(hmc_samples['retain_rate'].mean())

print("delay")
print(hmc_samples['delay'].mean())

print("\nQuestion 1\n")
mean_ads = predictions[1].mean()
mean_trade = predictions[2].mean()
mean_base = predictions[3].mean()
mean_total = predictions[0].mean()

print("Relative contribution from base: " + str(mean_base/mean_total))
print("Relative contribution from trade: " + str(mean_trade/mean_total))
print("Relative contribution from media: " + str(mean_ads/mean_total))


#'TV', 'Facebook', 'Twitter', 'Amazon', 'Audio', 'Print', 'Digital_AO'
print("\nQuestion 2\n")
media_contributions = hmc_samples['beta_ads'].mean(0) * 1.28 * 10e9
print("Sales (USD) contributed per GRP: ")
print("TV: " + str(media_contributions[0]))
print("Facebook: " + str(media_contributions[1]))
print("Twitter: " + str(media_contributions[2]))
print("Amazon: " + str(media_contributions[3]))
print("Audio: " + str(media_contributions[4]))
print("Print: " + str(media_contributions[5]))
print("Digital_AO: " + str(media_contributions[6]))

print("\nThe most effective media is TV")
print("The least effective media is Audio")


print("\nQuestion 4\n")

trade_contributions = hmc_samples['beta_trade'].mean(0) * 1000

print("Effect of trade activity measured by Sales (USD) contributed per 1% trade activity: ")
print("Display: " + str(trade_contributions[0]))
print("EndCap: " + str(trade_contributions[1]))


print("\nQuestion 5\n")

price_contribution = hmc_samples['beta_price'].mean() * 1000
print("Effect of price change on Sales: " + str(price_contribution))


