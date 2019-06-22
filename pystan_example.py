from pystan import StanModel
from sklearn.preprocessing import OneHotEncoder, MaxAbsScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from joblib import dump, load
import arviz as az
import networkx as nx


plt.rcParams["figure.figsize"] = (16,12)
plt.rcParams["font.size"] = 20

model_code = 'parameters {real y;} model {y ~ normal(0,1);}'
model = StanModel(model_code=model_code)
y = model.sampling().extract()['y']
print('If this worked, you will see a value near 0 now:')
print(y.mean())


# Example from „Kruschke: Doing Bayesian Data Analysis”
# create the DSO (dynamic shared object)
coin_model = StanModel(file=Path('models/bernoulli_example.stan').open())
# generate some data
N = 50;
z = 10;
y = [1] * z + [0] * (N-z)
coin_data = {'y': y, 'N': N}
# warmup is the same as burnin in JAGS
coin_fit = coin_model.sampling(data=coin_data, chains=3, iter=1000, warmup=200)
coin_test = coin_fit.extract()


##
# 1st example from Stan User's Guide
##
linear_model = StanModel(file=Path('models/linear_example.stan').open())
x = list(range(10))
y = [1.1, 2.04, 3.07, 3.88, 4.95, 6.11, 7.03, 7.89, 8.91, 10]
linear_data = {'x':x, 'y':y, 'N': 10}
linear_fit = linear_model.sampling(data=linear_data)
linear_res = linear_fit.extract()
α = np.mean(linear_res['alpha'])
β = np.mean(linear_res['beta'])

##
# Tobit Model
# according to https://www.r-bloggers.com/bayesian-models-with-censored-data-a-comparison-of-ols-tobit-and-bayesian-models/
##
file_location = Path('data/ucla_tobit_example.csv')
tobit_data = pd.read_csv(file_location)
# Stan does not support categorical features - need dummy variables.
cat_trans = Pipeline(steps=[('onehot', OneHotEncoder())])
ct = ColumnTransformer([('cat', cat_trans, ['prog'])])
coded_progs = ['academic', 'general','vocational']
trans_progs = pd.DataFrame(ct.fit_transform(tobit_data), columns=coded_progs)
tobit_data = pd.concat([tobit_data, trans_progs], axis=1)
predictors = ['read', 'math', 'general', 'vocational']

not_800 = tobit_data['apt'] != 800
is_800 = tobit_data['apt'] == 800
N_cens = is_800.sum()

censored_dict = {'X': tobit_data[predictors], 'N': tobit_data.shape[0], 
                 'y': tobit_data['apt'], 'N_cens': N_cens,
                 'K': len(predictors), 'y_cens': tobit_data[is_800]['apt']}

# 1) as comparison, do a linear model:
tobit_datadict = {'y': tobit_data['apt'], 'N': tobit_data.shape[0], 'K': len(predictors),
                  'X': tobit_data[predictors]}
tobit_linear_model = StanModel(file=Path('models/linear_students.stan').open())
tob_lin_fit = tobit_linear_model.sampling(data=tobit_datadict, iter=20000, chains=4)
tob_lin_res = tob_lin_fit.extract()

al = tob_lin_res['alpha'][5001:].mean()
beta = tob_lin_res['beta'][5001:].mean(axis=0)
# getting similar values for read and math like in the example, cool!
# intercept: 242.735; mydata$read: 2.553; mydata$math 5.383 
# CAVEAT: verify the tobit_data that the columns are correctly encoded!
# first, I got different values but now also the coeffs for general and vocational match
# mydata$general: -13.741, mydata$vocational: -48.835


####
# 2) using a censored model:
# but now I have the same sigma? Does this make sense??
# yes - in the paper, the distinction between ε_{it} ~ normal(0,σ^2) 
# and θ^m_{it} ~ normal(0, δ^2_m) is clearly made.

# can I specify U directly or does it have to be included in data? -> both possible.
censored_model = StanModel(file=Path('models/tobit_students_explicit.stan').open())

censored_dict_excluded = {'X': tobit_data[not_800][predictors], 
                          'N': tobit_data.shape[0] - N_cens, 
                          'y': tobit_data[not_800]['apt'], 'N_cens': N_cens, 
                          'K': len(predictors), 'X_cens': tobit_data[is_800][predictors], 
                          'y_cens': tobit_data[is_800]['apt']}
censored_fit = censored_model.sampling(data=censored_dict_excluded, iter=20000, chains=4)
censored_res = censored_fit.extract()
al_2 = censored_res['alpha'][5001:].mean()
beta_2 = censored_res['beta'][5001:].mean(axis=0)

# nice - this looks quite close to the values from the tutorial:
# Intercept:  209.5488
# mydata$read: 2.6980, mydata.math: 5.9148

# if I include all original points and run the model specified before:
# Intercept: 198.18
# read: 2.72, math: 6.15

# without them it's better. Need to filter the out the censored values!:
# Intercept: 208.666
# read: 2.69, math: 5.93
####

# trying out if I can use a loop instead of two matrices s.t. it will work smoothly
# with an adjacency matrix:
censored_loop_model = StanModel(file=Path('models/tobit_students_ifelse.stan').open())
censored_loop_fit = censored_loop_model.sampling(data=censored_dict, iter=20000, 
                                                 chains=4)
az.plot_trace(censored_loop_fit)
az.plot_energy(censored_loop_fit)

cens_loop_res = censored_loop_fit.extract()
al_loop = cens_loop_res['alpha'][5001:].mean()
beta_loop = cens_loop_res['beta'][5001:].mean(axis=0)
# yay works. intercept: 208.6, read: 2.70, math: 5.93, gen: -12.75, voc: -46.6


####
# Let's now try with the cumulative distribution function.
# OK so I've learned the following:
# - if no lower/ upper bounds are given for the parameters, stan will start around 0
#   usually +/- 2
# real normal_lccdf(reals y | reals mu, reals sigma)
# - if the (y-mu)/sigma < -37.5 or > 8.25, the will be an over/ underflow
cens_cum_model = StanModel(file=Path('models/tobit_students_cumulative.stan').open())
cens_cum_model.diagnose()
cens_cum_dict = censored_dict.copy()
del cens_cum_dict['y_cens']
del cens_cum_dict['X_cens']
cens_cum_dict['U'] = 800.0
init_test = [{'alpha': 240, 'beta': [2.5, 5.4, -13, -48], 'sigma':50}] * 4
cens_cum_fit = cens_cum_model.sampling(data=cens_cum_dict, iter=20000, chains=4, init=init_test)
cens_cum_res = cens_cum_fit.extract()
al_3 = cens_cum_res['alpha'][5001:].mean()
beta_3 = cens_cum_res['beta'][5001:].mean(axis=0)
# fails due to:
#   Log probability evaluates to log(0), i.e. negative infinity.
#   Stan can't start sampling from this initial value.

# I could maybe use Phi_approx?? but no idea how.
# would it help to specify a prior for alpha and beta? could run OLS first 
# and orient towards these values. But when initialising them towards these values,
# the model runs, spams some warnings and gives completely different estimates.
#####



# in the paper, there is in addition to different covariates:
# 1) a temporal relation (same coefficients for different t)
# 2) a spatial relation (same coefficients for different locations)

#####
# 1) let's simulate data for 3 years from that school where the students get sligthly 
#    dumber on average, but the same amount of 800ers
year_list = []
T=3
indices_800 = tobit_data[tobit_data['apt'] == 800].index
for t in range(T):
    df_copy = tobit_data.copy()
    df_copy['apt'] = tobit_data['apt'] - np.round(t + t*np.random.rand(tobit_data.shape[0]))
    df_copy['read'] = tobit_data['read'] - np.round(t + t/2*np.random.rand(tobit_data.shape[0]))
    df_copy['math'] = tobit_data['math'] - np.round(t + t/2*np.random.rand(tobit_data.shape[0]))
    if tobit_data['apt'].max() > 800:
        print('bad')
    # just ensure we still have some vals
    for i in indices_800:
        df_copy.loc[i, 'apt'] = 800
    year_list.append(df_copy)
    
# the documentation gives some hints on how to iterate through n-dim structures
# - matrices should be used when doing matrix calculation, else build arrays
# - vectorisation is better than loops (but I guess I need some loops here)
    
# now this becomes difficult with splitting censored from uncensored :(
# should maybe try the other approach with the normal_lccdf
temp_model = StanModel(file=Path('models/tobit_stud_temp_same_dim.stan').open())

N_cens = 17
T = len(year_list)
K = len(predictors)
N_T = list()
N_cens_T = list()
y_T = list()
X_T = np.zeros((T, 200, K))
X_cens_T = np.zeros((T,17,K))
y_cens_T = list()

# this is now basically cheating - I have the same number of censored and uncensored
# vars for each year. But otherwise I can't use matrix-arrays and would need to specify
# all the vars individually
for i in range(T):
    year_data = year_list[i]
    not_800 = year_data['apt'] != 800
    is_800 = year_data['apt'] == 800
    N_cens = is_800.sum()
    X_T[i] = year_data[predictors].to_numpy()
    N_T.append(year_data.shape[0])
    y_T.append(year_data['apt'])
    N_cens_T.append(N_cens)
    X_cens_T[i] = year_data[is_800][predictors].to_numpy()
    y_cens_T.append(year_data[is_800]['apt'])
temp_dict = {'X': X_T, 'N': N_T, 'y': y_T, 'N_cens': N_cens_T, 'K': K, 'T': T, 
             'X_cens': X_cens_T, 'y_cens': y_cens_T}

# can I just stack everything into one array for the different values of t?
# -> for my toy example yes, but I'll get into trouble with the location-specific
#    stuff. I need to be able to work on the whole dataset properly, can't just do
#    the split. 

temp_fit = temp_model.sampling(data=temp_dict, iter=20000, chains=4)
temp_res = temp_fit.extract()
intercept_temp = temp_res['beta_0'][5001:].mean()
beta_temp = temp_res['beta'][5001:].mean(axis=0)
# good - as expected, the values are similar as before, but slightly lower because
# the students have gotten more dumb


##
# some debugging - unfortunately, I can't store matrices of different size in one
# array in numpy and thereby also not in stan.
## 
test = year_data[not_800][predictors]
np_test = test.to_numpy() # looks good

# OK - here it would work. But I have the problem that the dimensions don't match
# and _for sure_ they won't match when working with real data...
np.issubdtype(np.asarray( temp_dict['X']).dtype, np.float64) # -> geht nicht.

test_2 = np.empty((5,200,4))
np.issubdtype(np.asarray( test_2).dtype, np.float64)

one_arr = np.array([[1,2,3], [4,5,6]])
next_arr = np.array([[1,2,3], [4,5,6], [7,8,9]])
arr_list = [one_arr, next_arr]
np.asarray(arr_list) # works, but becomes dtype=object. STAN needs int or float.
##
#
##
#####


# In order to proceed, I need to:
# a) specify the data explicitly - one for every t.
# b) work around the limitation by not splitting explicity
# c) choose the largest vals for each and fill up with zeros?
# d) simply define T=1 :)
# e) define and use my own pdf

# I'm choosing d) now.

# next steps would be:
# - now consider each individual student as being one road segment.
# - simulate some friendships/ study groups and assume that people in the same group
# - based on this, incorporate the CAR prior.

ad_matrix = np.zeros((200,200), dtype=int)

# let's make it very obvious and put some of the best ones and some of the
# worst ones into friends groups
friend_groups = []
for prog in coded_progs:
    sub_df = tobit_data[tobit_data['prog']==prog].sort_values('apt')
    curr_len = sub_df.shape[0]
    # split into 5 friend groups by their grades
    for i in range(5):
        curr_group = []
        for j in range(int(i*curr_len/5), int((i+1)*curr_len/5)):
            curr_group.append(sub_df.iloc[j]['id'] - 1)
        friend_groups.append(curr_group)

for group_list in friend_groups:
    for i in group_list:
        for j in group_list:
            if i!=j:
                ad_matrix[i,j] = 1

# and few some random friends. Observation: need to ensure that all are connected
# same applies to the road segments...
num_friends = 0
while num_friends < 50:
    i = np.random.randint(0,200)
    j = np.random.randint(0,200)
    if i != j:
        ad_matrix[i,j] = 1
        ad_matrix[j,i] = 1
        num_friends += 1
ad_graph = nx.Graph(ad_matrix)
if not nx.is_connected(ad_graph):
    raise(Exception('ERR: Adjacency matrix not connected! try again.'))

# - In the model of the researchers, phi is distributed around phi_bar
#   is this handled by the multi_normal_prec??? Need to understand docs and adjust if not.
#     - seems to be legit. Documentation of WinBUGS does it in a similar way.
#https://mc-stan.org/docs/2_19/functions-reference/multivariate-normal-distribution-precision-parameterization.html
# - find out what the CAR prior in car.normal is. Right now I just have 2/-2 ...
#   - Unfortunately, there is no information available. Just need to set something that works.
car_model = StanModel(file=Path('models/tobit_car_students.stan').open())
car_dict = censored_dict.copy()
car_dict['W'] = ad_matrix
car_dict['U'] = 800
#car_fit = car_model.sampling(data=car_dict, iter=20000, warmup=4000, chains=4, max_treedepth=20)

# this smaller run still took 45 mins to sample...
# And still getting too low E-BFMI values
car_fit2 = car_model.sampling(data=car_dict, iter=4000, warmup = 500, chains=4)
dump(car_fit2, Path('data/car_students_4000.joblib'))

az.plot_trace(car_fit2)

car_res = car_fit2.extract()
car_res['beta_zero'][501:].mean()
car_res['beta'][501:].mean(axis=0)

# getting many rejections - bad? Phi is a bit like a covariance matrix
# -> only in the beginning, after 200 iterations all fine.
# result from the run: chains have not mixed, might need to re-parametrize...
# but let's try the same metric as in the paper:
for res_dict in [cens_loop_res, car_res]:
    print(pd.Series(res_dict['sigma'][10001:]).describe())
# bad: compared with the previous tobit model, the parameters are nearly the same and
# the sigma's std has not been reduced significantly
# am I contraining the variables too much??? Need to center somehow?
"""
WARNING:pystan:Chain 1: E-BFMI = 0.0426
WARNING:pystan:Chain 2: E-BFMI = 0.0434
WARNING:pystan:Chain 3: E-BFMI = 0.0431
WARNING:pystan:Chain 4: E-BFMI = 0.0454
WARNING:pystan:E-BFMI below 0.2 indicates you may need to reparameterize your model
"""
    
sparse_model = StanModel(file=Path('models/sparse_tobitcar_students.stan').open())
tobit_data['ones'] = np.ones(tobit_data.shape[0])
new_preds = ['ones'] + predictors
sparse_dict = {'X': tobit_data[new_preds], 'n': tobit_data.shape[0], 
               'y': tobit_data['apt'], 'n_cens': N_cens,
               'p': len(new_preds), 'y_cens': tobit_data[is_800]['apt'],
               'W': ad_matrix, 'U': 800, 'W_n': ad_matrix.sum()//2} # same as len(ad_graph.edges)
sparse_fit = sparse_model.sampling(sparse_dict, iter=4000, warmup=500, chains=4)
print(sparse_fit.stansummary())
# higher values for read and math that with usual tobit model.
# but: still the same  sigma and std(sigma) as before :/
# it's weird, I'd expect more variance in the model explained by the spatial effects
# is it due to no demeaning etc? 

##
# will try with values closer to 0 now.
# sigma was  67.3  with stdev 3.74
# even worse - E-BMFI is still small, but now also much treedepth saturation (OK)
# and chain divergence (bad!)
# would need to check energy-plots and what correlates...
# TODO: if I scale, I have the danger of missing not hitting the condition for U...
# -> should not be a problem if I have zeros there as lower bound
trans = MaxAbsScaler().fit_transform(tobit_data[new_preds + ['apt']])
data_centered = pd.DataFrame(trans, columns=new_preds + ['apt'])
c_sparse_dict = {'X': data_centered[new_preds], 'n': tobit_data.shape[0], 
                 'y': data_centered['apt'], 'n_cens': N_cens,
                 'p': len(new_preds), 'y_cens': data_centered[is_800]['apt'],
                 'W': ad_matrix, 'U': 1, 'W_n': ad_matrix.sum()//2} 
c_sp_model = StanModel(file=Path('models/sparse_tobitcar_students.stan').open(), 
                       verbose=False) # extra_compile_args=["-w"]
c_params = {'adapt_delta': 0.90, 'max_treedepth':15}
# no more saturation, but still divergence...
c_sp_fit = c_sp_model.sampling(c_sparse_dict, iter=4000, warmup=500, control=c_params)
c_sp_res = c_sp_fit.extract()
print(c_sp_fit.stansummary())
az.plot_trace(c_sp_fit, compact=True)
az.plot_pair(c_sp_fit, ['tau', 'alpha', 'sigma'], divergences=True)
# seems like I'm having a lot of divergences where:
# - sigma below 0.0025
# - alpha > 0.99 (would imply IAR)

az.plot_energy(c_sp_fit)
plt.scatter(c_sp_fit['lp__'], c_sp_fit['sigma'])
dump(c_sp_fit, 'data/c_sp_4000.joblib')
# sigma looks very correlated. But where do I get the energy from??


# y_cens look ok though
# tau quite large -> 23, close to the largest „friend_group“
# larger phis now :) in negative and positive!




















