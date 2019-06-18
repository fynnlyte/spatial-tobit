from pystan import StanModel
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import numpy as np
import pandas as pd
from joblib import dump, load
from urllib.request import urlretrieve

model_code = 'parameters {real y;} model {y ~ normal(0,1);}'
model = StanModel(model_code=model_code)
y = model.sampling().extract()['y']
print('If this worked, you will see a value near 0 now:')
print(y.mean())


# Example from „Kruschke: Doing Bayesian Data Analysis”
coin_code = """
data {
    int<lower=0> N;
    int y[N]; // vec of N ints
}
parameters {
    real<lower=0, upper=1> theta;
}
model {
    theta ~ beta(1,1);
2    y ~ bernoulli(theta);
}
"""
# create the DSO (dynamic shared object)
coin_model = StanModel(model_code=coin_code, model_name='coin')
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
linear_code = """
data {
    int<lower=0> N;
    vector[N] x;
    vector[N] y;
}
parameters {
    real alpha;
    real beta;
    real<lower=0> sigma;
}
model {
    y ~ normal(alpha + beta * x, sigma);
}
"""
linear_model = StanModel(model_name='linear', model_code=linear_code)
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
file_location = 'data/ucla_tobit_example.csv'
try:
    tobit_data = pd.read_csv(file_location)
except:
    urlretrieve("https://stats.idre.ucla.edu/stat/data/tobit.csv", file_location)
    tobit_data = pd.read_csv(file_location)
# Stan does not support categorical features - need dummy variables.
cat_trans = Pipeline(steps=[('onehot', OneHotEncoder())])
ct = ColumnTransformer([('cat', cat_trans, ['prog'])])
coded_progs = ['academic', 'general','vocational']
trans_progs = pd.DataFrame(ct.fit_transform(tobit_data), columns=coded_progs)
tobit_data = pd.concat([tobit_data, trans_progs], axis=1)
predictors = ['read', 'math', 'general', 'vocational']



# 1) as comparison, do a linear model:
tobit_linear_code = """
data {
    int<lower=0> N; // number of data items
    int<lower=0> K; // number of predictors
    matrix[N, K] X; // predictor matrix
    vector[N] y;  // outcome vector
}
parameters {
    real alpha; // intercept
    vector[K] beta;  // coefficients for predictors
    real<lower=0> sigma ; //  error scale
}
model {
    y ~ normal(X * beta + alpha, sigma); // likelihood
}
"""
tobit_datadict = {'y': tobit_data['apt'], 'N': tobit_data.shape[0], 'K': len(predictors),
                  'X': tobit_data[predictors]}

tobit_linear_model = StanModel(model_name='tobit_linear', model_code=tobit_linear_code)
tob_lin_fit = tobit_linear_model.sampling(data=tobit_datadict, iter=50000, chains=4)
tob_lin_res = tob_lin_fit.extract()

al = tob_lin_res['alpha'][25001:].mean()
beta = tob_lin_res['beta'][25001:].mean(axis=0)
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

# can I specify U directly or does it have to be included in data? -> must include.
censored_code_implicit = """
data {
    int<lower=0> N; // number of data items
    int<lower=0> K; // number of predictors
    matrix[N, K] X; // predictor matrix (uncensored)
    vector[N] y;  // observed variables
    int<lower=0> N_cens; // number of censored variables
    matrix[N_cens, K] X_cens; // predictor matrix (censored)
    real<lower=max(y)> U;
}
parameters {
    real alpha;
    vector[K] beta;
    real<lower=0> sigma;
    vector<lower=U>[N_cens] y_cens; // censored vars as sampled parameter
}
model {
    y ~ normal(X * beta + alpha, sigma);
    y_cens ~ normal(X_cens * beta + alpha, sigma); // and likelihood
}
"""

censored_code_explicit = """
data {
    int<lower=0> N; // number of data items
    int<lower=0> K; // number of predictors
    matrix[N, K] X; // predictor matrix (uncensored)
    vector[N] y;  // observed variables
    int<lower=0> N_cens; // number of censored variables
    matrix[N_cens, K] X_cens; // predictor matrix (censored)
}
parameters {
    real alpha;
    vector[K] beta;
    real<lower=0> sigma;
    vector<lower=800>[N_cens] y_cens; // censored vars as sampled parameter
}
model {
    y ~ normal(X * beta + alpha, sigma);
    y_cens ~ normal(X_cens * beta + alpha, sigma); // and likelihood
}
"""


censored_model = StanModel(model_code=censored_code_explicit)
not_800 = tobit_data['apt'] != 800
is_800 = tobit_data['apt'] == 800
N_cens = is_800.sum()

censored_dict_excluded = {'X': tobit_data[not_800][predictors], 
                          'N': tobit_data.shape[0] - N_cens, 
                          'y': tobit_data[not_800]['apt'], 'N_cens': N_cens, 
                          'K': len(predictors), 'X_cens': tobit_data[is_800][predictors], 
                          'y_cens': tobit_data[is_800]['apt']}
censored_fit = censored_model.sampling(data=censored_dict_excluded, iter=50000, chains=4)
censored_res = censored_fit.extract()
al_2 = censored_res['alpha'][25001:].mean()
beta_2 = censored_res['beta'][25001:].mean(axis=0)

# nice - this looks quite close to the values from the tutorial:
# Intercept:  209.5488
# mydata$read: 2.6980, mydata.math: 5.9148

# if I include all original points:
# Intercept: 198.18
# read: 2.72, math: 6.15

# without them it's better:
# censored_dict = {'X': tobit_data[predictors], 'N': tobit_data.shape[0], 
                 'y': tobit_data['apt'], 'N_cens': N_cens,
                 'K': len(predictors), 'X_cens': tobit_data[is_800][predictors], 
                 'y_cens': tobit_data[is_800]['apt']}Intercept: 208.666
# read: 2.69, math: 5.93
####

####
# Let's now try with the cumulative distribution function.
# OK so I've learned the following:
# - if no lower/ upper bounds are given for the parameters, stan will start around 0
#   usually +/- 2
# real normal_lccdf(reals y | reals mu, reals sigma)
# - if the (y-mu)/sigma < -37.5 or > 8.25, the will be an over/ underflow
censored_cum = """
data {
    int<lower=0> N; // number of data items
    int<lower=0> K; // number of predictors
    matrix[N, K] X; // predictor matrix
    vector[N] y;  // all observed variables
    int<lower=0> N_cens; // number of censored variable
    real<lower=max(y)> U; // censoring point
}
parameters {
    real alpha;
    vector[K] beta;
    real<lower=0> sigma;
}
model {
    y ~ normal(X * beta + alpha, sigma);
    target += N_cens * (normal_lccdf(U | X * beta + alpha, sigma));
}
"""
cens_cum_model = StanModel(model_code=censored_cum)
cens_cum_model.diagnose()
cens_cum_dict = censored_dict.copy()
del cens_cum_dict['y_cens']
del cens_cum_dict['X_cens']
cens_cum_dict['U'] = 800.0
init_test = [{'alpha': 240, 'beta': [2.5, 5.4, -13, -48], 'sigma':50}] * 4
cens_cum_fit = cens_cum_model.sampling(data=cens_cum_dict, iter=50000, chains=4, init=init_test)
cens_cum_res = cens_cum_fit.extract()
al_3 = cens_cum_res['alpha'][40000:].mean()
beta_3 = cens_cum_res['beta'][40000:].mean(axis=0)
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
tobit_temporal = """
data {
    int<lower=0> T; // period t = 1,...,T
    int<lower=0> K; // number of predictors
    int<lower=0> N[T]; // uncensored vars per time
    int<lower=0> N_cens[T]; // cens vars per time
    vector[N[T]] y[T];  // arr of X for uncensored
    matrix[N[T], K] X[T]; // arr of X for uncensored
    matrix[N_cens[T], K] X_cens[T]; // arr of X for censored
}
parameters {
    real beta_0;
    vector[K] beta;
    real<lower=0> sigma;
    vector<lower=800>[N_cens[T]] y_cens[T];
}
model {
    for (t in 1:T)
      y[t] ~ normal(X[t] * beta + beta_0, sigma);
    for (t in 1:T)
      y_cens[t] ~ normal(X_cens[t] * beta + beta_0, sigma);
}
"""
temp_model = StanModel(model_code=tobit_temporal)

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

temp_fit = temp_model.sampling(data=temp_dict, iter=50000, chains=4)
temp_res = temp_fit.extract()
intercept_temp = temp_res['beta_0'][25001:].mean()
beta_temp = temp_res['beta'][25001:].mean(axis=0)
# good - as expected, the values are similar as before, but slightly lower because
# the students have gotten more stupid


##
# some debuggin - unfortunately, I can't store matrices of different size in one
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
friend_groups = [[186, 168, 70, 24, 30, 91, 156, 61, 148, 194], [174, 133, 128, 9, 62],
                 [39, 166, 114, 122, 86, 35, 154], [34,84,129,28,85], [6,25, 67,153, 187],
                 [58,32,101,134,167,76,111], [10,47,54,77,190,198], [161, 43, 90, 152, 63, 97],
                 [139,14,105,137,1,144,44,66], [149,147,116,140,82,36,65]]
for group_list in friend_groups:
    for i in group_list:
        for j in group_list:
            if i!=j:
                ad_matrix[i,j] = 1

# and some random friends
for _ in range(1000):
    i = np.random.randint(0,200)
    j = np.random.randint(0,200)
    ad_matrix[i,j] = 1
    ad_matrix[j,i] = 1
    

car_code = """
data {
    int<lower=1> N; // number of data items
    int<lower=1> K; // number of predictors
    matrix[N, K] X; // predictor matrix (uncensored)
    vector<lower=0>[N] y;  // observed variables
    int<lower=0> N_cens; // number of censored variables
    matrix[N_cens, K] X_cens; // predictor matrix (censored)
    real<lower=max(y)> U;
    matrix<lower=0,upper=1>[N,N] W;
}
transformed data {
    vector[N] zeros;
    matrix<lower=0>[N,N] D;
    {
        vector[N] W_rowsums;
        for (i in 1:N){
            W_rowsums[i] = sum(W[i,]);
        }
        D = diag_matrix(W_rowsums);
    }
    zeros = rep_vector(0,N);
}
parameters {
    real<lower=0> tau;
    vector[N] phi;
    real<lower=0, upper=1> alpha; // spatial dep gets a uniform [0,1] prior
    real beta_zero;
    vector[K] beta;
    real<lower=0> sigma;
    vector<lower=U>[N_cens] y_cens; // censored vars as sampled parameter
}
model {
    phi ~ multi_normal_prec(zeros, tau * (D - alpha * W));
    tau ~ gamma(2,2);
    y ~ normal(X * beta + beta_zero + phi, sigma); // todo: this currently includes all values :(
    y_cens ~ normal(X_cens * beta + beta_zero, sigma); // todo: I'm lacking phi here :(
}
"""
# todo: 
# - fuck, now I have the dimensionality problem with the censored vars again, but at
#   a different point... using all for y and no phi for y_cens but I know it's wrong
#   - or I somehow need the information about the indices and specify y[i] and y_cens[i]
#     via for-loops and if/else for what to use :)
# - In my model, phi is distributed around phi_bar
#   is this handled by the multi_normal_prec???
# https://mc-stan.org/docs/2_19/functions-reference/multivariate-normal-distribution-precision-parameterization.html
# - find out what the CAR prior in car.normal is
car_model = StanModel(model_code = car_code)
car_dict = censored_dict.copy()
car_dict['W'] = ad_matrix
car_dict['U'] = 800
car_fit = car_model.sampling(data=car_dict)
car_res = car_fit.extract()
dump(car_res, 'data/car_result.joblib')
can_load = load('data/car_result.joblib')
# getting many rejections - bad? Phi is a bit like a covariance matrix
# -> only in the beginning, after 200 iterations all fine.
# result from my run: chains have not mixed, might need to re-parametrize...





















