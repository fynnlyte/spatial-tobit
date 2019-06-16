from pystan import StanModel
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import numpy as np
import pandas as pd

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
tobit_data = pd.read_csv("https://stats.idre.ucla.edu/stat/data/tobit.csv")
# Stan does not support categorical features - need dummy variables.
cat_trans = Pipeline(steps=[('onehot', OneHotEncoder())])
ct = ColumnTransformer([('cat', cat_trans, ['prog'])])
coded_progs = ['vocational', 'general','academic']
trans_progs = pd.DataFrame(ct.fit_transform(tobit_data), columns=coded_progs)
tobit_data = pd.concat([tobit_data, trans_progs], axis=1)


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
    real<lower=0> sigma; // error scale
}
model {
    y ~ normal(X * beta + alpha, sigma); // likelihood
}
"""
tobit_datadict = {'y': tobit_data['apt'], 'N': tobit_data.shape[0], 'K': 5,
                  'X': tobit_data[['read', 'math'] + coded_progs]}
tobit_linear_model = StanModel(model_name='tobit_linear', model_code=tobit_linear_code)
tob_lin_fit = tobit_linear_model.sampling(data=tobit_datadict)
tob_lin_res = tob_lin_fit.extract()































