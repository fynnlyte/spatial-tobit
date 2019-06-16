from pystan import StanModel
import numpy as np
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

