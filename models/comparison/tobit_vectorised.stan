data {
    int<lower=0> n_obs; // number of uncensored items
    int<lower=0> n_cens; // number of censored variable
    int<lower=0> p; // number of predictors
    int<lower=1, upper = n_obs + n_cens> ii_obs[n_obs]; // indices of observed
    int<lower=1, upper = n_obs + n_cens> ii_cens[n_cens]; // indices of censored
    vector[n_obs] y_obs;  // all uncensored variables
    real<lower=max(y_obs)> U; // censoring point
    matrix[n_cens+n_obs, p] X; // predictor matrix (all values)
}
transformed data {
   int<lower = 0> n = n_obs + n_cens;
}
parameters {
    vector[p] beta;
    real<lower=0> sigma;
    vector<lower=U>[n_cens] y_cens;
}
transformed parameters {
    vector[n] y;
    y[ii_obs] = y_obs;
    y[ii_cens] = y_cens;
}
model {
    sigma ~ inv_gamma(0.001, 0.001);
    y ~ normal(X * beta, sigma);
}
