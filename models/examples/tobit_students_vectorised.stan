// vectorising the whole thing
data {
    int<lower=0> n_obs; // number of uncensored items
    int<lower=0> n_cens; // number of censored variable
    int<lower=0> p; // number of predictors
    int<lower=1, upper = n_obs + n_cens> ii_obs[n_obs]; // indices of observed
    int<lower=1, upper = n_obs + n_cens> ii_cens[n_cens]; // indices of censored
    vector[n_obs] y_obs;  // all uncensored variables
    real<lower=max(y_obs)> U; // censoring point
    matrix[n_cens+n_obs, p] X; // predictor matrix (all values)
//    matrix[n_cens, p] X_cens; // added
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
    target += normal_lpdf(y| X * beta, sigma); 
//      y ~ normal(X * beta, sigma); // above is an equivalent formulation w/o warning
}
