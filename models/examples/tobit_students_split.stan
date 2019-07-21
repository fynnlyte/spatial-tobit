data {
    int<lower=0> n; // number of data items (uncensored
    int<lower=0> p; // number of predictors
    matrix[n, p] X; // predictor matrix (uncensored)
    vector[n] y;  // observed variables (uncensored)
    int<lower=0> n_cens; // number of censored variables
    matrix[n_cens, p] X_cens; // predictor matrix (censored)
    real<lower=max(y)> U; // censoring point
}
parameters {
    vector[p] beta;
    real<lower=0> sigma;
    vector<lower=U>[n_cens] y_cens; // censored vars as sampled parameter
}
model {
    y ~ normal(X * beta, sigma);
    y_cens ~ normal(X_cens * beta, sigma); // and likelihood
}