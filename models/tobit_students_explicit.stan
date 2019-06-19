// here, the censoring value is not passed but specified in the code.
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