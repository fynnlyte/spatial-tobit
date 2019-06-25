data {
    int<lower=0> n; // number of all data rows
    int<lower=0> p; // number of predictors
    matrix[N, K] X; // full predictor matrix
    vector[N] y;  // all observed variables
    int<lower=0> N_cens; // number of censored variables
    real<lower = min(y)> U; // censoring point. should account for rounding errors.
}
parameters {
    real alpha;
    vector[p] beta;
    real<lower=0> sigma;
    vector<upper=0>[N_cens] y_cens; // censored vars as sampled parameter
}
model {
    int j = 1;
    for (i in 1:n){
        if(y[i] < U){
            y_cens[j] ~ normal(X[i] * beta + alpha, sigma);
            j += 1;
        } else {
            y[i] ~ normal(X[i] * beta + alpha, sigma);
        }
    }
}