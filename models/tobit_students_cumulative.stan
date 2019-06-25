// this model compiles, but doesn't do the job :(
data {
    int<lower=0> n; // number of data items
    int<lower=0> p; // number of predictors
    matrix[n, p] X; // predictor matrix
    vector[n] y;  // all observed variables
    int<lower=0> n_cens; // number of censored variable
    real<lower=max(y)> U; // censoring point
}
parameters {
    vector[p] beta;
    real<lower=0> sigma;
}
model {
    y ~ normal(X * beta, sigma);
    target += n_cens * (normal_lccdf(U | X * beta, sigma));
}