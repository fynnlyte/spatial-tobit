// this model compiles, but doesn't do the job :(
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