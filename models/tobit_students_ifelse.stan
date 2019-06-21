data {
    int<lower=0> N; // number of all data rows
    int<lower=0> K; // number of predictors
    matrix[N, K] X; // full predictor matrix
    vector[N] y;  // all observed variables
    int<lower=0> N_cens; // number of censored variables
}
parameters {
    real alpha;
    vector[K] beta;
    real<lower=0> sigma;
    vector<lower=800>[N_cens] y_cens; // censored vars as sampled parameter
}
model {
    int j = 1;
    for (i in 1:N){
        if(y[i] == 800){
            y_cens[j] ~ normal(X[i] * beta + alpha, sigma);
            j += 1;
        } else {
            y[i] ~ normal(X[i] * beta + alpha, sigma);
        }
    }
}