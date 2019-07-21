data {
    int<lower=0> n; // number of all data rows
    int<lower=0> p; // number of predictors
    matrix[N, p] X; // full predictor matrix
    vector[N] y;  // all observed variables
    int<lower=0> n_cens; // number of censored variables
}
parameters {
    real beta_zero;
    vector[p] beta;
    real<lower=0> sigma;
    vector<uppper=0>[n_cens] y_cens; // censored vars as sampled parameter
}
model {
    sigma ~ inv_gamma(0.001, 0.001);
    int j = 1;
    for (i in 1:n){
        if(y[i] < 0.00000001){
            y_cens[j] ~ normal(X[i] * beta + beta_zero, sigma);
            j += 1;
        } else {
            y[i] ~ normal(X[i] * beta + beta_zero, sigma);
        }
    }
}
