data {
    int<lower=2> n; // number of all data rows
    int<lower=1> p; // number of predictors
    matrix[n, p] X; // full predictor matrix
    vector[n] y;  // all observed variables
    int<lower=1> n_cens; // number of censored variables
    real U; // censoring point. account for rounding errors.
}
parameters {
    vector[p] beta;
    real<lower=0> sigma;
    vector<upper=U>[n_cens] y_cens; // censored vars as sampled parameter
}
model {
    int j = 1;
    for (i in 1:n){
        if(y[i] < U){
            y_cens[j] ~ normal(X[i] * beta, sigma);
            j += 1;
        } else {
            y[i] ~ normal(X[i] * beta, sigma);
        }
    }
}