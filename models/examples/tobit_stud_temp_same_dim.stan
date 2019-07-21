// This model works only if the number of censored and uncensored vars is the same over t
data {
    int<lower=0> T; // period t = 1,...,T
    int<lower=0> K; // number of predictors
    int<lower=0> N[T]; // uncensored vars per time
    int<lower=0> N_cens[T]; // cens vars per time
    vector[N[T]] y[T];  // arr of X for uncensored
    matrix[N[T], K] X[T]; // arr of X for uncensored
    matrix[N_cens[T], K] X_cens[T]; // arr of X for censored
}
parameters {
    real beta_0;
    vector[K] beta;
    real<lower=0> sigma;
    vector<lower=800>[N_cens[T]] y_cens[T];
}
model {
    for (t in 1:T)
      y[t] ~ normal(X[t] * beta + beta_0, sigma);
    for (t in 1:T)
      y_cens[t] ~ normal(X_cens[t] * beta + beta_0, sigma);
}