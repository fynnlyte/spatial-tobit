// Need to verify that the matrix simplification for phi applies
data {
    int<lower=1> N; // number of data items
    int<lower=1> K; // number of predictors
    matrix[N, K] X; // predictor matrix (uncensored)
    vector<lower=0>[N] y;  // observed variables
    int<lower=0> N_cens; // number of censored variables
    matrix[N_cens, K] X_cens; // predictor matrix (censored)
    real<lower=max(y)> U;
    matrix<lower=0,upper=1>[N,N] W;
}
transformed data {
    vector[N] zeros;
    matrix<lower=0>[N,N] D;
    {
        vector[N] W_rowsums;
        for (i in 1:N){
            W_rowsums[i] = sum(W[i,]);
        }
        D = diag_matrix(W_rowsums);
    }
    zeros = rep_vector(0,N);
}
parameters {
    real<lower=0> tau;
    vector[N] phi;
    real<lower=0, upper=1> alpha; // spatial dep gets a uniform [0,1] prior
    real beta_zero;
    vector[K] beta;
    real<lower=0> sigma;
    vector<lower=U>[N_cens] y_cens; // censored vars as sampled parameter
}
model {
    int j = 1;
    phi ~ multi_normal_prec(zeros, tau * (D - alpha * W));
    tau ~ gamma(2,2);
    for (i in 1:N){
        if(y[i] == 800){
            y_cens[j] ~ normal(X[i] * beta + beta_zero + phi[i], sigma);
            j += 1;
        } else {
            y[i] ~ normal(X[i] * beta + beta_zero + phi[i], sigma);
        }
    }
}