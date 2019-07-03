data {
    int<lower=0> n_obs; // number of uncensored rows
    int<lower=0> n_cens; // number of censored rows
    int<lower=0> p; // number of predictors
    int<lower=1, upper = n_obs + n_cens> ii_obs[n_obs]; // indices of observed
    int<lower=1, upper = n_obs + n_cens> ii_cens[n_cens]; // indices of censored
    vector[n_obs] y_obs;  // all uncensored variables
    real<lower=0,upper = min(y_obs)> U; // censoring point, accounting for rounding errors.
    matrix[n_cens+n_obs, p] X; // predictor matrix (full)
}
transformed data {
   int<lower = 0> n = n_obs + n_cens;
   matrix[n, p] Q_ast;
   matrix[p, p] R_ast;
   matrix[p, p] R_ast_inverse;
   // thin and scale the QR decomposition
   Q_ast = qr_Q(X)[, 1:p] * sqrt(n - 1);
   R_ast = qr_R(X)[1:p, ] / sqrt(n - 1);
   R_ast_inverse = inverse(R_ast);
}
parameters {
    vector[p] theta; // coefficients on Q_ast
    real beta_zero; // intercept
    real<lower=0> sigma;
    vector<upper=U>[n_cens] y_cens; // censored vars as sampled parameter
}
transformed parameters {
    vector[n] y;
    y[ii_obs] = y_obs;
    y[ii_cens] = y_cens;
}
model {
    sigma ~ inv_gamma(0.001, 0.001); // trying this as prior.
    y ~ normal(Q_ast * theta + beta_zero, sigma);
}
generated quantities{
    vector[p] beta;
    beta = R_ast_inverse * theta; // coefficients on X
}