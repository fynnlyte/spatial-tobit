// vectorising and using QR decomposition
data {
    int<lower=0> n_obs; // number of uncensored items
    int<lower=0> n_cens; // number of censored variable
    int<lower=0> p; // number of predictors
    int<lower=1, upper = n_obs + n_cens> ii_obs[n_obs]; // indices of observed
    int<lower=1, upper = n_obs + n_cens> ii_cens[n_cens]; // indices of censored
    vector[n_obs] y_obs;  // all uncensored variables
    real<lower=max(y_obs)> U; // censoring point
    matrix[n_cens+n_obs, p] X; // predictor matrix (all values)
//    matrix[n_cens, p] X_cens; // added
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
    vector[p] theta;      // coefficients on Q_ast
    real<lower=0> sigma;
    vector<lower=U>[n_cens] y_cens;
}
transformed parameters {
    vector[n] y;
    y[ii_obs] = y_obs;
    y[ii_cens] = y_cens;
}
model {
    y ~ normal(Q_ast * theta, sigma); // todo: or do i need the intercept in this case???
}
generated quantities{
    vector[p] beta;
    beta = R_ast_inverse * theta; // coefficients on X
}
