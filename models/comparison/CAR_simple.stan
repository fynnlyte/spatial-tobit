data {
  int<lower=0> n_obs; // number of uncensored rows
  int<lower=0> n_cens; // number of censored rows
  int<lower=n_obs+n_cens, upper=n_obs+n_cens > n; // total no of rows
  int<lower = 1> p;
  int<lower=1, upper = n> ii_obs[n_obs]; // indices of observed
  int<lower=1, upper = n> ii_cens[n_cens]; // indices of censored
  vector[n_obs] y_obs;  // all uncensored variables
  matrix[n, p] X;
  int<lower = 0> y[n];
  vector[n] log_offset;
  matrix<lower = 0, upper = 1>[n, n] W;
}
transformed data{
  vector[n] zeros;
  matrix<lower = 0>[n, n] D;
  {
    vector[n] W_rowsums;
    for (i in 1:n) {
      W_rowsums[i] = sum(W[i, ]);
    }
    D = diag_matrix(W_rowsums);
  }
  zeros = rep_vector(0, n);
}
parameters {
  vector[p] beta;
  vector[n] phi;
  real<lower = 0> tau;
  real<lower = 0, upper = 0.99> alpha;
  vector<upper = U>[n_cens] y_cens;// to-be estimated censored values < 0
}
transformed parameters {
    vector[n] y;
    y[ii_obs] = y_obs;
    y[ii_cens] = y_cens;
}
model {
  phi ~ multi_normal_prec(zeros, tau * (D - alpha * W));
  tau ~ gamma(2, 2);
  y ~ normal(X * beta + phi, sigma);
}