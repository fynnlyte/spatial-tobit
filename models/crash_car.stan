functions {
  /**
  * Return the log probability of a proper conditional autoregressive (CAR) prior 
  * with a sparse representation for the adjacency matrix
  *
  * @param phi Vector containing the parameters with a CAR prior
  * @param tau Precision parameter for the CAR prior (real)
  * @param alpha Dependence (usually spatial) parameter for the CAR prior (real)
  * @param W_sparse Sparse representation of adjacency matrix (int array)
  * @param n Length of phi (int)
  * @param W_n Number of adjacent pairs (int)
  * @param D_sparse Number of neighbors for each location (vector)
  * @param lambda Eigenvalues of D^{-1/2}*W*D^{-1/2} (vector)
  *
  * @return Log probability density of CAR prior up to additive constant
  */
  real sparse_car_lpdf(vector phi, real tau, real alpha, 
    int[,] W_sparse, vector D_sparse, vector lambda, int n, int W_n) {
      row_vector[n] phit_D; // phi' * D
      row_vector[n] phit_W; // phi' * W
      vector[n] ldet_terms;
    
      phit_D = (phi .* D_sparse)';
      phit_W = rep_row_vector(0, n);
      for (i in 1:W_n) {
        phit_W[W_sparse[i, 1]] = phit_W[W_sparse[i, 1]] + phi[W_sparse[i, 2]];
        phit_W[W_sparse[i, 2]] = phit_W[W_sparse[i, 2]] + phi[W_sparse[i, 1]];
      }
    
      for (i in 1:n) ldet_terms[i] = log1m(alpha * lambda[i]);
      return 0.5 * (n * log(tau)
                    + sum(ldet_terms)
                    - tau * (phit_D * phi - alpha * (phit_W * phi)));
  }
}
data {
  int<lower=0> n_obs; // number of uncensored rows
  int<lower=0> n_cens; // number of censored rows
  int<lower=n_obs+n_cens, upper=n_obs+n_cens > n; // total no of rows
  int<lower=0> p; // number of predictors
  int<lower=1, upper = n> ii_obs[n_obs]; // indices of observed
  int<lower=1, upper = n> ii_cens[n_cens]; // indices of censored
  vector[n_obs] y_obs;  // all uncensored variables
  real<lower=0,upper = min(y_obs)> U; // censoring point, accounting for rounding errors.
  matrix[n, p] X; // predictor matrix (full)
  matrix<lower = 0, upper = 1>[n, n] W; // adjacency matrix
  int<lower = n, upper= n*n> W_n; // number of adjacent region pairs (i.e. edges in graph)
}
transformed data {
  matrix[n, p] Q_ast;
  matrix[p, p] R_ast;
  matrix[p, p] R_ast_inverse;
  int W_sparse[W_n, 2];   // adjacency pairs
  vector[n] D_sparse;     // diagonal of D (number of neigbors for each site)
  vector[n] lambda;       // eigenvalues of invsqrtD * W * invsqrtD
  // thin and scale the QR decomposition
  Q_ast = qr_Q(X)[, 1:p] * sqrt(n - 1);
  R_ast = qr_R(X)[1:p, ] / sqrt(n - 1);
  R_ast_inverse = inverse(R_ast);
  { // generate sparse representation for W
  int counter;
  counter = 1;
  // loop over upper triangular part of W to identify neighbor pairs
    for (i in 1:(n - 1)) {
      for (j in (i + 1):n) {
        if (W[i, j] == 1) {
          W_sparse[counter, 1] = i;
          W_sparse[counter, 2] = j;
          counter = counter + 1;
        }
      }
    }
  }
  for (i in 1:n) D_sparse[i] = sum(W[i]);
  {
    vector[n] invsqrtD;  
    for (i in 1:n) {
      invsqrtD[i] = 1 / sqrt(D_sparse[i]);
    }
    lambda = eigenvalues_sym(quad_form(W, diag_matrix(invsqrtD)));
  }
}
parameters {
  vector[p] theta; // coefficients on Q_ast
  real beta_zero; // intercept
  vector[n] phi;
  real<lower = 0> tau; 
  real<lower = 0, upper = 1> alpha; // spatial dependence
  real<lower = 0> sigma; // todo: might need some higher bound to avoid divergences
  vector<upper = U>[n_cens] y_cens; // to-be estimated censored values < 0
}
transformed parameters {
    vector[n] y;
    y[ii_obs] = y_obs;
    y[ii_cens] = y_cens;
}
model {
  sigma ~ inv_gamma(0.001, 0.001);
  tau ~ gamma(2, 2); // this is from CARstan
  phi ~ sparse_car(tau, alpha, W_sparse, D_sparse, lambda, n, W_n);
  y ~ normal(Q_ast * theta + beta_zero + phi, sigma);
}
generated quantities{
    vector[p] beta;
    beta = R_ast_inverse * theta; // coefficients on X
}
