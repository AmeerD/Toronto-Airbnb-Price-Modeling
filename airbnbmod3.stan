/* Airbnb Room Type Model */
data {
  int<lower=1> N;              // number of observations
  int<lower=1> J;              // number of room types
  int<lower=1> K;              // number of covariates with constant slopes
  int<lower=1> L;              // number of covariates with varying slopes
  int<lower=1,upper=J> room[N];// room type membership
  vector[K] xi[N];             // matrix of covariate data with constant slope
  vector[L] zi[N];             // matrix of covariate data with varying slope
  vector[N] yi;                // log price data
}
parameters {
  vector[K] beta;
  vector[L+1] mu;                // alpha and gamma prior means
  corr_matrix[L+1] Omega;        // prior correlation - L+1 is for alpha & gamma 
  vector<lower=0>[L+1] tau;      // prior scale 
  vector[L+1] ag[J];             // alpha is the intercept and gamma terms combined for convenience
  real<lower=0> sigma_y;
}
model {
  vector[N] y_hat;
  for (i in 1:N)
    y_hat[i] = ag[room[i]][1] + dot_product(ag[room[i]][2:L+1],zi[i]) + dot_product(beta,xi[i]);
  
  // priors
  mu ~ normal(0,1);
  beta ~ normal(0,1);
  Omega ~ lkj_corr(2); // prior selected from stan recommendations
  tau ~ cauchy(0,2.5); // prior selected from stan recommendations
  sigma_y ~ normal(0,1);
  
  // data generation
  ag ~ multi_normal(mu, quad_form_diag(Omega, tau));
  yi ~ normal(y_hat, sigma_y);
}
generated quantities {
  vector[N] log_lik;        // pointwise log-likelihood for LOO
  vector[N] log_price_rep;  // replications from posterior predictive dist

  for (n in 1:N) {
    real log_price = ag[room[n]][1] + dot_product(ag[room[n]][2:L+1],zi[n]) + dot_product(beta,xi[n]);
    log_lik[n] = normal_lpdf(yi[n] | log_price, sigma_y);
    log_price_rep[n] = normal_rng(log_price, sigma_y);
  }
}
