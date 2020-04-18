/* Airbnb Room Type Model */
data {
  int<lower=1> N;       // number of observations
  int<lower=1> J;       // number of room types
  int<lower=1> K;       // number of covariates 
  int<lower=1,upper=J> room[N]; // room type membership
  vector[K] xi[N];     // matrix of covariate data with constant slope
  vector[N] yi;    // log price data
}
parameters {
  vector[J] alpha;
  vector[K] beta;
  real mu;
  real<lower=0> sigma_a;  
  real<lower=0> sigma_y;
}
model {
  vector[N] y_hat;
  for (i in 1:N)
    y_hat[i] = alpha[room[i]] + dot_product(beta,xi[i]);
  
  // priors
  mu ~ normal(0,1);
  beta ~ normal(0,1);
  sigma_a ~ normal(0,1);
  sigma_y ~ normal(0,1);
  
  // data generation
  alpha ~ normal(mu, sigma_a);
  yi ~ normal(y_hat, sigma_y);
}
generated quantities {
  vector[N] log_lik;        // pointwise log-likelihood for LOO
  vector[N] log_price_rep;  // replications from posterior predictive dist

  for (n in 1:N) {
    real log_price = alpha[room[n]] + dot_product(beta, xi[n]);
    log_lik[n] = normal_lpdf(yi[n] | log_price, sigma_y);
    log_price_rep[n] = normal_rng(log_price, sigma_y);
  }
}
