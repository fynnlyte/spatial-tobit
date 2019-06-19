data {
    int<lower=0> N;
    int y[N]; // vec of N ints
}
parameters {
    real<lower=0, upper=1> theta;
}
model {
    theta ~ beta(1,1);
    y ~ bernoulli(theta);
}
