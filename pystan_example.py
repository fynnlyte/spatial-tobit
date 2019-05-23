import pystan
print('Running pystan version {}, should be 2.19.0.0'.format(pystan.__version__))
model_code = 'parameters {real y;} model {y ~ normal(0,1);}'
model = pystan.StanModel(model_code=model_code)
y = model.sampling().extract()['y']
print('If this worked, you will see a value near 0 now:')
print(y.mean())
