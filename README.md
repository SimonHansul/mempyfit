# mempyfit

This package is part of the `mempy` ecosystem. <br>
The goals of this package is to provide basic functioanlity to fit (DEB-)TKTD models using either local optimization, global optimization or likelihood-free Bayesian inference.

## Installation

You can install the package using the repo URL and pip:

```bash
pip install git+https://github.com/simonhansul/mempyfit
```

## Quickstart


`mempyfit` defines a `ModelFit` class, which can be initialized without arguments:

```Python
f = ModelFit()
```

Then, we add the information required for the fitting process. <br>
In any case, we need:

- `f.defaultparams`: The full set of parameters needed to run the model (whether they will be fitted or not)
- `f.simulator`: A function which takes a parameter set as dictionaries as argument and returns a prediction.
- `f.data`: The data to which the model will be fitted, e.g. as `pandas.DataFrame` or dict containing multiple dataframes
- `f.loss`: A function which takes `f.data` and the return value of `f.simulator` as argument, and returns the loss.

 We can perform model fitting using local optimization through scipy or likelihood-free Bayesian inference through pyabc. <br>
 For local optimization, we need to define a dict `f.intguess`, listing initial guesses of parameters to be fitted. <br>
 For Bayesian inference, we need to define `f.prior`, which is a `pyabc.Distribution`. 
 Priors can also be defined as log-normal distributions with the medians equal to values in some dict:

 ```Python
medians = {
  'p1' : 1,
  'p2' : 2
}
f.define_lognorm_prior(medians, sigma = 1)
```

The `define_lognorm_prior` method will then apply the same log-variance to all parameters. <br> 

Now we can execute the model fit using either one of two methods:

```Python
f.run_optimization() # run local optimization
f.run_bayesian_inference() # run likelihod-free bayesian inference
```
