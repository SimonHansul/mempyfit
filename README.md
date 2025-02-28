# mempyfit

This package is part of the `mempy` ecosystem. <br>
The goals of this package is to provide basic functioanlity to fit (DEB-)TKTD models using either local optimization, global optimization or likelihood-free Bayesian inference.

## When to (not) use `mempyfit`

`mempyfit` is first and foremost designed to be used in combination with `mempyDEB`, but the functions provided here can be applied to any model. <br>
The `mempy` ecosystem also includes `mempyGUTS`, which already has built-in functionality to fit GUTS models. <br>
Furthermore, there is [pymob](https://pymob.readthedocs.io/en/latest/), which also allows to flexibly configure mechanistic model fits. <br>
This might be confusing at first, but we can provide some rough guidelines for what to use when:

 - `mempyGUTS`: Fitting GUTS models to standard toxicity data (including mixtures). Local optimization of GUTS models is easy with `mempyGUTS`, but uncertainty analysis is rudimentary. Fitting of standard GUTS models can be done without much knowledge of what happens *under the hood*. 
 - `mempyfit`: Primarily designed for fitting models defined in `mempyDEB`. Only provides some basic infrastructure. Application to new cases requires some knowledge of how model fitting works (e.g. defining your own loss function).
- `pymob`: Basically a domain-specific language for specifiyng model ODE-based model calibrations. This is best to use if you want to apply your fitting process to a large number of cases, and if it is worhtwhile to invest a few hours to get acquainted with the interface.


## Why are not doing everything with one package?

I am writing this in February 2025. Eventually, the goal might be to do everything with `pymob`. <br>
In essence, we currently have no off-the-shelf solution for fitting dynamic models to data. <br><br>
The challenges here mostly lie in the variability of the input data and the consequences for formulatig a loss function or likelihood. For example, fitting GUTS models to standard toxicity data has the advantage that the input data is always survival over time. For DEB models, the situation can be very different. Sometimes we have growth and reproduction over time, sometimes we have only scalar measurements (e.g. somatic growth rate, maximum reproductive output etc.), sometimes a combination of both. Therefore, we also cannot always directly compare the ODE solutions to observations, but need to perform some additional processing before we can calculate the loss (or likelihood, error, whatever...). <br>
All of this necessitates a certain amount of flexibilty in the definition of the fitting problem. <br>
Currently, we cannot provide a simple `fit_model()`-function that will fit your model to any kind of data. <br>


Others have solved parts of these problems before.
For example, the `AMPtool`, written in matlab, is doing an amazing job at incorporating different types of observations, but does not account for uncertainties. <br>
The matlab package `BYOM` is also an excellent tool to fit dynamic models to data, but incorporating data which is not time-resolved, or includes independent variables which have not been considere before can be tricky. Also, it is desirable to have a solution within an open-source environment, especially since many universities are outphasing their matlab-licenses nowadays, and Python/Julia/Rust/Go are crystallyizing themselves as the next-generation programming languages. <br><br> 


# TODO

- Should mempyfit use pymob as a depency? Should they become one eventually?