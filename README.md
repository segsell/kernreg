# KernReg
[![PyPI](https://img.shields.io/pypi/v/kernreg.svg)](https://pypi.org/project/kernreg/)
[![Continuous Integration](https://github.com/segsell/kernreg/workflows/Continuous%20Integration/badge.svg?branch=main)](https://github.com/segsell/kernreg/actions?query=workflow%3A%22Continuous+Integration%22)
[![Codecov](https://codecov.io/gh/segsell/kernreg/branch/main/graph/badge.svg)](https://codecov.io/gh/segsell/kernreg)
[![Codacy Badge](https://app.codacy.com/project/badge/Grade/5dd752959ec8415c8fa9cc9c18ac7d9a)](https://www.codacy.com/gh/segsell/kernreg/dashboard?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=segsell/kernreg&amp;utm_campaign=Badge_Grade)
[![Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

## Introduction
**KernReg** provides a pure-Python routine for local polynomial kernel regression based on [Wand & Jones (1995)](http://matt-wand.utsacademics.info/webWJbook/) and their accompanying *R* package [KernSmooth](https://www.rdocumentation.org/packages/KernSmooth/versions/2.23-18). In addition, **KernReg** comes with an automatic bandwidth selection procedure that minimizes the residual squares criterion proposed by [Fan & Gijbels (1996)](https://www.taylorfrancis.com/books/local-polynomial-modelling-applications-fan-gijbels/10.1201/9780203748725).

**KernReg** allows for the estimation of a regression function as well as their *v*th derivatives. The degree of the polynomial *p* must be equal to ```v + 1```,
```v + 3```, ```v + 5```, or ```v + 7```.

<p align="center">
  <img width="650" height="450" src="https://github.com/segsell/hypermodern-kernreg/blob/main/docs/images/Arthur_Radebaugh_retrofuturism.jpg?raw=true">
</p>

Local polynomial fitting provides a simple way of finding a functional relationship between two variables (where X typically denotes the predictor, and Y the response variable)  without the imposition of a parametric model. It is a natural extension of local mean smoothing, as described by [Nadaraya (1964)](https://www.semanticscholar.org/paper/On-Estimating-Regression-Nadaraya/05175204318c3c01e3301fd864553071039605d2#paper-header) and [Watson (1964)](http://www.jstor.org/stable/25049340). Instead of fitting a local mean, local polynomial smooting involves fitting a local *p*th-order polynomial via locally weighted least-squares. The Nadaraya–Watson estimator is thus equivalent to fitting a local polynomial of degree zero. Local polynomials of higher order have better bias properties and, in general, do not require bias adjustment at the boundaries of the regression space.

## Installation
Install **KernReg** via PyPI.

```console
$ pip install kernreg
```

## Quick-Start
```python
import kernreg as kr

motorcycle = kr.get_example_data()
x, y = motorcycle["time"], motorcycle["accel"]

# By default, only x and y need to be provided.
# Derivative = 0 is chosen by default
# and hence the polynomial degree = 0 + 1.
rslt_default = kr.locpoly(x, y)
kr.plot(x, y, rslt_default, "motorcycle_default_fit.png")
```
![default fit](https://github.com/segsell/kernreg/blob/exampledata/docs/images/motorcycle_default_fit.png?raw=true)

```python
# We can improve on the default specification by
# choosing a higher order polynomial
rslt_user = kr.locpoly(x, y, degree=3)
kr.plot(x, y, rslt_user, "motorcycle_user_fit.png")
```
![user fit](https://github.com/segsell/kernreg/blob/exampledata/docs/images/motorcycle_user_fit.png?raw=true)

## References
Fan, J. and Gijbels, I. (1996). [Local Polynomial Modelling and Its Applications](https://www.taylorfrancis.com/books/local-polynomial-modelling-applications-fan-gijbels/10.1201/9780203748725). *Monographs on Statistics and Applied Probability, 66*. Chapman & Hall.

Wand, M.P. & Jones, M.C. (1995). [Kernel Smoothing](http://matt-wand.utsacademics.info/webWJbook/). *Monographs on Statistics and Applied Probability, 60*. Chapman & Hall.

Wand, M.P. and Ripley, B. D. (2015). [KernSmooth:  Functions for Kernel Smoothing for Wand and Jones (1995)](http://CRAN.R-project.org/package=KernSmooth). *R* package version 2.23-18.

-----
`*` The image is taken from futurist illustrator [Arthur Radebaugh's (1906–1974)](http://www.gavinrothery.com/my-blog/2012/7/15/arthur-radebaugh.html)
Sunday comic strip *Closer Than We Think!*, which was published by the Chicago Tribune - New York News Syndicate from 1958 to 1963.
