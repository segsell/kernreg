[![Continuous Integration](https://github.com/segsell/hypermodern-kernreg/workflows/Continuous%20Integration/badge.svg?branch=main)](https://github.com/segsell/hypermodern-kernreg/actions?workflow=%3A"Continuous+Integration")
[![Codecov](https://codecov.io/gh/segsell/hypermodern-kernreg/branch/main/graph/badge.svg)](https://codecov.io/gh/segsell/hypermodern-kernreg)
[![Codacy Badge](https://app.codacy.com/project/badge/Grade/5dd752959ec8415c8fa9cc9c18ac7d9a)](https://www.codacy.com?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=segsell/hypermodern-kernreg&amp;utm_campaign=Badge_Grade)
[![Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

# KernReg
**KernReg** provides a pure-Python routine for local polynomial kernel regression based on Wand & Jones (1995) and their accompanying R package [KernSmooth](https://www.rdocumentation.org/packages/KernSmooth/versions/2.23-18). In addition, **KernReg** provides an automatic bandwidth selection procedure that minimizes the residual squares criterion proposed by Fan & Gijbels (1996).
**KernReg** allows for the estimation of a regression function as well as their derivatives. The degree of the polynomial may be chosen ad libitum, but ```degree = derivative + 1``` is commonly recommended (see e.g. Fan & Gijbels, 1996) and thus set by default.

# Background
Local polynomial fitting provides a simple way of finding a functional relationship between two variables (usually denoted by X, the predictor, and Y, the response variable)  without the imposition of a parametric model. It is a natural extension of local mean smoothing, as described by Nadaraya (1964) and Watson (1964). Instead of fitting a local mean, local polynomial smooting involves fitting a local pth-order polynomial via locally weighted least-squares. The Nadaraya–Watson estimator is thus equivalent with fitting a local polynomial of degree zero. Local polynomials of higher order have better bias properties and, in general, do not require bias adjustment at the boundaries of the regression space. For a definitive reference on local polynomial smoothing, see Fan & Gijbels (1996).

<!-- (?The kernel weigth is the normal density, i.e. Gaussian kernel.)


Kernel smoothing refers to a general class of techniques for non- parametric estimation of functions. Suppose that you have a uni- variate set of data which you want to display graphically. Then kernel smoothing provides an attractive procedure for achieving this goal, known as kernel density estimation. Another funda- mental example is the simple nonparametric regression or scat- terplot smoothing problem where kernel smoothing offers a way of estimating the regression function without the specification of a parametric model.

Kernel smoothing provides a simple way of finding structure in data sets without the imposition of a parametric model. One of the most fundamental settings where kernel smoothing ideas can be applied is the simple regression problem, where paired observations for each of two variables are available and one is interested in determining an appropriate functional relationship between the two variables. One of the variables, usually denoted by X, is thought of as being a predictor for the other variable Y, usually called the response variable.
[Example]

Figure 1.2 shows an estimate of m for the age/log(income) data, using what is often called a local linear kernel estimator. The function shown at the bottom of the plot is a kernel function which is usually taken to be a symmetric probability density such as a normal density. The value of the estimate at the first point u is obtained by fitting a straight line to the data using weighted least squares, where the weights are chosen according to the height of the kernel function. This means that the data points closer to u have more influence on the linear fit than those far from u. This local straight line fit is shown by the dotted curve and the regression estimate at u is the height of the line at u. The estimate at a different point v is found the same way, but with the weights chosen according to the heights of the kernel when centred around v. This estimator fits into the class of local polynomial regression estimates (Cleveland, 1979). Nonparametric regression estimators are often called regression smoothers or scatterplot smoothers, while those based on kernel functions are often called kernel smoothers.

locpoly command for performing local polynomial regression

Local polynomial regression is a generalization of local mean smoothing as described by Nadaraya(1964) and Watson(1964). Instead of fitting a local mean, one instead fits a local pth-order polynomial.

Calculations for local polynomial regression are naturally more complex than those for local means, but local polynomial smooths have better statistical properties.  The computational complexity is, however, alleviated by using a Stata plugin.

The last twenty years or so have seen a significant outgrowth in the literature on thesubject of scatterplot smoothing, otherwise known as univariate nonparametric regression.  Of most appeal is the idea of not making any assumptions about the functionalform for the expected value of a response given a regressor but instead allowing thedata to “speak for itself”.
Various methods and estimators fall into the category ofnonparametric regression, including local mean smoothing, as described independently by Nadaraya(1964)and Watson(1964); the Gasser–Müller (1979) estimator; locally weighted scatterplot smoothing (LOWESS), as described by Cleveland(1979); wavelets(e.g.,Donoho 1995); and splines (Eubank 1988), to name a few. Much of the vast litera-ture focuses on automating the amount of smoothing to be performed and dealing withthe bias/variance trade-off inherent to this type of estimation. For example, in the caseof Nadaraya–Watson, the amount of smoothing is controlled by choosing abandwidth.
Smoothing via local polynomials is by no means a new idea but instead one that hasbeen rediscovered in recent years in articles such asFan(1992).

A natural extensionof the local mean smoothing of Nadaraya–Watson, local polynomial regression, involvesfitting the response to a polynomial form of the regressor via locally weighted leastsquares.  Compared with the Nadaraya–Watson estimator (local polynomial of degreezero), local polynomials of higher order have better bias properties and, in general, donot require bias adjustment at the boundary of the regression space.  For a definitive reference on local polynomial smoothing, seeFan and Gijbels(1996).

The apparent cost of these improved properties is that local polynomial smooths arecomputationally more complex. For example, the Nadaraya–Watson estimator requiresat each point in the smoothing grid the calculation of a locally weighted mean, whereas local polynomial smoothing would require a weighted regression at each grid point. This cost, however, can be alleviated by using approximation methods such as linear binning (Hall and Wand 1996) or by using updating methods that retain information from previous points in the smoothing grid (e.g.,Fan and Marron 1994).

* Implements residual squares criterion for
Direct computation computationally expensive -->

<p align="center">
  <img width="650" height="450" src="https://github.com/segsell/hypermodern-kernreg/blob/main/docs/images/Arthur_Radebaugh_retrofuturism.jpg?raw=true">
</p>

# References
Fan, J. and Gijbels, I. (1996). [Local Polynomial Modelling and Its Applications](https://www.taylorfrancis.com/books/local-polynomial-modelling-applications-fan-gijbels/10.1201/9780203748725). *Monographs on Statistics and Applied Probability 66*. Chapman & Hall.

Nadaraya, E. A. (1964). [On Estimating Regression](https://www.semanticscholar.org/paper/On-Estimating-Regression-Nadaraya/05175204318c3c01e3301fd864553071039605d2#paper-header). *Theory of Probability and Its Application*, 9(1): 141–142.

Wand, M.P. & Jones, M.C. (1995). [Kernel Smoothing](http://matt-wand.utsacademics.info/webWJbook/). *Monographs on Statistics and Applied Probability 60*. Chapman & Hall.

Wand, M.P. and Ripley, B. D. (2015). KernSmooth:  Functions for Kernel Smoothing for Wand and Jones (1995). **R** package version 2.23-18. http://CRAN.R-project.org/package=KernSmooth

Watson, G. S. (1964). [Smooth Regression Analysis](http://www.jstor.org/stable/25049340). *Sankhyā: The Indian Journal of Statistics, Series A*, 26(4): 359–372.

-----
`*` The image is taken from futurist illustrator [Arthur Radebaugh's (1906–1974)](http://www.gavinrothery.com/my-blog/2012/7/15/arthur-radebaugh.html)
Sunday comic strip *Closer Than We Think!*, which was published by the Chicago Tribune - New York News Syndicate from 1958 to 1963.
