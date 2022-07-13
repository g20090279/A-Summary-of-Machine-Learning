[Expectation Maximization / 期望最大化]

# Background

## Background Knowledge

- Probability theory
- Maximum-likelihood estimation

A random variable is denoted by a capital letter, e.g. $\boldsymbol{X}$, and its realization by a lowercase such as $\boldsymbol{x}$.

## Terminology

- Likelihood function: in maximum likelihood estimation, the likelihood function is the probability density function of the observation given by the parameter.
- Log likelihood function:

# History

The name *Expectation Maximization* (EM) comes from the work of Dempster et al. in 1977 (Dempster, 1997), where the algorithm consists of an expectation step followed by a maximization step.

# Brief Review of Maximum Likelihood Estimation

EM generalizes the maximum-likelihood estimation (MLEs) in an iterative manner. First, let's have a short review of MLE method. Assume $\boldsymbol{\theta}$ is the parameter to be estimated. $\boldsymbol{Z}$ is the observation data (or called measurement data). Note that $\boldsymbol{\theta}$ is not a random variable, while the observation $\boldsymbol{Z}$ is random. The observed data $\boldsymbol{z}$ are a realization from $\boldsymbol{Z}$.

Given an unknown deterministic parameter $\boldsymbol{\theta}$ and the observation $\boldsymbol{z}$. The probability density function (PDF) of the observation is then

$$f_{\boldsymbol{Z}}(\boldsymbol{z}|\boldsymbol{\theta}).\qquad(1)$$

Eq. (1) is a function of parameter $\boldsymbol{\theta}$ given by the observation data $\boldsymbol{Z}=\boldsymbol{z}$.

In the traditional MSE, the information of $\boldsymbol{\theta}$ can be observed in $\boldsymbol{z}$. The MSE algorithm is to find a value of $\boldsymbol{\theta}$ which maximizes the PDF in Eq. (1), or the likelihood function $\log f_{\boldsymbol{Z}}(\boldsymbol{z}|\boldsymbol{\theta})$. **For simplicity, suppose that $f_{\boldsymbol{Z}}(\boldsymbol{z}|\boldsymbol{\theta})$ has the regular exponential-family form** (Dempster, 1977). Note that the logarithm is strictly monotonously increasing, indicating that maximizing $f$ and $\log f$ are equivalent. It is convenient to introduce the logarithm in the calculate, resulting in a likelihood function

$$L(\boldsymbol{\theta})=\log f_{\boldsymbol{Z}}(\boldsymbol{z}|\boldsymbol{\theta}).$$

Note that the likelihood function is a function of hte parameter $\boldsymbol{\theta}$ give the data $\boldsymbol{Z}=\boldsymbol{z}$.

The optimal parameter is then found by maximizing the likelihood function, where the name "Maximum Likelihood" comes from.

$$\boldsymbol{\theta}^*=\argmax_{\boldsymbol{\theta}}\ \log f_{\boldsymbol{Z}}(\boldsymbol{z}|\boldsymbol{\theta}).\qquad(2)$$

To obtain the optimum in Eq. (2), make the derivative of the likelihood function with respect to $\boldsymbol{\theta}$ zero, whose solution is the optimal $\boldsymbol{\theta}^{*}$.

# EM for Incomplete Observation

Until now, we assume that the observation $\boldsymbol{Y}$ contains all statistical information for estimating parameter $\boldsymbol{\theta}$. However, sometimes only a part of observation is available, which is denoted as $\boldsymbol{Z}$. While the missing data (sometimes called latent data, latent variable) $\boldsymbol{X}=\boldsymbol{Y}-\boldsymbol{Z}$ is not observable. $\boldsymbol{Y}$ is also called complete data. There is a many-to-one mapping from $\boldsymbol{Y}$ to $\boldsymbol{Z}$, i.e. $\boldsymbol{Z}=h(\boldsymbol{Y})$. Sometimes the missing data may be introduced purely as an artifice to make the MLE of $\boldsymbol{\theta}$ tractable, i.e., easier to maximize the likelihood function.

## Missing data

The EM algorithm deals with the problem of accidental or unintended missing data. Formally, we need to assume that (Dempster, 1977)

1. $\boldsymbol{\theta}$ is a priori independent of the parameters of the missing data process,
2. the missing data are missing at random.  

With the missing data, the likelihood function, the PDF of observation $\boldsymbol{Z}$ given by $\boldsymbol{\theta}$, is the marginal PDF of complete-data likelihood function

$$\begin{aligned}
    L(\boldsymbol{\theta})=&\log \int_{\boldsymbol{X}}f_{\boldsymbol{Y}}(\boldsymbol{y}|\boldsymbol{\theta})\ d\boldsymbol{x}\\
    =&\log \int_{\boldsymbol{X}}f_{\boldsymbol{Z,X}}(\boldsymbol{z},\boldsymbol{x}|\boldsymbol{\theta})\ d\boldsymbol{x}
    \\
    =&\log \int_{\boldsymbol{X}}f_{\boldsymbol{Z}}(\boldsymbol{z}|\boldsymbol{x},\boldsymbol{\theta})f_{\boldsymbol{X}}(\boldsymbol{x}|\boldsymbol{\theta})\ d\boldsymbol{x},
\end{aligned}$$

which is related to the PDF of the complete data. Denote the likelihood function with respect to complete data as

$$L_c(\boldsymbol{\theta})=\log f_{\boldsymbol{Y}}(\boldsymbol{y}|\boldsymbol{\theta})=\log f_{\boldsymbol{Z},\boldsymbol{X}}(\boldsymbol{z},\boldsymbol{x}|\boldsymbol{\theta}).$$

However, since $\boldsymbol{X}$ is unobservable and hence unknown, the probability of it given $\boldsymbol{\theta}$ is attainable only after $\boldsymbol{\theta}$ is obtained.

## EM Algorithm Detail

The EM algorithm consists of two steps. Assume at the $t$-th iteration.

## E-step: estimate the complex-data statistics

The E-step is to estimate the distribution for the complete data $\boldsymbol{Y}$ by averaging the miss data $\boldsymbol{X}$ given the observation data $\boldsymbol{Z}$ and the currently estimated parameter $\boldsymbol{\theta}^{(t)}$ as

$$Q\left(\boldsymbol{\theta}|\boldsymbol{\theta}^{(t)}\right)=\mathbb{E}_{\boldsymbol{Z}|\boldsymbol{X},\boldsymbol{\theta}}\left[L_c\left(\boldsymbol{\theta}^{(t)}\right)\right].$$

## M-step: update the parameter

Once we have the estimated likelihood function with respect to the complete data based on the estimated parameter at $i$-th iteration, we can update the parameter as

$$\boldsymbol{\theta}^{(t+1)}=\argmax_{\boldsymbol{\theta}}Q\left(\boldsymbol{\theta}|\boldsymbol{\theta}^{(t)}\right).$$

# References

Dempster, A. P., Laird N. M. & Rubin D. B. (1977). Maximum Likelihood from incomplete data via the EM algorithm. *Journal of the Royal Statistical Society: Series B, 39*(1):1-38.