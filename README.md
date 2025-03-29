# Project 1: Linear Dynamics in a Monetary Model

This is the course project for FE5213.

## Overview

Central banks aim to stabilize inflation and output fluctuations using monetary policy instruments such as the interest rate, typically via a Taylor rule. The analysis of monetary policy is an important issue in the New Keynesian monetary economics. In this project, we will analyze the linear dynamics in a monetary model using Python. It involves formulating and setting up the state-space representation, solving impulse responses to economic and monetary shocks, and conducting comparative statics as the model parameters change.

## Problem Setup

### Variables

Variables in the model: the output $(\pi_t, x_t, i_t)$, three shocks $\left(u_{t}, r_{t}^{n}, \nu_{t}\right)$, with unit white noise $(\varepsilon_{u t}, \varepsilon_{r t}, \varepsilon_{\nu t})$.

| Variables | Meaning | Setting |
|:---------:|:-------:|:-------:|
|$\pi_t$|inflation|New Keynesian Phillips curve|
|$x_t$|output gap|IS curve|
|$i_t$|nominal interest rate (policy instrument)|Taylor rule|
|$u_t$|cost-push shock| AR(1) process |
|$r_{t}^{n}$|demand shock| AR(1) process |
|$\nu_t$|monetary policy shock| AR(1) process |
|$\varepsilon_{u t}$| unit white noise for cost-push shock | $N(0,1)$|
|$\varepsilon_{r t}$| unit white noise for demand shock | $N(0,1)$|
|$\varepsilon_{\nu t}$| unit white noise for monetary policy shock | $N(0,1)$|

### Parameters

Parameters in the model. In the benchmark analysis, use the calibrated parameters taken from the literature.

| Parameter | Meaning | Benchmark Value  |
|:--------:|:------:|:------:|
| $\beta$  | discount factor | 0.99   |
| $\sigma$ | sensitivity to the real interest rate | 1/6 |
| $\kappa$ | slope of the Phillips curve | 0.024  |
| $\phi_{\pi}$ | sensitivity to the inflation | 1.5 |
| $\phi_{x}$ | sensitivity to the output gap | 0.5 |
| $\rho_{r}$ | AR(1) coefficient for demand shock| 0.35  |
| $\rho_{u}$ | AR(1) coefficient for cost-push shock | 0.35  |
| $\rho_{\nu}$ | AR(1) coefficient for monetary policy shock | 0.35 |
| $\sigma_{r}$ | std for demand shock white noise | 3.7 |
| $\sigma_{u}$ |std for cost-push shock white noise | 0.4 |
| $\sigma_{\nu}$ |std for monetary policy shock white noise | 1 |

## Model Solution

### Solve the Model

The Rational Expectations equilibrium for this model can be solved as follows. Conjecture a model solution where the output is linear in the state comprised of three shocks $\left(u_{t}, r_{t}^{n}, \nu_{t}\right):$

$$
    \left[\begin{array}{c}
    \pi_{t}\\
    x_{t} \\
    i_{t}
    \end{array}\right]= \mathbf{P} \left[\begin{array}{c}
    u_{t} \\
    r_{t}^{n} \\
    \nu_{t}
    \end{array}\right] := \left[\begin{array}{ccc}
    \gamma_{\pi}^{u} & \gamma_{\pi}^{r} & \gamma_{\pi}^{\nu} \\
    \gamma_{x}^{u} & \gamma_{x}^{r} & \gamma_{x}^{\nu} \\
    \gamma_{i}^{u} & \gamma_{i}^{r} & \gamma_{i}^{\nu}
    \end{array}\right]\left[\begin{array}{c}
    u_{t} \\
    r_{t}^{n} \\
    \nu_{t}
    \end{array}\right].
$$
Solve the system of nine linear equations to get the matrix \mathbf{P}.
$$
    \begin{align*}
        & \gamma_{\pi}^{u}=\beta \gamma_{\pi}^{u} \rho_{u}+\kappa \gamma_{x}^{u}+1\\
        & \gamma_{\pi}^{r}=\beta \gamma_{\pi}^{r} \rho_{r}+\kappa \gamma_{x}^{r}\\
        & \gamma_{\pi}^{\nu}=\beta \gamma_{\pi}^{\nu} \rho_{\nu}+\kappa \gamma_{x}^{\nu}\\
        & \gamma_{x}^{u}=\gamma_{x}^{u} \rho_{u}-\sigma\left(\gamma_{i}^{u}-\gamma_{\pi}^{u} \rho_{u}\right)\\
        & \gamma_{x}^{r}=\gamma_{x}^{r} \rho_{r}-\sigma\left(\gamma_{i}^{r}-\gamma_{\pi}^{r} \rho_{r}-1\right)\\
        & \gamma_{x}^{\nu}=\gamma_{x}^{\nu} \rho_{\nu}-\sigma\left(\gamma_{i}^{\nu}-\gamma_{\pi}^{\nu} \rho_{\nu}\right)\\
        & \gamma_{i}^{u}=\phi_{\pi} \gamma_{\pi}^{u}+\phi_{x} \gamma_{x}^{u}\\
        & \gamma_{i}^{r}=\phi_{\pi} \gamma_{\pi}^{r}+\phi_{x} \gamma_{x}^{r}\\
        & \gamma_{i}^{\nu}=\phi_{\pi} \gamma_{\pi}^{\nu}+\phi_{x} \gamma_{x}^{\nu}+1
    \end{align*}
$$
The system of linear equations can be packed into matrix form
$$
    \mathbf{A}\mathbf{x} = \mathbf{b},
$$
where $\mathbf{x} := \operatorname{vec}(\mathbf{P}) = \left[\gamma_{\pi}^u,\ \gamma_x^u,\ \gamma_i^u,\,
\gamma_{\pi}^r,\ \gamma_x^r,\ \gamma_i^r,\,
\gamma_{\pi}^{\nu},\ \gamma_x^{\nu},\ \gamma_i^{\nu}
\right]^{\top},$
$$
\mathbf{A} = \left[
\begin{array}{ccccccccc}
     1-\beta \rho_u & -\kappa & 0 & 0 & 0 & 0 & 0 & 0 & 0 \\
     0 & 0 & 0 & 1-\beta \rho_r & -\kappa & 0 & 0 & 0 & 0 \\
     0 & 0 & 0 & 0 & 0 & 0 & 1-\beta \rho_{\nu} & -\kappa & 0 \\
     \sigma\rho_u & \rho_u-1 & -\sigma & 0 & 0 & 0 & 0 & 0 & 0 \\
     0 & 0 & 0 & \sigma\rho_r & \rho_r-1 & -\sigma & 0 & 0 & 0 \\
     0 & 0 & 0 & 0 & 0 & 0 & \sigma\rho_{\nu} & \rho_{\nu}-1 & -\sigma \\
     \phi_{\pi} & \phi_x & -1 & 0 & 0 & 0 & 0 & 0 & 0 \\
     0 & 0 & 0 & \phi_{\pi} & \phi_x & -1 & 0 & 0 & 0 \\
     0 & 0 & 0 & 0 & 0 & 0 & \phi_{\pi} & \phi_x & -1 \\
\end{array}
\right],\, \mathbf{b} = \left[
\begin{array}{c}
    1 \\ 0 \\ 0 \\ 0 \\ -\sigma \\ 0 \\ 0 \\ 0 \\ -1
\end{array}
\right].
$$
Hence, $\mathbf{P}$ can be easily solved using matrix inversion.

### State-Space Form

With $\mathbf{P}$, the model is characterized by a linear state-space system with the measurement equation

$$
    \left[\begin{array}{c}
    \pi_{t} \\
    x_{t} \\
    i_{t}
    \end{array}\right]=\mathbf{P}\left[\begin{array}{c}
    u_{t} \\
    r_{t}^{n} \\
    \nu_{t}
    \end{array}\right],
$$

and the transition equation

$$
    \left[\begin{array}{l}
    u_{t}\\
    r_{t}^{n} \\
    \nu_{t}
    \end{array}\right]
    =
    \left[\begin{array}{ccc}
    \rho_{u} & 0 & 0 \\
    0 & \rho_{r} & 0 \\
    0 & 0 & \rho_{\nu}
    \end{array}\right]
    \left[\begin{array}{l}
    u_{t-1} \\
    r_{t-1}^{n} \\
    \nu_{t-1}
    \end{array}\right]
    +
    \left[\begin{array}{ccc}
    \sigma_{u} & 0 & 0 \\
    0 & \sigma_{r} & 0 \\
    0 & 0 & \sigma_{\nu}
    \end{array}\right]
    \left[\begin{array}{l}
    \varepsilon_{u t} \\
    \varepsilon_{r t} \\
    \varepsilon_{\nu t}
    \end{array}\right].
$$

## Main Problems to Address

- Solve $\mathbf{P}$ and formulate the model solutions as a linear state space system.
- Plot and discuss the dynamic responses of $\pi_{t+i}, x_{t+i}, i_{t+i}$ to a one standard deviation change in economic shocks $\varepsilon_{r t}, \varepsilon_{u t}$ and the monetary policy shock $\varepsilon_{\nu t}$.
- Investigate how the results would change if we change $\kappa, \rho_{u}, \rho_{r}, \phi_{\pi}, \phi_{x}$. Discuss your results.
