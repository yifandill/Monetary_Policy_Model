# Project 1: Linear Dynamics in a Monetary Model

This is the course project for FE5213.

## Overview

Central banks aim to stabilize inflation and output fluctuations using monetary policy instruments such as the interest rate, typically via a Taylor rule. The analysis of monetary policy is an important issue in the New Keynesian monetary economics. In this project, students will analyze the linear dynamics in a monetary model using Python. It involves formulating and setting up the state-space representation, solving impulse responses to economic and monetary shocks, and conducting comparative statics as the model parameters change.

## Problem Setup

### Economic Dynamics

We consider a simple New Keynesian model, where the economy is described by a Phillips curve and IS curve. In particular,

$$
\begin{align*}
\pi_{t} & =\beta \mathbb{E}_{t} \pi_{t+1}+\kappa x_{t}+u_{t}  \tag{1}\\
x_{t} & =\mathbb{E}_{t} x_{t+1}-\sigma\left(i_{t}-\mathbb{E}_{t} \pi_{t+1}-r_{t}^{n}\right) \tag{2}
\end{align*}
$$

where (1) is the New Keynesian Phillips curve and (2) is the IS curve. In these equations, $\pi_{t}$ denotes inflation, $x_{t}$ output gap, $i_{t}$ nominal interest rate (policy instrument), $u_{t}$ cost-push shock, $r_{t}^{n}$ demand shock, $\beta$ discount factor, $\kappa$ slope of the Phillips curve, $\sigma$ sensitivity to the real interest rate.

Assume that $u_{t}$ and $r_{t}^{n}$ follow the following AR(1) process:

$$
\begin{align*}
u_{t} & =\rho_{u} u_{t-1}+\sigma_{u} \varepsilon_{u t}  \tag{3}\\
r_{t}^{n} & =\rho_{r} r_{t-1}^{n}+\sigma_{r} \varepsilon_{r t} \tag{4}
\end{align*}
$$

where $\varepsilon_{u t} \sim N(0,1)$ and $\varepsilon_{r t} \sim N(0,1)$.

### Central Bank's Policy Rule

The central bank sets interest rate via the Taylor rule, which takes the following form:

$$
\begin{equation*}
i_{t}=\phi_{\pi} \pi_{t}+\phi_{x} x_{t}+\nu_{t} \tag{5}
\end{equation*}
$$

where

$$
\begin{equation*}
\nu_{t}=\rho_{\nu} \nu_{t-1}+\sigma_{\nu} \varepsilon_{\nu t} \tag{6}
\end{equation*}
$$

$\varepsilon_{\nu t} \sim N(0,1)$ is the monetary policy surprise/shock.

## Model Solution

### Solve the Model

The Rational Expectations equilibrium for this model can be solved as follows. Conjecture a model solution where the output is linear in the state comprised of three shocks $\left(u_{t}, r_{t}^{n}, \nu_{t}\right):$

$$
\left[\begin{array}{c}
\pi_{t}  \tag{7}\\
x_{t} \\
i_{t}
\end{array}\right]=\underbrace{\left[\begin{array}{ccc}
\gamma_{\pi}^{u} & \gamma_{\pi}^{r} & \gamma_{\pi}^{\nu} \\
\gamma_{x}^{u} & \gamma_{x}^{r} & \gamma_{x}^{\nu} \\
\gamma_{i}^{u} & \gamma_{i}^{r} & \gamma_{i}^{\nu}
\end{array}\right]}_{\text {denoted by } P}\left[\begin{array}{c}
u_{t} \\
r_{t}^{n} \\
\nu_{t}
\end{array}\right].
$$

With (7), we can write the one-period-ahead expectations accordingly:

$$
\begin{align*}
\mathbb{E}_{t} \pi_{t+1} & =\gamma_{\pi}^{u} \rho_{u} u_{t}+\gamma_{\pi}^{r} \rho_{r} r_{t}^{n}+\gamma_{\pi}^{\nu} \rho_{\nu} \nu_{t}  \tag{8}\\
\mathbb{E}_{t} x_{t+1} & =\gamma_{x}^{u} \rho_{u} u_{t}+\gamma_{x}^{r} \rho_{r} r_{t}^{n}+\gamma_{x}^{\nu} \rho_{\nu} \nu_{t},  \tag{9}\\
\mathbb{E}_{t} i_{t+1} & =\gamma_{i}^{u} \rho_{u} u_{t}+\gamma_{i}^{r} \rho_{r} r_{t}^{n}+\gamma_{i}^{\nu} \rho_{\nu} \nu_{t} \tag{10}
\end{align*}
$$

To solve for $P$, we can substitute items in (1), (2) and (5) with (7) and (8)-(10) and solve $P$ using the method of undetermined coefficients.

From (1):

$$
\gamma_{\pi}^{u} u_{t}+\gamma_{\pi}^{r} r_{t}^{n}+\gamma_{\pi}^{\nu} \nu_{t}
=\beta\left(\gamma_{\pi}^{u} \rho_{u} u_{t}+\gamma_{\pi}^{r} \rho_{r} r_{t}^{n}+\gamma_{\pi}^{\nu} \rho_{\nu} \nu_{t}\right)
+\kappa\left(\gamma_{x}^{u} u_{t}+\gamma_{x}^{r} r_{t}^{n}+\gamma_{x}^{\nu} \nu_{t}\right)
+u_{t},
$$

collecting items and re-arranging:

$$
\left(\gamma_{\pi}^{u}-\beta \gamma_{\pi}^{u} \rho_{u}-\kappa \gamma_{x}^{u}-1\right) u_{t}
+\left(\gamma_{\pi}^{r}-\beta \gamma_{\pi}^{r} \rho_{r}-\kappa \gamma_{x}^{r}\right) r_{t}^{n}
+\left(\gamma_{\pi}^{\nu}-\beta \gamma_{\pi}^{\nu} \rho_{\nu}-\kappa \gamma_{x}^{\nu}\right) \nu_{t}=0
$$

This implies that

$$
\begin{align*}
& \gamma_{\pi}^{u}=\beta \gamma_{\pi}^{u} \rho_{u}+\kappa \gamma_{x}^{u}+1  \tag{11}\\
& \gamma_{\pi}^{r}=\beta \gamma_{\pi}^{r} \rho_{r}+\kappa \gamma_{x}^{r}  \tag{12}\\
& \gamma_{\pi}^{\nu}=\beta \gamma_{\pi}^{\nu} \rho_{\nu}+\kappa \gamma_{x}^{\nu} \tag{13}
\end{align*}
$$

From (2):

$$
\begin{aligned}
\gamma_{x}^{u} u_{t}+\gamma_{x}^{r} r_{t}^{n}+\gamma_{x}^{\nu} \nu_{t}
&=\left(\gamma_{x}^{u} \rho_{u} u_{t}+\gamma_{x}^{r} \rho_{r} r_{t}^{n}+\gamma_{x}^{\nu} \rho_{\nu} \nu_{t}\right) \\
&\quad -\sigma\left[\left(\gamma_{i}^{u} u_{t}+\gamma_{i}^{r} r_{t}^{n}+\gamma_{i}^{\nu} \nu_{t}\right)
-\left(\gamma_{\pi}^{u} \rho_{u} u_{t}+\gamma_{\pi}^{r} \rho_{r} r_{t}^{n}+\gamma_{\pi}^{\nu} \rho_{\nu} \nu_{t}\right)
-r_{t}^{n}\right]
\end{aligned}
$$

collecting items, we have

$$
\begin{align*}
& \gamma_{x}^{u}=\gamma_{x}^{u} \rho_{u}-\sigma\left(\gamma_{i}^{u}-\gamma_{\pi}^{u} \rho_{u}\right)  \tag{14}\\
& \gamma_{x}^{r}=\gamma_{x}^{r} \rho_{r}-\sigma\left(\gamma_{i}^{r}-\gamma_{\pi}^{r} \rho_{r}-1\right)  \tag{15}\\
& \gamma_{x}^{\nu}=\gamma_{x}^{\nu} \rho_{\nu}-\sigma\left(\gamma_{i}^{\nu}-\gamma_{\pi}^{\nu} \rho_{\nu}\right) \tag{16}
\end{align*}
$$

From (5):

$$
\gamma_{i}^{u} u_{t}+\gamma_{i}^{r} r_{t}^{n}+\gamma_{i}^{\nu} \nu_{t}
=\phi_{\pi}\left(\gamma_{\pi}^{u} u_{t}+\gamma_{\pi}^{r} r_{t}^{n}+\gamma_{\pi}^{\nu} \nu_{t}\right)
+\phi_{x}\left(\gamma_{x}^{u} u_{t}+\gamma_{x}^{r} r_{t}^{n}+\gamma_{x}^{\nu} \nu_{t}\right)
+\nu_{t}
$$

So we have

$$
\begin{align*}
& \gamma_{i}^{u}=\phi_{\pi} \gamma_{\pi}^{u}+\phi_{x} \gamma_{x}^{u}  \tag{17}\\
& \gamma_{i}^{r}=\phi_{\pi} \gamma_{\pi}^{r}+\phi_{x} \gamma_{x}^{r}  \tag{18}\\
& \gamma_{i}^{\nu}=\phi_{\pi} \gamma_{\pi}^{\nu}+\phi_{x} \gamma_{x}^{\nu}+1 \tag{19}
\end{align*}
$$

Note that in the system of equations (11) through (19), there are 9 equations with 9 unknowns in the matrix $P$:

$$
P=\left[\begin{array}{lll}
\gamma_{\pi}^{u} & \gamma_{\pi}^{r} & \gamma_{\pi}^{\nu} \\
\gamma_{x}^{u} & \gamma_{x}^{r} & \gamma_{x}^{\nu} \\
\gamma_{i}^{u} & \gamma_{i}^{r} & \gamma_{i}^{\nu}
\end{array}\right]
$$

which can be easily solved using matrix inversion.

### State-Space Form

With $P$, the model is characterized by a linear state-space system with the measurement equation

$$
\left[\begin{array}{c}
\pi_{t}  \tag{20}\\
x_{t} \\
i_{t}
\end{array}\right]=P\left[\begin{array}{c}
u_{t} \\
r_{t}^{n} \\
\nu_{t}
\end{array}\right],
$$

and the transition equation

$$
    \left[\begin{array}{l}
    u_{t}  \tag{21}\\
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

## Parameters

In the benchmark analysis, use the following calibrated parameters that are taken from the literature.

| Parameter | Value  |
|:--------:|:------:|
| $\beta$   | 0.99   |
| $\sigma$  | $1 / 6$ |
| $\kappa$  | 0.024  |
| $\rho_{r}$ | 0.35  |
| $\rho_{u}$ | 0.35  |
| $\rho_{\nu}$ | 0.35 |
| $\sigma_{r}$ | 3.7 |
| $\sigma_{u}$ | 0.4 |
| $\sigma_{\nu}$ | 1 |
| $\phi_{\pi}$ | 1.5 |
| $\phi_{x}$ | 0.5 |

## Main Problems to Address

- Solve $P$ and formulate the model solutions as a linear state space system.
- Plot and discuss the dynamic responses of $\pi_{t+i}, x_{t+i}, i_{t+i}$ to a one standard deviation change in economic shocks $\varepsilon_{r t}, \varepsilon_{u t}$ and the monetary policy shock $\varepsilon_{\nu t}$.
- Investigate how the results would change if we change $\kappa, \rho_{u}, \rho_{r}, \phi_{\pi}, \phi_{x}$. Discuss your results.
