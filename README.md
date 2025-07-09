# 📊 Option Pricing Models in Python

This repository presents three fundamental option pricing models implemented in Python:

- 🧠 **Black-Scholes Model** – Analytical formula for European options  
- 🎲 **Monte Carlo Simulation** – Probabilistic method using asset path simulations  
- 🌳 **Binomial Tree Model** – Discrete-time model for pricing with flexibility  

## 📘 Model Descriptions

### 🔹 Black-Scholes Model  
An analytical model for pricing European call and put options. Assumes:
- Constant volatility and interest rates  
- Log-normal distribution of stock prices  
- No dividends or transaction costs  

### 🔹 Monte Carlo Simulation  
Simulates thousands of potential paths for the underlying asset and calculates the average payoff discounted at the risk-free rate. Useful for complex path-dependent options.

### 🔹 Binomial Tree Model  
Constructs a recombining binomial tree of possible stock prices and evaluates option prices by backward induction. Can handle American and European options but here we will focus only on European option.

