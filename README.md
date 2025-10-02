# Infinitesimal Jackknife for Deep Learning Uncertainty

This repository accompanies the paper:  
**[Automatic Differentiation and the Infinitesimal Jackknife for Uncertainty Quantification in Deep Learning](./latex/main.tex)**.  

The project explores how the **Infinitesimal Jackknife (IJ)**, a classical resampling method, can be adapted via **automatic differentiation** to provide fast and scalable **uncertainty quantification** for deep learning models. Instead of relying on expensive bootstrapping or Bayesian ensembles, IJ propagates per-sample influences through training, yielding predictive variance estimates with minimal computational overhead.

---

## ✨ Key Ideas

- **Influence Tracking**: Each training point contributes a *sensitivity vector* (influence), updated alongside model parameters during SGD.  
- **Variance Estimation**: IJ projects influence vectors into prediction space to estimate the variance of any test prediction.  
- **Bridging Classical & Modern**: In linear and quadratic settings, IJ confidence intervals match those from closed-form OLS theory. In nonlinear neural networks, IJ provides uncertainty bands where no closed form exists.  
- **Training Dynamics**: We visualize how IJ bands evolve during optimization, revealing both convergence behavior and per-sample sensitivity.  

---

## 📂 Repository Contents

- **Linear & Quadratic Verification**  
  Scripts comparing IJ estimates against OLS confidence intervals, ensuring correctness in simple settings.  

- **Nonlinear Experiments**  
  Extensions to MLPs (e.g. sine wave regression) where IJ serves as the only source of uncertainty quantification.  

- **Training Evolution**  
  Visualizations showing how uncertainty bands shrink as models train, providing insight into optimization dynamics.  

- **Additional Variants**  
  Different architectures, datasets, and verification pipelines to stress-test the method.  

---

## 🚀 Getting Started

### Requirements
- Python 3.9+
- [PyTorch](https://pytorch.org/)
- NumPy
- Matplotlib

Install with:
```bash
pip install torch numpy matplotlib
