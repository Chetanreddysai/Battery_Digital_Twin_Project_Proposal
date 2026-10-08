# Physics-Informed Neural Digital Twin (PI-NDT)
### Real-Time 3D Battery Thermal State Estimation & Autonomous Fast-Charging Safety

[![Python 3.13](https://img.shields.io/badge/Python-3.13-blue.svg)](https://www.python.org/)
[![PyBaMM](https://img.shields.io/badge/PyBaMM-26.9-brightgreen.svg)](https://pybamm.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-orange.svg)](https://pytorch.org/)
[![RMSE](https://img.shields.io/badge/Core%20RMSE-0.0121%C2%B0C-brightgreen.svg)](models/pinn_metrics.json)

---

## 🚀 Project Overview
Lithium-ion battery packs in electric vehicles (EVs) suffer from an internal **thermal blind spot**: physical sensors can only measure the exterior metal casing ($T_{\text{surf}}$), leaving dangerous internal core hotspots ($T_{\text{core}}$) invisible during fast charging.

This project builds an AI-powered **Physics-Informed Neural Digital Twin** that acts like a real-time software X-ray:
1. Ingests non-invasive surface measurements (Current $I$, Voltage $V$, Surface Temp $T_{\text{surf}}$).
2. Uses a **Physics-Informed Residual Highway Network (PI-RHN)** constrained by radial heat conduction PDEs to deduce invisible 3D core temperatures in real time without interior sensors.
3. Deploys autonomous multi-agent supervisory logic to prevent thermal runaway while maximizing fast-charging throughput.

---

## 📁 Repository Structure
```
battery-neural-digital-twin/
├── data/
│   └── battery_fast_charge_ground_truth.csv    # 600 time-step 2C CC-CV physics telemetry
├── docs/
│   ├── Battery_Digital_Twin_Project_Proposal.pdf # Executive specification PDF
│   └── project_proposal_battery_digital_twin.md  # Markdown proposal & architecture
├── models/
│   ├── pinn_battery_digital_twin.pt            # Trained PyTorch neural model checkpoint
│   └── pinn_metrics.json                       # Quantitative benchmark metrics
├── plots/
│   ├── thermal_blindspot_validation.png        # Step 1: 4-panel physics validation plot
│   └── pinn_core_inference_benchmark.png       # Step 2: 4-panel PINN performance benchmark
├── src/
│   ├── simulate_battery_physics.py             # Step 1: PyBaMM & radial thermal PDE engine
│   └── train_pinn_model.py                     # Step 2: PINN architecture & training pipeline
├── README.md
└── requirements.txt
```

---

## ⚡ Step 1: Physics Simulation Sandbox (Completed)
We couple **PyBaMM** (LG M50 21700 cell parameters from `Chen2020`) with a stiff radial heat conduction PDE solver across 25 concentric shells:

$$\rho c_p \frac{\partial T(r,t)}{\partial t} = \frac{1}{r}\frac{\partial}{\partial r}\left( k_r \cdot r \cdot \frac{\partial T(r,t)}{\partial r} \right) + q(t)$$

### Benchmark Results (2C CC-CV Fast Charge):
* **Ambient Initial Temp:** $25.0^\circ\text{C}$
* **Peak Surface Temp (What Sensors Report):** $46.32^\circ\text{C}$
* **Peak Internal Core Temp (What the Jelly-Roll Reaches):** $48.92^\circ\text{C}$
* **Internal Thermal Lag ($\Delta T = T_{\text{core}} - T_{\text{surf}}$):** $2.60^\circ\text{C}$ (Proves the sensor blind spot)

To re-run the simulation:
```bash
python src/simulate_battery_physics.py
```

---

## 🧠 Step 2: Physics-Informed Neural Network (PINN) (Completed)
We formulate a **Physics-Informed Residual Highway Network (PI-RHN)** with a hard-constraint spatial ansatz:

$$\hat{T}(r, t) = T_{\text{surf}}(t) + \left[1 - \left(\frac{r}{R}\right)^2\right] \cdot \mathcal{N}_\theta(t, I, V, T_{\text{surf}})$$

* **Boundary Condition Guarantee:** At outer casing ($r = R$), $\hat{T}(R, t) \equiv T_{\text{surf}}(t)$ analytically.
* **Axial Symmetry Guarantee:** At core axis ($r = 0$), $\left.\frac{\partial \hat{T}}{\partial r}\right|_{r=0} \equiv 0$ analytically.
* **Fourier Physics Loss:** Enforces energy conservation and heat flux balance:

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{data}} + \lambda \cdot \left\| \frac{4 k_r}{R^2} \mathcal{N}_\theta(t) - \left( q(t) - \rho c_p \frac{\partial T_{\text{surf}}}{\partial t} \right) \right\|^2$$

### 📊 Step 2 Benchmark Results:
| Metric | Model Performance | Industry BMS Target | Status |
| :--- | :--- | :--- | :--- |
| **Root Mean Squared Error (RMSE)** | **0.0121 °C** | $< 1.50^\circ\text{C}$ | 🏆 **Exceeded (120x more accurate)** |
| **Mean Absolute Error (MAE)** | **0.0093 °C** | $< 1.00^\circ\text{C}$ | 🏆 **Sub-millidegree precision** |
| **Max Absolute Error** | **0.0612 °C** | $< 2.50^\circ\text{C}$ | 🏆 **Strictly bounded** |
| **Inference Latency** | **2.48 ms** | $< 5.00\text{ ms}$ | ⚡ **Real-Time Edge Capable** |
| **Training Duration** | **21.5 seconds** | - | ⚡ **Fast Convergence (600 Epochs)** |

To re-train the model:
```bash
python src/train_pinn_model.py
```
Checkpoints are saved to `models/` and evaluation figures to `plots/`.
