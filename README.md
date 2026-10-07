# Physics-Informed Neural Digital Twin (PI-NDT)
### Real-Time 3D Battery Thermal State Estimation & Autonomous Fast-Charging Safety

[![Python 3.13](https://img.shields.io/badge/Python-3.13-blue.svg)](https://www.python.org/)
[![PyBaMM](https://img.shields.io/badge/PyBaMM-26.9-brightgreen.svg)](https://pybamm.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-orange.svg)](https://pytorch.org/)

---

## 🚀 Project Overview
Lithium-ion battery packs in electric vehicles (EVs) suffer from an internal **thermal blind spot**: physical sensors can only measure the exterior metal casing ($T_{\text{surf}}$), leaving dangerous internal core hotspots ($T_{\text{core}}$) invisible during fast charging.

This project builds an AI-powered **Physics-Informed Neural Digital Twin** that acts like a real-time software X-ray:
1. Ingests non-invasive surface measurements (Current $I$, Voltage $V$, Surface Temp $T_{\text{surf}}$).
2. Uses Physics-Informed Neural Networks (PINNs) constrained by radial heat conduction PDEs to deduce invisible 3D core temperatures in $<2\text{ ms}$.
3. Deploys autonomous multi-agent supervisory logic to prevent thermal runaway without sacrificing charging speed.

---

## 📁 Repository Structure
```
battery-neural-digital-twin/
├── data/
│   └── battery_fast_charge_ground_truth.csv  # 600 time-step 2C CC-CV physics telemetry
├── plots/
│   └── thermal_blindspot_validation.png      # 4-panel benchmark & thermal lag plot
├── src/
│   ├── simulate_battery_physics.py           # Step 1: Electrochemical & thermal PDE engine
│   └── (upcoming) train_pinn_model.py        # Step 2: Physics-Informed Neural Network
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
Outputs are automatically written to `data/` and `plots/`.
