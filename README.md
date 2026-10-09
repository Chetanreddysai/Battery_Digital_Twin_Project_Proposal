# Physics-Informed Neural Digital Twin (PI-NDT)
### Real-Time 3D Battery Thermal State Estimation & Autonomous Fast-Charging Safety

[![Python 3.13](https://img.shields.io/badge/Python-3.13-blue.svg)](https://www.python.org/)
[![PyBaMM](https://img.shields.io/badge/PyBaMM-26.9-brightgreen.svg)](https://pybamm.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-orange.svg)](https://pytorch.org/)
[![Multi-Agent](https://img.shields.io/badge/Multi--Agent-Autonomous%20BMS-purple.svg)](src/battery_agents.py)
[![Streamlit](https://img.shields.io/badge/Dashboard-Streamlit%203D-red.svg)](app.py)
[![RMSE](https://img.shields.io/badge/Core%20RMSE-0.0121%C2%B0C-brightgreen.svg)](models/pinn_metrics.json)

---

## 🚀 Project Overview
Lithium-ion battery packs in electric vehicles (EVs) suffer from an internal **thermal blind spot**: physical sensors can only measure the exterior metal casing ($T_{\text{surf}}$), leaving dangerous internal core hotspots ($T_{\text{core}}$) invisible during fast charging.

This project builds an AI-powered **Physics-Informed Neural Digital Twin** that acts like a real-time software X-ray:
1. Ingests non-invasive surface measurements (Current $I$, Voltage $V$, Surface Temp $T_{\text{surf}}$).
2. Uses a **Physics-Informed Residual Highway Network (PI-RHN)** constrained by radial heat conduction PDEs to deduce invisible 3D core temperatures in real time without interior sensors.
3. Deploys an **Autonomous Multi-Agent System** that monitors physical consistency, detects internal short circuits in $<5\text{ ms}$, and dynamically modulates charging current to prevent thermal runaway.
4. Provides an **Interactive 3D Web Dashboard** with real-time 3D battery cutaway rendering, fault injection testing, and live agent decision telemetry.

---

## 📁 Repository Structure
```
battery-neural-digital-twin/
├── app.py                                      # Step 4: Root entrypoint for Streamlit Dashboard
├── data/
│   ├── battery_fast_charge_ground_truth.csv    # 600 time-step 2C CC-CV physics telemetry
│   └── multi_agent_simulation_logs.csv         # Closed-loop multi-agent streaming logs
├── docs/
│   ├── Battery_Digital_Twin_Project_Proposal.pdf # Executive specification PDF
│   └── project_proposal_battery_digital_twin.md  # Markdown proposal & architecture
├── models/
│   ├── pinn_battery_digital_twin.pt            # Trained PyTorch neural model checkpoint
│   └── pinn_metrics.json                       # Quantitative benchmark metrics
├── plots/
│   ├── thermal_blindspot_validation.png        # Step 1: 4-panel physics validation plot
│   ├── pinn_core_inference_benchmark.png       # Step 2: 4-panel PINN performance benchmark
│   └── multi_agent_supervisory_benchmark.png   # Step 3: 4-panel multi-agent control benchmark
├── src/
│   ├── app_dashboard.py                        # Step 4: Interactive 3D Streamlit Dashboard
│   ├── simulate_battery_physics.py             # Step 1: PyBaMM & radial thermal PDE engine
│   ├── train_pinn_model.py                     # Step 2: PINN architecture & training pipeline
│   ├── battery_agents.py                       # Step 3: 4 Cooperating autonomous agents
│   └── run_multi_agent_simulation.py           # Step 3: Real-time closed-loop runner
├── README.md
└── requirements.txt
```

---

## 🌐 Step 4: Interactive 3D Web Dashboard (Completed)

To launch the interactive dashboard locally:
```bash
streamlit run app.py
```

### Dashboard Features:
* **Interactive 3D Cylindrical Cutaway (Plotly 3D):** Rotate, pan, and zoom a full 3D visual rendering of the LG M50 21700 cell, visualizing the molten internal core temperature field reconstructed by the PINN in real time.
* **Continuous Radial Cross-Section Curve $T(r)$:** Real-time 2D temperature profile from center axis ($r = 0\text{ mm}$) to outer casing ($r = 10.5\text{ mm}$).
* **Telemetry Scrubber & Fast-Charging Scenarios:** Scrub through the 50-minute fast-charging cycle or jump directly to peak-heat events.
* **Live Fault & Anomaly Injection:** Toggle simulated internal micro-shorts to observe the Diagnostic Critic and Supervisory Controller clamping current to trickle charge within $<5\text{ ms}$.
* **Live Cooperating Agent Bus:** Status, inference latency, and action logs for all 4 agents.

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

---

## 🤖 Step 3: Autonomous Multi-Agent Supervisory Control (Completed)
The system deploys four cooperating agents collaborating over a streaming telemetry bus:
* **Agent 1: DigitalTwinObserver** (Deduces 3D core temp in **0.658 ms**)
* **Agent 2: DiagnosticCritic** (Detects micro-shorts & gradient drift in **$<5\text{ ms}$**)
* **Agent 3: SupervisoryController** (Modulates charging current & coolant demand)
* **Agent 4: TelemetryExplainer** (Outputs explainable engineering audit logs)

### 📊 Step 3 Benchmark:
* **Observer Latency:** 0.658 ms (Sub-millisecond)
* **Autonomous Interventions:** 83 dynamic control events
* **Thermal Runaway Incidents:** 0 (100% Prevented)
* **Total Energy Delivered:** 18.35 Wh

To run the multi-agent closed-loop simulation:
```bash
python src/run_multi_agent_simulation.py
```
