# Physics-Informed Neural Digital Twin (PI-NDT)
### Real-Time 3D Battery Thermal State Estimation & Autonomous Fast-Charging Safety

[![Python 3.13](https://img.shields.io/badge/Python-3.13-blue.svg)](https://www.python.org/)
[![PyBaMM](https://img.shields.io/badge/PyBaMM-26.9-brightgreen.svg)](https://pybamm.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-orange.svg)](https://pytorch.org/)
[![NASA PCoE](https://img.shields.io/badge/NASA%20PCoE-Validated-blueviolet.svg)](data/nasa_battery_benchmark_results.csv)
[![Multi-Agent](https://img.shields.io/badge/Multi--Agent-Autonomous%20BMS-purple.svg)](src/battery_agents.py)
[![Streamlit](https://img.shields.io/badge/Dashboard-Streamlit%203D-red.svg)](app.py)
[![Tests](https://img.shields.io/badge/Tests-Passing%20(4%2F4)-brightgreen.svg)](tests/test_digital_twin.py)
[![RMSE](https://img.shields.io/badge/Core%20RMSE-0.0121%C2%B0C-brightgreen.svg)](models/pinn_metrics.json)

---

## 🚀 Project Overview
Lithium-ion battery packs in electric vehicles (EVs) suffer from an internal **thermal blind spot**: physical sensors can only measure the exterior metal casing ($T_{\text{surf}}$), leaving dangerous internal core hotspots ($T_{\text{core}}$) invisible during fast charging.

This project builds an AI-powered **Physics-Informed Neural Digital Twin** that acts like a real-time software X-ray:
1. Ingests non-invasive surface measurements (Current $I$, Voltage $V$, Surface Temp $T_{\text{surf}}$).
2. Uses a **Physics-Informed Residual Highway Network (PI-RHN)** constrained by radial heat conduction PDEs to deduce invisible 3D core temperatures in real time without interior sensors.
3. Deploys an **Autonomous Multi-Agent System** that monitors physical consistency, detects internal short circuits in $<5\text{ ms}$, and dynamically modulates charging current to prevent thermal runaway.
4. Provides an **Interactive 3D Web Dashboard** with real-time 3D battery cutaway rendering, fault injection testing, and live agent decision telemetry.
5. Evaluates out-of-distribution transferability against real experimental datasets from the **NASA Ames Prognostics Center of Excellence (PCoE)**.

---

## 📁 Repository Structure
```
battery-neural-digital-twin/
├── app.py                                      # Step 4: Root entrypoint for Streamlit Dashboard
├── data/
│   ├── battery_fast_charge_ground_truth.csv    # Step 1: 600-step 2C CC-CV physics telemetry
│   ├── multi_agent_simulation_logs.csv         # Step 3: Closed-loop multi-agent streaming logs
│   └── nasa_battery_benchmark_results.csv      # Step 5: NASA B0005 experimental benchmark
├── docs/
│   ├── Battery_Digital_Twin_Project_Proposal.pdf # Executive specification PDF
│   └── project_proposal_battery_digital_twin.md  # Markdown proposal & architecture
├── models/
│   ├── pinn_battery_digital_twin.pt            # Trained PyTorch neural model checkpoint
│   ├── pinn_metrics.json                       # Synthetic 21700 benchmark metrics
│   └── nasa_benchmark_metrics.json             # NASA B0005 benchmark metrics
├── plots/
│   ├── thermal_blindspot_validation.png        # Step 1: 4-panel physics validation plot
│   ├── pinn_core_inference_benchmark.png       # Step 2: 4-panel PINN performance benchmark
│   ├── multi_agent_supervisory_benchmark.png   # Step 3: 4-panel multi-agent control benchmark
│   └── nasa_benchmark_validation.png           # Step 5: 4-panel NASA experimental benchmark
├── src/
│   ├── app_dashboard.py                        # Step 4: Interactive 3D Streamlit Dashboard
│   ├── simulate_battery_physics.py             # Step 1: PyBaMM & radial thermal PDE engine
│   ├── train_pinn_model.py                     # Step 2: PINN architecture & training pipeline
│   ├── battery_agents.py                       # Step 3: 4 Cooperating autonomous agents
│   ├── run_multi_agent_simulation.py           # Step 3: Real-time closed-loop runner
│   └── benchmark_nasa_data.py                  # Step 5: NASA Ames PCoE evaluation engine
├── tests/
│   └── test_digital_twin.py                    # Step 5: Unit & integration test suite (4/4 Pass)
├── README.md
└── requirements.txt
```

---

## 🏆 Dual Benchmark Performance Matrix

| Evaluation Benchmark | Cell Type | Charging Protocol | Core RMSE | Core MAE | Max Absolute Error | Inference Latency | Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Synthetic Physics Ground Truth (PyBaMM)** | LG M50 (21700, 5.0 Ah) | 2.0C CC-CV (10.0 A) | **0.0121 °C** | **0.0093 °C** | **0.0612 °C** | **0.658 ms** | 🏆 **120× Target** |
| **NASA Ames PCoE Experimental (B0005)** | 18650 Cell (2.0 Ah) | 0.75C CC-CV (1.5 A) | **0.3128 °C** | **0.2506 °C** | **0.6047 °C** | **0.174 ms** | 🏆 **High Generalization** |

---

## ⚡ Quickstart Guide

### 1. Clone & Install Dependencies
```bash
git clone https://github.com/Chetanreddysai/Battery_Digital_Twin_Project_Proposal.git
cd Battery_Digital_Twin_Project_Proposal
pip install -r requirements.txt
```

### 2. Run the Interactive 3D Dashboard
```bash
streamlit run app.py
```

### 3. Run Automated Unit Tests (4/4 Tests)
```bash
python -m unittest tests/test_digital_twin.py
```

### 4. Run NASA Ames PCoE Benchmark
```bash
python src/benchmark_nasa_data.py
```

---

## 🔬 Scientific Foundations & Methodology

### 1. Hard-Constraint Physics Ansatz
To guarantee exact boundary compliance without numerical optimization drift, the neural network satisfies boundary conditions analytically:

$$\hat{T}(r, t) = T_{\text{surf}}(t) + \left[1 - \left(\frac{r}{R}\right)^2\right] \cdot \mathcal{N}_\theta(t, I, V, T_{\text{surf}})$$

* **Surface Boundary:** At $r = R$, $\hat{T}(R, t) \equiv T_{\text{surf}}(t)$ analytically with zero error.
* **Axial Symmetry:** At $r = 0$, $\left.\frac{\partial \hat{T}}{\partial r}\right|_{r=0} \equiv 0$ analytically.

### 2. Fourier Radial Heat Conduction Loss
The interior weights of $\mathcal{N}_\theta$ are trained using backpropagation by penalizing deviations from Fourier's radial heat diffusion law:

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{data}} + \lambda \cdot \left\| \frac{4 k_r}{R^2} \mathcal{N}_\theta(t) - \left( q(t) - \rho c_p \frac{\partial T_{\text{surf}}}{\partial t} \right) \right\|^2$$

---

## 🤖 Cooperating Autonomous Agents Architecture

```
[Vehicle Telemetry] (Current I, Voltage V, Surface Temp T_surf)
        │
        ▼
┌────────────────────────────────────────────────────────┐
│  Agent 1: DigitalTwinObserver (PINN Neural Engine)     │  ──> Deduces 3D core temp in < 0.7 ms
└────────────────────────────────────────────────────────┘
        │
        ▼
┌────────────────────────────────────────────────────────┐
│  Agent 2: DiagnosticCritic (Unsupervised Physics)      │  ──> Detects micro-shorts & gradient drift
└────────────────────────────────────────────────────────┘
        │
        ▼
┌────────────────────────────────────────────────────────┐
│  Agent 3: SupervisoryController (Closed-Loop Policy)   │  ──> Modulates charging current & coolant
└────────────────────────────────────────────────────────┘
        │
        ▼
┌────────────────────────────────────────────────────────┐
│  Agent 4: TelemetryExplainer (Diagnostics & Audit Log) │  ──> Outputs explainable engineering rationale
└────────────────────────────────────────────────────────┘
```

* **Observer Step Latency:** 0.658 ms (Sub-millisecond)
* **Autonomous Interventions:** 83 dynamic control events
* **Fault Detection Response:** $< 5.0\text{ ms}$ (Immediate trickle clamp)
* **Thermal Runaway Incidents:** 0 (100% Prevented)
* **Total Energy Delivered:** 18.35 Wh

---

## 🛠️ Phase 2: Embedded Hardware Deployment Roadmap
* **Model Checkpoint Size:** `< 180 KB`
* **Target Edge MCUs:** ESP32 (\$6), Raspberry Pi Pico (\$5), or STM32 ARM Cortex-M4
* **Quantization Format:** `ONNX Runtime` & `TensorFlow Lite for Microcontrollers (TFLite Micro)`
* **Physical Sensors:** 2× DS18B20 1-Wire Digital Thermistors + INA219 Voltage/Current I2C Monitor
