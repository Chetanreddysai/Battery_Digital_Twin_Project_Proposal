"""
Physics Simulation Sandbox: Battery Fast-Charging & Thermal Gradient Simulator
Couples PyBaMM electrochemical simulation (Chen2020 21700 cell) with a 
1D radial heat conduction PDE solver to capture internal core vs surface thermal dynamics.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import pybamm

def run_electrochemical_simulation(c_rate=2.0, ambient_temp_c=25.0, initial_soc=0.05):
    """
    Simulates high-rate CC-CV fast charging using PyBaMM (Chen2020 parameter set).
    Returns uniform time series of Voltage [V], Current [A], and Volumetric Heat Generation [W/m^3].
    """
    print(f"[1/4] Initializing PyBaMM Electrochemical Model at {c_rate}C charge rate...")
    
    # Use Single Particle Model with electrolyte (SPMe) with lumped thermal coupling
    model = pybamm.lithium_ion.SPMe(options={"thermal": "lumped"})
    param = pybamm.ParameterValues("Chen2020")
    
    capacity = param["Nominal cell capacity [A.h]"]
    charge_current = c_rate * capacity  # e.g., 2.0 * 5.0 Ah = 10.0 A
    
    # Configure thermal parameters
    param["Ambient temperature [K]"] = ambient_temp_c + 273.15
    param["Initial temperature [K]"] = ambient_temp_c + 273.15
    
    # Define realistic CC-CV Fast-Charging Experiment
    experiment = pybamm.Experiment([
        f"Charge at {charge_current:.1f}A until 4.2V",
        "Hold at 4.2V until 1A"
    ])
    
    print("[2/4] Solving electrochemical CC-CV equations...")
    sim = pybamm.Simulation(model, parameter_values=param, experiment=experiment)
    solution = sim.solve(initial_soc=initial_soc)
    
    raw_t = solution["Time [s]"].data.flatten()
    raw_v = solution["Terminal voltage [V]"].data.flatten()
    raw_i = np.abs(solution["Current [A]"].data.flatten())
    raw_q = solution["Volume-averaged total heating [W.m-3]"].data.flatten()
    
    # Resample onto uniform time grid (every 2 seconds) for consistent time-series modeling
    t_uniform = np.linspace(raw_t[0], raw_t[-1], 600)
    v_uniform = np.interp(t_uniform, raw_t, raw_v)
    i_uniform = np.interp(t_uniform, raw_t, raw_i)
    q_uniform = np.interp(t_uniform, raw_t, raw_q)
    
    print(f"      Simulation finished: {raw_t[-1]:.1f} seconds total charging time ({len(t_uniform)} uniform steps).")
    return t_uniform, v_uniform, i_uniform, q_uniform, ambient_temp_c


def solve_radial_heat_pde(t, q_vol, ambient_temp_c=25.0, n_shells=25):
    """
    Solves the 1D cylindrical radial heat conduction PDE:
        rho * cp * dT/dt = (1/r) * d/dr [ k_r * r * dT/dr ] + q(t)
    
    Boundary Conditions:
        r = 0 (Center/Core):  dT/dr = 0 (axial symmetry)
        r = R (Surface):      -k_r * dT/dr = h * (T_surf - T_amb) (convective cooling)
    """
    print(f"[3/4] Solving 1D Radial Heat PDE across {n_shells} concentric shells...")
    
    # Dimensions of 21700 cylindrical battery cell
    radius_m = 0.0105      # 10.5 mm radius
    height_m = 0.070       # 70 mm height
    
    # Thermal properties
    rho = 2500.0           # Density [kg/m^3]
    cp = 1050.0            # Specific heat capacity [J / (kg*K)]
    k_r = 0.95             # Low radial thermal conductivity [W / (m*K)] across wound layers
    h_conv = 22.0          # Convective heat transfer coefficient [W / (m^2*K)]
    
    # Spatial discretization along radius
    r = np.linspace(0, radius_m, n_shells)
    dr = r[1] - r[0]
    
    def get_q(t_curr):
        return np.interp(t_curr, t, q_vol)
    
    # Semi-discretized ODE system
    def heat_equation_system(t_curr, T):
        dT_dt = np.zeros(n_shells)
        q_now = get_q(t_curr)
        
        # Center core (r = 0): L'Hopital limit -> 2 * d^2T/dr^2
        d2T_dr2_center = 2.0 * (T[1] - T[0]) / (dr ** 2)
        dT_dt[0] = (k_r * d2T_dr2_center + q_now) / (rho * cp)
        
        # Interior shells (0 < r < R)
        for i in range(1, n_shells - 1):
            r_i = r[i]
            dT_dr_fwd = (T[i + 1] - T[i]) / dr
            dT_dr_bwd = (T[i] - T[i - 1]) / dr
            
            flux_out = (r_i + 0.5 * dr) * dT_dr_fwd
            flux_in  = (r_i - 0.5 * dr) * dT_dr_bwd
            div_flux = (flux_out - flux_in) / (r_i * dr)
            
            dT_dt[i] = (k_r * div_flux + q_now) / (rho * cp)
            
        # Surface boundary (r = R): Convective heat loss to ambient air
        r_surf = r[-1]
        T_surf = T[-1]
        T_ghost = T[-2] - 2.0 * dr * (h_conv / k_r) * (T_surf - ambient_temp_c)
        div_flux_surf = ((r_surf + 0.5 * dr) * (T_ghost - T_surf) / dr - 
                         (r_surf - 0.5 * dr) * (T_surf - T[-2]) / dr) / (r_surf * dr)
        dT_dt[-1] = (k_r * div_flux_surf + q_now) / (rho * cp)
        
        return dT_dt

    T_init = np.full(n_shells, ambient_temp_c)
    
    sol = solve_ivp(
        heat_equation_system,
        t_span=(t[0], t[-1]),
        y0=T_init,
        t_eval=t,
        method="Radau"  # Stiff implicit Runge-Kutta method
    )
    
    T_profile = sol.y  # Shape: (n_shells, len(t))
    return r, T_profile


def save_and_plot_results(t, v, i, q_vol, r, T_profile, output_dir):
    """
    Saves the ground-truth dataset to CSV and generates a publication-quality validation plot.
    """
    print("[4/4] Exporting ground-truth dataset and generating validation plot...")
    
    core_temp = T_profile[0, :]       # r = 0 mm (Core)
    mid_temp = T_profile[len(r)//2, :] # r = 5.25 mm (Mid-layer)
    surf_temp = T_profile[-1, :]      # r = 10.5 mm (Surface casing)
    delta_T = core_temp - surf_temp    # The invisible thermal lag gap
    
    # 1. Create DataFrame and export to CSV
    df = pd.DataFrame({
        "time_s": t,
        "current_a": i,
        "voltage_v": v,
        "heat_gen_w_m3": q_vol,
        "surface_temp_c": surf_temp,
        "mid_temp_c": mid_temp,
        "core_temp_c": core_temp,
        "thermal_gradient_delta_t_c": delta_T
    })
    
    os.makedirs(os.path.join(output_dir, "data"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "plots"), exist_ok=True)
    
    csv_path = os.path.join(output_dir, "data", "battery_fast_charge_ground_truth.csv")
    df.to_csv(csv_path, index=False)
    print(f"      Saved ground-truth dataset: {csv_path} ({len(df)} records)")
    
    # 2. Generate Publication-Quality Figure
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Physics Simulation Sandbox: 2C Fast-Charging & Internal Thermal Blind Spot", fontsize=14, fontweight="bold")
    
    # Subplot A: Electrical Charging Curve
    ax1 = axs[0, 0]
    color_i = "tab:blue"
    ax1.set_xlabel("Time (s)", fontsize=10)
    ax1.set_ylabel("Current (A)", color=color_i, fontsize=10)
    ax1.plot(t, i, color=color_i, linewidth=2.2, label="Charge Current")
    ax1.tick_params(axis="y", labelcolor=color_i)
    ax1.grid(True, linestyle="--", alpha=0.5)
    
    ax1_twin = ax1.twinx()
    color_v = "tab:purple"
    ax1_twin.set_ylabel("Terminal Voltage (V)", color=color_v, fontsize=10)
    ax1_twin.plot(t, v, color=color_v, linewidth=2.2, linestyle="--", label="Voltage")
    ax1_twin.tick_params(axis="y", labelcolor=color_v)
    axs[0, 0].set_title("A. CC-CV Electrical Charging Protocol (10A -> 4.2V)", fontsize=11, fontweight="bold")
    
    # Subplot B: Volumetric Heat Generation Rate
    axs[0, 1].plot(t, q_vol / 1e3, color="darkorange", linewidth=2.2)
    axs[0, 1].set_title("B. Internal Volumetric Heat Generation Rate", fontsize=11, fontweight="bold")
    axs[0, 1].set_xlabel("Time (s)", fontsize=10)
    axs[0, 1].set_ylabel("Heat Generation (kW/m³)", fontsize=10)
    axs[0, 1].grid(True, linestyle="--", alpha=0.5)
    
    # Subplot C: Surface vs Core Temperature (The Blind Spot Gap)
    axs[1, 0].plot(t, surf_temp, color="teal", linewidth=2.5, label="Surface Temp T_surf (Outer Sensor)")
    axs[1, 0].plot(t, core_temp, color="crimson", linewidth=2.5, linestyle="-", label="Core Temp T_core (Internal Jelly-Roll)")
    axs[1, 0].fill_between(t, surf_temp, core_temp, color="red", alpha=0.15, label="Dangerous Thermal Lag Gap")
    axs[1, 0].axhline(y=55.0, color="darkred", linestyle=":", linewidth=1.8, label="Safety Warning Threshold (55°C)")
    axs[1, 0].set_title("C. The Sensor Blind Spot (Core vs Surface Temperature)", fontsize=11, fontweight="bold")
    axs[1, 0].set_xlabel("Time (s)", fontsize=10)
    axs[1, 0].set_ylabel("Temperature (°C)", fontsize=10)
    axs[1, 0].legend(loc="upper left", fontsize=8.5)
    axs[1, 0].grid(True, linestyle="--", alpha=0.5)
    
    # Subplot D: Radial Cross-Section Temperature at Peak Heat
    peak_idx = np.argmax(core_temp)
    peak_time = t[peak_idx]
    axs[1, 1].plot(r * 1000.0, T_profile[:, peak_idx], marker="o", color="firebrick", linewidth=2.2)
    axs[1, 1].set_title(f"D. Radial Temperature Profile at Peak Heat (t = {peak_time:.0f} s)", fontsize=11, fontweight="bold")
    axs[1, 1].set_xlabel("Radial Distance from Center (mm)", fontsize=10)
    axs[1, 1].set_ylabel("Temperature (°C)", fontsize=10)
    axs[1, 1].axvline(x=0, color="gray", linestyle="--", alpha=0.7, label="Center Core (r = 0 mm)")
    axs[1, 1].axvline(x=r[-1]*1000, color="black", linestyle="--", alpha=0.7, label="Outer Casing (r = 10.5 mm)")
    axs[1, 1].legend(loc="upper right", fontsize=8.5)
    axs[1, 1].grid(True, linestyle="--", alpha=0.5)
    
    plt.tight_layout()
    plot_path = os.path.join(output_dir, "plots", "thermal_blindspot_validation.png")
    plt.savefig(plot_path, dpi=200)
    plt.close()
    print(f"      Saved validation plot: {plot_path}")
    
    # Telemetry summary
    print("\n================= SIMULATION BENCHMARK SUMMARY =================")
    print(f"  Ambient Temperature:               {surf_temp[0]:.1f} °C")
    print(f"  Peak Surface Temperature:           {surf_temp.max():.2f} °C (Sensors report MILD)")
    print(f"  Peak Core Temperature:              {core_temp.max():.2f} °C (Internal Jelly-Roll)")
    print(f"  Max Internal Thermal Lag (Delta T): {delta_T.max():.2f} °C (The Invisible Danger Gap)")
    print("================================================================")
    
    return csv_path, plot_path


if __name__ == "__main__":
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    t, v, i, q_vol, amb = run_electrochemical_simulation(c_rate=2.0, ambient_temp_c=25.0, initial_soc=0.05)
    r, T_profile = solve_radial_heat_pde(t, q_vol, ambient_temp_c=amb, n_shells=25)
    save_and_plot_results(t, v, i, q_vol, r, T_profile, base_dir)
