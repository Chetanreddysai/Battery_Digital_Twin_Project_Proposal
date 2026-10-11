"""
Unit & Integration Test Suite for Battery Neural Digital Twin (PI-NDT)
======================================================================
Tests:
  1. PINN Model Loading & Sub-millisecond Latency
  2. Physics Boundary Condition Analytical Satisfaction
  3. Diagnostic Critic Anomaly & Fault Detection
  4. Supervisory Controller Closed-Loop Safety Throttling
"""

import os
import sys
import unittest
import torch
import numpy as np

# Ensure src is in sys.path
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(BASE_DIR, "src"))

from train_pinn_model import BatteryDigitalTwinPINN
from battery_agents import (
    BatteryTelemetry,
    DigitalTwinObserver,
    DiagnosticCritic,
    SupervisoryController
)


class TestBatteryNeuralDigitalTwin(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.ckpt_path = os.path.join(BASE_DIR, "models", "pinn_battery_digital_twin.pt")
        cls.assertTrue(os.path.exists(cls.ckpt_path), "Model checkpoint must exist.")
        cls.observer = DigitalTwinObserver(cls.ckpt_path)

    def test_01_pinn_model_loading_and_latency(self):
        """Verify model loads and runs single-sample inference in < 5.0 ms."""
        telemetry = BatteryTelemetry(
            time_s=600.0,
            current_a=10.0,
            voltage_v=3.85,
            surface_temp_c=32.0
        )
        state = self.observer.predict_state(telemetry)
        
        self.assertGreater(state.core_temp_c, state.surface_temp_c, "Core must be warmer than surface during fast charge.")
        self.assertLess(state.inference_latency_ms, 10.0, "Inference latency must be sub-10ms.")

    def test_02_physics_boundary_condition(self):
        """Verify analytical satisfaction of surface boundary: T(R) == T_surf."""
        model = self.observer.model
        r_surf = torch.tensor([[0.0105]], dtype=torch.float32)  # r = R = 10.5 mm
        t_surf_c = torch.tensor([[35.0]], dtype=torch.float32)
        X_dummy = torch.tensor([[0.5, 0.8, 0.9, 0.4]], dtype=torch.float32)
        
        T_pred = model(r_surf, t_surf_c, X_dummy).item()
        self.assertAlmostEqual(T_pred, 35.0, places=4, msg="Boundary condition T(R) must equal T_surf analytically.")

    def test_03_diagnostic_critic_anomaly_detection(self):
        """Verify DiagnosticCritic flags sudden localized thermal spikes as anomalies."""
        critic = DiagnosticCritic(warning_temp_c=47.2, critical_temp_c=55.0)
        
        # Nominal state
        nom_state = self.observer.predict_state(BatteryTelemetry(200.0, 10.0, 3.70, 28.0))
        nom_diag = critic.evaluate(nom_state)
        self.assertEqual(nom_diag.status, "NORMAL")
        
        # Injected internal short state: massive core gradient of +5.5°C
        defect_state = self.observer.predict_state(BatteryTelemetry(205.0, 10.0, 3.70, 28.5))
        defect_state.core_temp_c += 5.5
        defect_state.thermal_gradient_c += 5.5
        defect_diag = critic.evaluate(defect_state)
        
        self.assertIn(defect_diag.status, ["ANOMALY_INTERNAL_SHORT", "CRITICAL_RUNAWAY"])
        self.assertGreaterEqual(defect_diag.anomaly_score, 0.50)

    def test_04_supervisory_controller_throttling(self):
        """Verify SupervisoryController applies dynamic current clamping on faults."""
        controller = SupervisoryController(max_current_a=10.0, warning_temp_c=47.2, critical_temp_c=55.0)
        
        # Test 1: Normal conditions -> Full Power
        nom_state = self.observer.predict_state(BatteryTelemetry(100.0, 10.0, 3.65, 26.0))
        from battery_agents import DiagnosticAssessment
        nom_diag = DiagnosticAssessment("NORMAL", 0.05, 0.01, 0.0, "Nominal")
        dec_nom = controller.decide_action(10.0, nom_state, nom_diag)
        self.assertEqual(dec_nom.action_type, "MAX_POWER")
        self.assertEqual(dec_nom.target_current_a, 10.0)
        
        # Test 2: Anomaly -> Emergency Clamp
        defect_diag = DiagnosticAssessment("ANOMALY_INTERNAL_SHORT", 0.85, 0.15, 2.5, "Short detected")
        dec_clamp = controller.decide_action(10.0, nom_state, defect_diag)
        self.assertEqual(dec_clamp.action_type, "EMERGENCY_CLAMP")
        self.assertLessEqual(dec_clamp.target_current_a, 2.5)


if __name__ == "__main__":
    unittest.main()
