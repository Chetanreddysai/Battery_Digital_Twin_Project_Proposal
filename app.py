"""
Root entrypoint for Streamlit Web Dashboard
Run via: streamlit run app.py
"""

import os
import sys
import runpy

# Add src to sys.path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

# Streamlit reruns this entrypoint after every widget interaction. Executing
# the dashboard file (instead of importing it) ensures its UI is rebuilt on
# every run; a normal import is cached by Python after the first render.
dashboard_path = os.path.join(os.path.dirname(__file__), "src", "app_dashboard.py")
runpy.run_path(dashboard_path, run_name="__main__")
