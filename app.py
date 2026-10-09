"""
Root entrypoint for Streamlit Web Dashboard
Run via: streamlit run app.py
"""

import os
import sys

# Add src to sys.path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

# Execute dashboard
import app_dashboard
