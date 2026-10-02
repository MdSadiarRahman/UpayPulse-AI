"""
app.py
======
UpayPulse AI — Streamlit MFS Intelligence Dashboard (Root Entry Point)

Forwards execution to dashboard/app.py or runs the unified dashboard.
"""

from pathlib import Path
import sys

# Ensure dashboard directory is in sys.path
DASHBOARD_DIR = Path(__file__).resolve().parent / "dashboard"
if str(DASHBOARD_DIR) not in sys.path:
    sys.path.insert(0, str(DASHBOARD_DIR))

# Execute the main dashboard code
with open(DASHBOARD_DIR / "app.py", "r", encoding="utf-8") as f:
    code = f.read()

exec(code)
