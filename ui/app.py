"""Personal Telemetry - Streamlit entry point.

Run from the project root::

    streamlit run ui/app.py      (or: python cli.py ui)
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

st.set_page_config(page_title="Personal Telemetry", page_icon="📈", layout="wide")

navigation = st.navigation(
    [
        st.Page("dashboard.py", title="Dashboard", icon="📈", default=True),
        st.Page("connections.py", title="Connections", icon="🔌"),
    ]
)
navigation.run()
