"""
KineticGuard - Dashboard CLI Runner (Phase 6)
Starts the real-time Autonomous Safety Intelligence dashboard server.

Usage:
  python dashboard.py --port 8080 --mock
  python dashboard.py --port 8080           # Connects to real AWS backend
"""

import argparse
import sys
from src.dashboard_server import run_dashboard_server
from src.config import settings

import os

def main():
    default_host = os.getenv("HOST", "0.0.0.0")
    default_port = int(os.getenv("PORT", "8080"))

    parser = argparse.ArgumentParser(description="KineticGuard Autonomous Safety Intelligence Dashboard")
    parser.add_argument("--host", type=str, default=default_host, help=f"Host interface (default: {default_host})")
    parser.add_argument("--port", type=int, default=default_port, help=f"Port to listen on (default: {default_port})")
    parser.add_argument("--mock", action="store_true", default=False, help="Force local mock mode (zero cost)")
    args = parser.parse_args()

    mock_mode = args.mock or not settings.has_aws_credentials()
    if mock_mode and not args.mock:
        print("[Dashboard] Notice: Local Build It mode active (zero AWS cloud dependency).")

    run_dashboard_server(host=args.host, port=args.port, mock_mode=mock_mode)

if __name__ == "__main__":
    main()
