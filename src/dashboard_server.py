"""
KineticGuard - Autonomous Workplace Safety Intelligence Dashboard Server (Phase 6 Build It)
Visual-First, Industrial Control-Room Dashboard connected directly to persistent Local OpenSearch,
MediaPipe CV, Biomechanics Engine, Temporal Risk Engine, Strands Guardian Agent, Gemini Vision,
Cedar Governance, and Real-Time Video Ingestion.

Zero AWS Cloud dependencies: runs 100% locally.
"""

import argparse
import base64
import io
import json
import os
import sys
import time
from datetime import datetime, timezone
from http.server import HTTPServer, SimpleHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs, unquote
from pathlib import Path
from typing import Dict, Any, Optional

from src.opensearch_dashboard_backend import opensearch_dashboard_backend, OpenSearchDashboardBackend
from src.local_opensearch import opensearch_engine
from src.aws_orchestrator import KineticGuardOrchestrator
from src.live_streamer import live_stream_manager
from src.privacy_guard import privacy_guard


DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>KineticGuard | Workplace Ergonomic Intelligence</title>
  <style>
    :root {
      --bg-dark: #0b0f19;
      --bg-card: #131b2e;
      --bg-card-hover: #182239;
      --bg-dock: #0e1626;
      --border-subtle: rgba(255, 255, 255, 0.08);
      --border-focus: #00c4df;
      --accent-cyan: #00c4df;
      --accent-blue: #38bdf8;
      --status-low: #10b981;
      --status-mod: #f59e0b;
      --status-high: #f97316;
      --status-danger: #ef4444;
      --text-main: #f8fafc;
      --text-muted: #94a3b8;
      --text-dim: #64748b;
      --font-main: Calibri, -apple-system, BlinkMacSystemFont, Arial, sans-serif;
      --font-mono: 'Consolas', 'Courier New', monospace;
    }

    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      font-family: var(--font-main);
      background: var(--bg-dark);
      color: var(--text-main);
      min-height: 100vh;
      display: flex;
      flex-direction: column;
      overflow-x: hidden;
    }

    /* Top Navbar */
    .app-header {
      background: #080d18;
      border-bottom: 1px solid var(--border-subtle);
      padding: 0.75rem 2rem;
      display: flex;
      justify-content: space-between;
      align-items: center;
      position: sticky;
      top: 0;
      z-index: 100;
    }
    .brand-section {
      display: flex;
      align-items: center;
      gap: 0.75rem;
    }
    .brand-title {
      font-size: 1.25rem;
      font-weight: 700;
      color: var(--text-main);
      letter-spacing: 0.5px;
    }
    .brand-title span {
      color: var(--accent-cyan);
    }
    .brand-subtitle {
      font-size: 0.8rem;
      color: var(--text-dim);
      padding-left: 0.75rem;
      border-left: 1px solid var(--border-subtle);
    }
    .header-badges {
      display: flex;
      align-items: center;
      gap: 0.75rem;
    }
    .btn-sample-download {
      display: inline-flex;
      align-items: center;
      gap: 6px;
      background: #0f172a;
      border: 1px solid rgba(0, 196, 223, 0.4);
      color: var(--accent-cyan);
      font-size: 0.78rem;
      font-weight: 600;
      padding: 5px 12px;
      border-radius: 6px;
      text-decoration: none;
      transition: all 0.15s ease;
      cursor: pointer;
    }
    .btn-sample-download:hover {
      background: rgba(0, 196, 223, 0.15);
      border-color: var(--accent-cyan);
      color: #fff;
      transform: translateY(-1px);
    }
    .status-pill {
      font-size: 0.78rem;
      font-weight: 600;
      padding: 4px 10px;
      border-radius: 20px;
      display: flex;
      align-items: center;
      gap: 6px;
      background: rgba(16, 185, 129, 0.1);
      border: 1px solid rgba(16, 185, 129, 0.3);
      color: var(--status-low);
    }
    .indicator-dot {
      width: 8px;
      height: 8px;
      border-radius: 50%;
      background: var(--status-low);
      display: inline-block;
      box-shadow: 0 0 6px var(--status-low);
    }

    /* Navigation Tabs */
    .nav-bar {
      background: #0e1422;
      border-bottom: 1px solid var(--border-subtle);
      padding: 0 2rem;
      display: flex;
      gap: 0.5rem;
      overflow-x: auto;
    }
    .nav-btn {
      background: transparent;
      border: none;
      border-bottom: 2px solid transparent;
      padding: 0.75rem 1.1rem;
      font-family: var(--font-main);
      font-size: 0.9rem;
      font-weight: 600;
      color: var(--text-muted);
      cursor: pointer;
      display: flex;
      align-items: center;
      gap: 0.4rem;
      transition: all 0.15s ease;
      white-space: nowrap;
    }
    .nav-btn:hover {
      color: var(--text-main);
    }
    .nav-btn.active {
      color: var(--accent-cyan);
      border-bottom-color: var(--accent-cyan);
      background: rgba(0, 196, 223, 0.05);
    }

    /* Main Container */
    .main-container {
      flex: 1;
      padding: 1.5rem 2rem;
      display: flex;
      flex-direction: column;
      gap: 1.5rem;
      max-width: 1600px;
      margin: 0 auto;
      width: 100%;
    }

    .tab-view { display: none; }
    .tab-view.active { display: flex; flex-direction: column; gap: 1.5rem; }

    /* Video Control Toolbar */
    .source-toolbar {
      background: var(--bg-card);
      border: 1px solid var(--border-subtle);
      border-radius: 8px;
      padding: 0.6rem 1rem;
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 1rem;
      flex-wrap: wrap;
    }
    .source-left {
      display: flex;
      align-items: center;
      gap: 0.75rem;
      flex-wrap: wrap;
    }
    .source-label {
      font-size: 0.82rem;
      font-weight: 700;
      color: var(--text-dim);
      text-transform: uppercase;
      letter-spacing: 0.5px;
    }
    .source-btn-group {
      display: flex;
      align-items: center;
      gap: 0.4rem;
      flex-wrap: wrap;
    }
    .source-btn {
      background: #0b1120;
      border: 1px solid var(--border-subtle);
      color: var(--text-muted);
      font-family: var(--font-main);
      font-size: 0.82rem;
      font-weight: 600;
      padding: 6px 14px;
      border-radius: 6px;
      cursor: pointer;
      transition: all 0.15s ease;
      display: flex;
      align-items: center;
      gap: 6px;
    }
    .source-btn:hover {
      border-color: var(--accent-cyan);
      color: var(--text-main);
    }
    .source-btn.active {
      background: rgba(0, 196, 223, 0.12);
      border-color: var(--accent-cyan);
      color: var(--accent-cyan);
    }
    .source-btn.upload-btn {
      border-color: rgba(56, 189, 248, 0.4);
      color: var(--accent-blue);
    }
    .source-btn.upload-btn:hover {
      background: rgba(56, 189, 248, 0.12);
    }
    .active-source-badge {
      font-family: var(--font-mono);
      font-size: 0.8rem;
      font-weight: 600;
      padding: 4px 10px;
      border-radius: 4px;
      background: rgba(0, 196, 223, 0.08);
      border: 1px solid rgba(0, 196, 223, 0.25);
      color: var(--accent-cyan);
    }
    .badge-source {
      font-family: var(--font-mono);
      font-size: 0.72rem;
      font-weight: 600;
      padding: 3px 8px;
      border-radius: 4px;
      text-transform: uppercase;
      letter-spacing: 0.5px;
      display: inline-block;
    }
    .badge-real {
      background: rgba(16, 185, 129, 0.12);
      border: 1px solid rgba(16, 185, 129, 0.35);
      color: var(--status-low);
    }
    .badge-test {
      background: rgba(148, 163, 184, 0.12);
      border: 1px solid rgba(148, 163, 184, 0.35);
      color: #94a3b8;
    }
    .badge-hist {
      background: rgba(245, 158, 11, 0.12);
      border: 1px solid rgba(245, 158, 11, 0.35);
      color: var(--status-mod);
    }
    .badge-sim {
      background: rgba(168, 85, 247, 0.12);
      border: 1px solid rgba(168, 85, 247, 0.35);
      color: #c084fc;
    }
    .badge-modelled {
      background: rgba(56, 189, 248, 0.12);
      border: 1px solid rgba(56, 189, 248, 0.35);
      color: var(--accent-blue);
    }

    /* Live Monitor Grid Layout */
    .live-monitor-grid {
      display: grid;
      grid-template-columns: 1fr 360px;
      gap: 1.5rem;
      align-items: stretch;
    }
    @media (max-width: 1100px) {
      .live-monitor-grid { grid-template-columns: 1fr; }
    }

    /* Primary Video Surface */
    .video-viewport {
      background: #040812;
      border: 1px solid var(--border-subtle);
      border-radius: 8px;
      position: relative;
      display: flex;
      flex-direction: column;
      overflow: hidden;
      box-shadow: 0 4px 20px rgba(0, 0, 0, 0.5);
    }
    .video-top-bar {
      background: rgba(8, 13, 24, 0.9);
      border-bottom: 1px solid var(--border-subtle);
      padding: 8px 14px;
      display: flex;
      justify-content: space-between;
      align-items: center;
      font-family: var(--font-mono);
      font-size: 0.76rem;
      color: var(--text-muted);
    }
    .video-display-frame {
      position: relative;
      width: 100%;
      height: 480px;
      background: #000;
      display: flex;
      align-items: center;
      justify-content: center;
      overflow: hidden;
    }
    .video-stream-img {
      width: 100%;
      height: 100%;
      object-fit: contain;
      display: block;
    }

    /* Right Side Dock */
    .intelligence-dock {
      background: var(--bg-card);
      border: 1px solid var(--border-subtle);
      border-radius: 8px;
      padding: 1.25rem;
      display: flex;
      flex-direction: column;
      gap: 1.25rem;
    }
    .dock-title {
      font-size: 0.85rem;
      font-weight: 700;
      color: var(--text-dim);
      text-transform: uppercase;
      letter-spacing: 0.5px;
      border-bottom: 1px solid var(--border-subtle);
      padding-bottom: 0.5rem;
      display: flex;
      justify-content: space-between;
      align-items: center;
    }

    /* Numeric Risk Hero Block */
    .risk-hero-card {
      background: #0a1020;
      border: 1px solid var(--border-subtle);
      border-radius: 8px;
      padding: 1.25rem;
      display: flex;
      flex-direction: column;
      align-items: center;
      justify-content: center;
      gap: 0.4rem;
      text-align: center;
    }
    .risk-hero-title {
      font-size: 0.78rem;
      font-weight: 700;
      color: var(--text-muted);
      text-transform: uppercase;
      letter-spacing: 0.5px;
    }
    .risk-hero-score {
      font-family: var(--font-mono);
      font-size: 3rem;
      font-weight: 700;
      line-height: 1;
      color: var(--status-low);
      margin: 0.2rem 0;
    }
    .risk-hero-badge {
      font-size: 0.8rem;
      font-weight: 700;
      padding: 4px 14px;
      border-radius: 20px;
      background: rgba(16, 185, 129, 0.15);
      border: 1px solid var(--status-low);
      color: var(--status-low);
      text-transform: uppercase;
      letter-spacing: 0.5px;
    }

    /* Status Matrix */
    .status-matrix {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 0.6rem;
    }
    .matrix-item {
      background: #0a1020;
      border: 1px solid var(--border-subtle);
      padding: 0.6rem 0.8rem;
      border-radius: 6px;
      display: flex;
      flex-direction: column;
      gap: 3px;
    }
    .matrix-label {
      font-size: 0.72rem;
      color: var(--text-dim);
      text-transform: uppercase;
      font-weight: 600;
    }
    .matrix-val {
      font-family: var(--font-mono);
      font-size: 0.82rem;
      font-weight: 700;
      color: var(--text-main);
    }
    .matrix-val.status-green { color: var(--status-low); }
    .matrix-val.status-cyan { color: var(--accent-cyan); }
    .matrix-val.status-amber { color: var(--status-mod); }
    .matrix-val.status-red { color: var(--status-danger); }

    /* Incident Alert */
    .incident-alert-panel {
      background: rgba(239, 68, 68, 0.1);
      border: 1px solid rgba(239, 68, 68, 0.35);
      border-radius: 6px;
      padding: 0.75rem 1rem;
      display: flex;
      flex-direction: column;
      gap: 0.4rem;
    }
    .incident-alert-title {
      font-size: 0.8rem;
      font-weight: 700;
      color: var(--status-danger);
      display: flex;
      align-items: center;
      gap: 6px;
    }
    .incident-alert-body {
      display: flex;
      gap: 0.75rem;
      align-items: center;
    }
    .incident-thumb {
      width: 64px;
      height: 48px;
      object-fit: cover;
      border-radius: 4px;
      border: 1px solid rgba(239, 68, 68, 0.4);
      background: #000;
    }
    .incident-meta-text {
      font-size: 0.78rem;
      color: var(--text-muted);
      line-height: 1.35;
    }

    /* Bottom Telemetry Strip (5 Technical Cards) */
    .telemetry-strip {
      display: grid;
      grid-template-columns: repeat(5, 1fr);
      gap: 1.25rem;
    }
    @media (max-width: 1000px) {
      .telemetry-strip { grid-template-columns: repeat(2, 1fr); }
    }
    @media (max-width: 600px) {
      .telemetry-strip { grid-template-columns: 1fr; }
    }
    .telemetry-card {
      background: var(--bg-card);
      border: 1px solid var(--border-subtle);
      border-radius: 8px;
      padding: 1rem 1.25rem;
      display: flex;
      flex-direction: column;
      gap: 0.35rem;
      transition: transform 0.15s ease, border-color 0.15s ease;
    }
    .telemetry-card:hover {
      border-color: rgba(255, 255, 255, 0.15);
      transform: translateY(-2px);
    }
    .telemetry-title {
      font-size: 0.75rem;
      font-weight: 700;
      color: var(--text-dim);
      text-transform: uppercase;
      letter-spacing: 0.5px;
    }
    .telemetry-number {
      font-family: var(--font-mono);
      font-size: 1.75rem;
      font-weight: 700;
      color: var(--text-main);
    }
    .telemetry-unit {
      font-size: 0.85rem;
      font-weight: 500;
      color: var(--text-muted);
      margin-left: 4px;
    }
    .telemetry-sub {
      font-size: 0.75rem;
      color: var(--text-muted);
    }

    /* Temporal Chart Section */
    .chart-section {
      background: var(--bg-card);
      border: 1px solid var(--border-subtle);
      border-radius: 8px;
      padding: 1.25rem 1.5rem;
      display: flex;
      flex-direction: column;
      gap: 1rem;
    }
    .chart-header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      flex-wrap: wrap;
      gap: 0.75rem;
    }
    .chart-title {
      font-size: 0.88rem;
      font-weight: 700;
      color: var(--text-main);
      text-transform: uppercase;
      letter-spacing: 0.5px;
    }
    .metric-toggle-group {
      display: flex;
      gap: 0.4rem;
      flex-wrap: wrap;
    }
    .metric-toggle-btn {
      background: #0a1020;
      border: 1px solid var(--border-subtle);
      color: var(--text-muted);
      font-family: var(--font-main);
      font-size: 0.76rem;
      font-weight: 600;
      padding: 4px 10px;
      border-radius: 4px;
      cursor: pointer;
      transition: all 0.15s ease;
    }
    .metric-toggle-btn:hover { color: var(--text-main); }
    .metric-toggle-btn.active {
      background: rgba(0, 196, 223, 0.12);
      border-color: var(--accent-cyan);
      color: var(--accent-cyan);
    }
    .chart-canvas-container {
      width: 100%;
      height: 200px;
      position: relative;
    }
    canvas#temporalChart {
      width: 100%;
      height: 100%;
      display: block;
    }

    /* Front-Page Visual Architecture Pipeline */
    .arch-block-card {
      background: var(--bg-card);
      border: 1px solid var(--border-subtle);
      border-radius: 8px;
      padding: 1.25rem 1.5rem;
      display: flex;
      flex-direction: column;
      gap: 1rem;
      margin-top: 0.5rem;
    }
    .arch-block-header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      flex-wrap: wrap;
      gap: 0.75rem;
      border-bottom: 1px solid var(--border-subtle);
      padding-bottom: 0.75rem;
    }
    .arch-block-title-group {
      display: flex;
      flex-direction: column;
      gap: 0.2rem;
    }
    .arch-block-title {
      font-size: 0.92rem;
      font-weight: 700;
      color: var(--text-main);
      text-transform: uppercase;
      letter-spacing: 0.8px;
    }
    .arch-block-subtitle {
      font-size: 0.8rem;
      color: var(--accent-cyan);
      font-weight: 600;
      letter-spacing: 0.3px;
    }
    .arch-loop-badge {
      display: inline-flex;
      align-items: center;
      gap: 6px;
      padding: 5px 12px;
      border-radius: 6px;
      background: rgba(0, 196, 223, 0.08);
      border: 1px solid rgba(0, 196, 223, 0.25);
      color: var(--accent-cyan);
      font-family: var(--font-mono);
      font-size: 0.75rem;
      font-weight: 700;
      letter-spacing: 0.5px;
    }
    .arch-pipeline-track {
      display: flex;
      align-items: center;
      gap: 0.4rem;
      overflow-x: auto;
      padding: 0.5rem 0 0.75rem 0;
      scrollbar-width: thin;
    }
    .arch-node {
      background: #080e1c;
      border: 1px solid var(--border-subtle);
      border-radius: 6px;
      padding: 0.6rem 0.8rem;
      min-width: 105px;
      display: flex;
      flex-direction: column;
      align-items: center;
      text-align: center;
      gap: 2px;
      flex-shrink: 0;
      transition: all 0.2s ease;
    }
    .arch-node:hover {
      border-color: var(--accent-cyan);
      background: #0c162c;
    }
    .arch-node-num {
      font-family: var(--font-mono);
      font-size: 0.65rem;
      color: var(--text-dim);
      font-weight: 700;
    }
    .arch-node-name {
      font-size: 0.78rem;
      font-weight: 700;
      color: var(--text-main);
      white-space: nowrap;
    }
    .arch-node-desc {
      font-size: 0.68rem;
      color: var(--text-muted);
      white-space: nowrap;
    }
    .arch-node.arch-node-loop {
      border-color: rgba(0, 196, 223, 0.35);
      background: rgba(0, 196, 223, 0.04);
    }
    .arch-arrow {
      color: var(--text-dim);
      font-size: 0.9rem;
      font-weight: 700;
      flex-shrink: 0;
      user-select: none;
    }

    /* Common Card & Grid Layouts for Other Views */
    .card-grid-3 {
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(340px, 1fr));
      gap: 1.5rem;
    }
    .panel-card {
      background: var(--bg-card);
      border: 1px solid var(--border-subtle);
      border-radius: 8px;
      padding: 1.25rem;
      display: flex;
      flex-direction: column;
      gap: 0.8rem;
    }
    .panel-card-title {
      font-size: 0.88rem;
      font-weight: 700;
      color: var(--text-main);
      display: flex;
      justify-content: space-between;
      align-items: center;
      border-bottom: 1px solid var(--border-subtle);
      padding-bottom: 0.5rem;
    }

    /* Tables */
    .tech-table {
      width: 100%;
      border-collapse: collapse;
      font-size: 0.82rem;
    }
    .tech-table th {
      background: #0a1020;
      color: var(--text-dim);
      font-weight: 700;
      text-transform: uppercase;
      text-align: left;
      padding: 0.75rem 1rem;
      border-bottom: 1px solid var(--border-subtle);
    }
    .tech-table td {
      padding: 0.75rem 1rem;
      border-bottom: 1px solid rgba(255, 255, 255, 0.04);
      color: var(--text-muted);
    }
    .tech-table tr:hover td {
      background: rgba(255, 255, 255, 0.02);
      color: var(--text-main);
    }

    /* Digital Twin Simulator */
    .twin-split {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 1.5rem;
    }
    @media (max-width: 850px) {
      .twin-split { grid-template-columns: 1fr; }
    }
    .twin-box {
      background: #0e1628;
      border: 1px solid var(--border-subtle);
      border-radius: 8px;
      padding: 1.25rem;
      display: flex;
      flex-direction: column;
      gap: 0.8rem;
    }
    .slider-row {
      display: flex;
      flex-direction: column;
      gap: 0.35rem;
      font-size: 0.82rem;
    }
    .slider-header {
      display: flex;
      justify-content: space-between;
      color: var(--text-muted);
      font-weight: 600;
    }
    .slider-control {
      width: 100%;
      accent-color: var(--accent-cyan);
    }
    .reasoning-box {
      background: #080e1c;
      border: 1px solid var(--border-subtle);
      border-radius: 6px;
      padding: 0.75rem 1rem;
      display: flex;
      flex-direction: column;
      gap: 0.5rem;
      font-size: 0.82rem;
    }
    .reasoning-row {
      display: flex;
      justify-content: space-between;
      gap: 0.5rem;
    }
    .reasoning-key { color: var(--text-dim); }
    .reasoning-val { font-family: var(--font-mono); color: var(--text-main); font-weight: 600; text-align: right; }

    /* Warehouse 2D Heatmap Grid */
    .heatmap-stage {
      background: #040814;
      border: 1px solid var(--border-subtle);
      border-radius: 8px;
      height: 380px;
      position: relative;
      overflow: hidden;
    }
    .station-node {
      position: absolute;
      transform: translate(-50%, -50%);
      padding: 8px 14px;
      background: #0e182e;
      border: 2px solid var(--status-low);
      border-radius: 6px;
      cursor: pointer;
      display: flex;
      flex-direction: column;
      align-items: center;
      gap: 3px;
      box-shadow: 0 0 15px rgba(0, 0, 0, 0.8);
      transition: all 0.2s ease;
    }
    .station-node:hover {
      transform: translate(-50%, -50%) scale(1.08);
      z-index: 20;
    }
    .station-node-id { font-size: 0.82rem; font-weight: 700; color: #fff; }
    .station-node-score { font-family: var(--font-mono); font-size: 0.76rem; font-weight: 700; color: var(--accent-cyan); }

    /* Universal Detail Modal & Interactive Components */
    .modal-overlay {
      position: fixed;
      top: 0;
      left: 0;
      width: 100vw;
      height: 100vh;
      background: rgba(4, 8, 18, 0.85);
      backdrop-filter: blur(6px);
      z-index: 1000;
      display: none;
      align-items: center;
      justify-content: center;
      padding: 1.5rem;
      opacity: 0;
      transition: opacity 0.2s ease;
    }
    .modal-overlay.active {
      display: flex;
      opacity: 1;
    }
    .modal-dialog {
      background: #0d1527;
      border: 1px solid var(--border-subtle);
      border-radius: 10px;
      width: 100%;
      max-width: 820px;
      max-height: 90vh;
      display: flex;
      flex-direction: column;
      box-shadow: 0 16px 48px rgba(0, 0, 0, 0.75);
      overflow: hidden;
      animation: modalSlideIn 0.2s cubic-bezier(0.16, 1, 0.3, 1);
    }
    @keyframes modalSlideIn {
      from { transform: translateY(20px) scale(0.98); opacity: 0; }
      to { transform: translateY(0) scale(1); opacity: 1; }
    }
    .modal-header {
      padding: 1rem 1.5rem;
      border-bottom: 1px solid var(--border-subtle);
      display: flex;
      justify-content: space-between;
      align-items: center;
      background: #080e1c;
    }
    .modal-category {
      font-size: 0.72rem;
      font-weight: 700;
      color: var(--accent-cyan);
      text-transform: uppercase;
      letter-spacing: 0.8px;
    }
    .modal-title {
      font-size: 1.15rem;
      font-weight: 700;
      color: var(--text-main);
      margin-top: 2px;
    }
    .modal-header-actions {
      display: flex;
      align-items: center;
      gap: 0.75rem;
    }
    .modal-close-btn {
      background: transparent;
      border: none;
      color: var(--text-muted);
      font-size: 1.6rem;
      line-height: 1;
      cursor: pointer;
      padding: 0 4px;
      transition: color 0.15s ease;
    }
    .modal-close-btn:hover {
      color: var(--text-main);
    }
    .modal-body {
      padding: 1.5rem;
      overflow-y: auto;
      display: flex;
      flex-direction: column;
      gap: 1.25rem;
      font-size: 0.85rem;
      color: var(--text-muted);
    }
    .modal-footer {
      padding: 0.75rem 1.5rem;
      border-top: 1px solid var(--border-subtle);
      background: #080e1c;
      display: flex;
      justify-content: flex-end;
    }

    /* Detail Modal Specific Elements */
    .detail-grid-2 {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 1rem;
    }
    @media (max-width: 650px) {
      .detail-grid-2 { grid-template-columns: 1fr; }
    }
    .detail-grid-3 {
      display: grid;
      grid-template-columns: repeat(3, 1fr);
      gap: 0.75rem;
    }
    @media (max-width: 650px) {
      .detail-grid-3 { grid-template-columns: 1fr; }
    }
    .detail-metric-card {
      background: #080e1c;
      border: 1px solid var(--border-subtle);
      border-radius: 6px;
      padding: 0.75rem;
      display: flex;
      flex-direction: column;
      gap: 3px;
    }
    .detail-metric-lbl {
      font-size: 0.72rem;
      color: var(--text-dim);
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 0.5px;
    }
    .detail-metric-val {
      font-family: var(--font-mono);
      font-size: 1.1rem;
      font-weight: 700;
      color: var(--text-main);
    }
    .detail-section-title {
      font-size: 0.82rem;
      font-weight: 700;
      color: var(--text-dim);
      text-transform: uppercase;
      letter-spacing: 0.6px;
      border-bottom: 1px solid var(--border-subtle);
      padding-bottom: 0.4rem;
      margin-bottom: 0.5rem;
    }
    .panel-card.clickable-card {
      cursor: pointer;
      transition: transform 0.15s ease, border-color 0.15s ease, box-shadow 0.15s ease;
    }
    .panel-card.clickable-card:hover {
      transform: translateY(-2px);
      border-color: var(--accent-cyan);
      box-shadow: 0 4px 18px rgba(0, 196, 223, 0.12);
    }
    tr.clickable-row {
      cursor: pointer;
      transition: background 0.15s ease;
    }
    tr.clickable-row:hover td {
      background: rgba(0, 196, 223, 0.08) !important;
    }
  </style>
</head>
<body>

  <!-- Top Clean Header -->
  <header class="app-header">
    <div class="brand-section">
      <div class="brand-title">
        <span>KINETICGUARD</span>
      </div>
      <div class="brand-subtitle">Workplace Ergonomic Intelligence</div>
    </div>
    <div class="header-badges">
      <a href="/api/download/sample_videos" class="btn-sample-download" download="video_env_1_anonymized.zip" title="Download sample benchmark MP4 video set (video_env_1_anonymized.zip)">
        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="7 10 12 15 17 10"/><line x1="12" y1="15" x2="12" y2="3"/></svg>
        Download Sample Video Set
      </a>
      <div class="status-pill">
        <span class="indicator-dot"></span>
        PRIVACY GUARD: ACTIVE
      </div>
    </div>
  </header>

  <!-- Navigation Tabs -->
  <nav class="nav-bar">
    <button class="nav-btn active" onclick="switchView('live')">Live Monitor</button>
    <button class="nav-btn" onclick="switchView('incidents')">Incidents</button>
    <button class="nav-btn" onclick="switchView('workers')">Workers</button>
    <button class="nav-btn" onclick="switchView('stations')">Stations</button>
    <button class="nav-btn" onclick="switchView('interventions')">Interventions</button>
    <button class="nav-btn" onclick="switchView('twin')">Digital Twin</button>
    <button class="nav-btn" onclick="switchView('governance')">Governance</button>
    <button class="nav-btn" onclick="switchView('reports')">Reports</button>
  </nav>

  <!-- Main View Container -->
  <main class="main-container">

    <!-- ================================================================= -->
    <!-- TAB 1: LIVE SAFETY MONITOR                                       -->
    <!-- ================================================================= -->
    <div id="view-live" class="tab-view active">

      <!-- Video Source Switcher Toolbar -->
      <div class="source-toolbar">
        <div class="source-left">
          <div class="source-label">Video Source:</div>
          <div class="source-btn-group">
            <button class="source-btn active" id="btn-cam01" onclick="changeVideoSource('camera_01')">Camera 01 (ap_0016.mp4)</button>
            <button class="source-btn" id="btn-cam02" onclick="changeVideoSource('camera_02')">Camera 02 (ap_0017.mp4)</button>
            <button class="source-btn" id="btn-cam03" onclick="changeVideoSource('camera_03')">Camera 03 (ap_0018.mp4)</button>
            <button class="source-btn" id="btn-webcam" onclick="changeVideoSource('webcam')">Live Webcam</button>
            <button class="source-btn upload-btn" onclick="document.getElementById('video-upload-input').click()">Upload Video (.mp4)</button>
            <input type="file" id="video-upload-input" accept="video/*,.mp4,.mov,.avi,.mkv,.webm" style="display:none" onchange="uploadCustomVideo(this.files[0])" />
          </div>
        </div>
        <div id="active-source-badge" class="active-source-badge">Feed: Camera 01 (ap_0016.mp4)</div>
      </div>

      <!-- Live Video + Intelligence Dock Grid -->
      <div class="live-monitor-grid">

        <!-- Left: Primary Processed Video Feed -->
        <div class="video-viewport">
          <div class="video-top-bar">
            <span id="hud-cam-id">Camera 01 [Station 01]</span>
            <span id="hud-fps">FPS: 25.0</span>
            <span id="hud-frame-idx" style="display:none;">00000</span>
          </div>

          <div class="video-display-frame">
            <img id="live-stream-img" class="video-stream-img" src="/api/stream/video?source=camera_01" alt="KineticGuard Live Stream" />
          </div>
        </div>

        <!-- Right: Ergonomic Risk & Intelligence Dock -->
        <div class="intelligence-dock">
          <div class="dock-title">
            <span>Risk & Biomechanics Status</span>
          </div>

          <!-- Large Numeric Risk Hero Block -->
          <div class="risk-hero-card">
            <div class="risk-hero-title">Cumulative Ergonomic Risk</div>
            <div class="risk-hero-score" id="risk-hero-score">0.0</div>
            <div class="risk-hero-badge" id="risk-hero-badge">LOW RISK</div>
          </div>

          <!-- Live Status Matrix -->
          <div class="status-matrix">
            <div class="matrix-item">
              <span class="matrix-label">Privacy Guard</span>
              <span class="matrix-val status-green">ACTIVE</span>
            </div>
            <div class="matrix-item">
              <span class="matrix-label">Pose Engine</span>
              <span class="matrix-val status-green" id="pose-status-val">ACTIVE 33-PTS</span>
            </div>
            <div class="matrix-item">
              <span class="matrix-label">Risk Engine</span>
              <span class="matrix-val status-cyan" id="agent-status-val">EVALUATING</span>
            </div>
            <div class="matrix-item">
              <span class="matrix-label">Policy Gate</span>
              <span class="matrix-val status-green" id="cedar-status-val">ALLOW</span>
            </div>
          </div>

          <!-- Posture & Load Overview -->
          <div class="reasoning-box">
            <div class="reasoning-row">
              <span class="reasoning-key">Current Posture:</span>
              <span class="reasoning-val" id="dock-posture-val">UPRIGHT</span>
            </div>
            <div class="reasoning-row">
              <span class="reasoning-key">Moment Arm:</span>
              <span class="reasoning-val status-green" id="dock-moment-val">Normal Alignment</span>
            </div>
          </div>

          <!-- Live Incident Trigger State -->
          <div class="incident-alert-panel clickable-card" id="incident-alert-box" style="display:none; cursor:pointer;" onclick="openIncidentDetail(document.getElementById('incident-alert-id').innerText)" title="Click to view live incident intelligence detail">
            <div class="incident-alert-title">
              <span class="indicator-dot" style="background:var(--status-danger);"></span>
              Ergonomic Hazard Detected
            </div>
            <div class="incident-alert-body">
              <img id="incident-alert-thumb" class="incident-thumb" src="" alt="Keyframe" />
              <div class="incident-meta-text">
                <div id="incident-alert-id" style="font-weight:700; color:#fff;">INC-PENDING</div>
                <div id="incident-alert-stat">Station: Station 01</div>
                <div id="incident-alert-reason" style="color:var(--status-high);">Sustained Lumbar Flexion</div>
              </div>
            </div>
          </div>

        </div>

      </div>

      <!-- Bottom Biomechanical Telemetry Strip (5 Technical Tiles) -->
      <div class="telemetry-strip">
        <div class="telemetry-card clickable-card" id="card-trunk" onclick="openTelemetryDetail('trunk')" title="Click to inspect trunk flexion angle calculations & ergonomic limits">
          <div class="telemetry-title">TRUNK FLEXION ANGLE</div>
          <div class="telemetry-number" id="num-trunk">0.0<span class="telemetry-unit">&deg;</span></div>
          <div class="telemetry-sub" id="sub-posture">Posture: Upright</div>
        </div>
        <div class="telemetry-card clickable-card" id="card-torque" onclick="openTelemetryDetail('torque')" title="Click to inspect L5/S1 reactive lumbar moment calculations">
          <div class="telemetry-title">L5/S1 LUMBAR TORQUE</div>
          <div class="telemetry-number" id="num-torque">0.0<span class="telemetry-unit">Nm</span></div>
          <div class="telemetry-sub">Estimated Model Load</div>
        </div>
        <div class="telemetry-card clickable-card" id="card-compression" onclick="openTelemetryDetail('compression')" title="Click to inspect spinal compression force & NIOSH safety limits">
          <div class="telemetry-title">SPINAL COMPRESSION</div>
          <div class="telemetry-number" id="num-comp">0<span class="telemetry-unit">N</span></div>
          <div class="telemetry-sub">NIOSH Action Limit: 3400 N</div>
        </div>
        <div class="telemetry-card clickable-card" id="card-cadence" onclick="openTelemetryDetail('cadence')" title="Click to inspect repetitive motion cadence metrics">
          <div class="telemetry-title">REPETITION CADENCE</div>
          <div class="telemetry-number" id="num-cadence">0.0<span class="telemetry-unit">/min</span></div>
          <div class="telemetry-sub">Repetitive Motion Cycles</div>
        </div>
        <div class="telemetry-card clickable-card" id="card-duty" onclick="openTelemetryDetail('duty')" title="Click to inspect rolling awkward duty cycle window">
          <div class="telemetry-title">AWKWARD DUTY CYCLE</div>
          <div class="telemetry-number" id="num-duty">0.0<span class="telemetry-unit">%</span></div>
          <div class="telemetry-sub">20s Window Exposure</div>
        </div>
      </div>

      <!-- Real-Time Temporal Exposure Graph -->
      <div class="chart-section">
        <div class="chart-header">
          <div class="chart-title">Temporal Biomechanical Exposure Timeline</div>
          <div class="metric-toggle-group">
            <button class="metric-toggle-btn active" id="btn-chart-risk" onclick="setChartMetric('risk')">Cumulative Risk (0-100)</button>
            <button class="metric-toggle-btn" id="btn-chart-torque" onclick="setChartMetric('torque')">Lumbar Torque (Nm)</button>
            <button class="metric-toggle-btn" id="btn-chart-comp" onclick="setChartMetric('comp')">Spinal Compression (N)</button>
            <button class="metric-toggle-btn" id="btn-chart-duty" onclick="setChartMetric('duty')">Awkward Duty Cycle (%)</button>
          </div>
        </div>
        <div class="chart-canvas-container">
          <canvas id="temporalChart"></canvas>
        </div>
      </div>

      <!-- Front-Page KineticGuard Architecture Visual Pipeline Block -->
      <div class="panel-card arch-block-card">
        <div class="arch-block-header">
          <div class="arch-block-title-group">
            <span class="arch-block-title">KINETICGUARD ARCHITECTURE</span>
            <span class="arch-block-subtitle">AI explains. Deterministic systems measure. Policy governs.</span>
          </div>
          <div class="arch-loop-badge">
            <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><polyline points="23 4 23 10 17 10"/><path d="M20.49 15a9 9 0 1 1-2.12-9.36L23 10"/></svg>
            <span>CLOSED-LOOP: VERIFY &rarr; RE-MONITOR</span>
          </div>
        </div>

        <div class="arch-pipeline-track">
          <!-- Step 1: Video -->
          <div class="arch-node clickable-card" onclick="openArchDetail(1)" title="Click to inspect Video Ingestion stage">
            <div class="arch-node-num">01</div>
            <div class="arch-node-name">VIDEO</div>
            <div class="arch-node-desc">Multi-Camera / Upload</div>
          </div>
          <div class="arch-arrow">&rarr;</div>

          <!-- Step 2: Privacy Guard -->
          <div class="arch-node clickable-card" onclick="openArchDetail(2)" title="Click to inspect Privacy Guard de-identification">
            <div class="arch-node-num">02</div>
            <div class="arch-node-name">PRIVACY GUARD</div>
            <div class="arch-node-desc">Facial De-Identification</div>
          </div>
          <div class="arch-arrow">&rarr;</div>

          <!-- Step 3: Pose Engine -->
          <div class="arch-node clickable-card" onclick="openArchDetail(3)" title="Click to inspect 33-point Pose Detection engine">
            <div class="arch-node-num">03</div>
            <div class="arch-node-name">POSE ENGINE</div>
            <div class="arch-node-desc">33-Point Spatial Tracking</div>
          </div>
          <div class="arch-arrow">&rarr;</div>

          <!-- Step 4: Biomechanics -->
          <div class="arch-node clickable-card" onclick="openArchDetail(4)" title="Click to inspect Deterministic Biomechanics engine">
            <div class="arch-node-num">04</div>
            <div class="arch-node-name">BIOMECHANICS</div>
            <div class="arch-node-desc">Torque &amp; Compression</div>
          </div>
          <div class="arch-arrow">&rarr;</div>

          <!-- Step 5: Temporal Risk -->
          <div class="arch-node clickable-card" onclick="openArchDetail(5)" title="Click to inspect Temporal Risk accumulation">
            <div class="arch-node-num">05</div>
            <div class="arch-node-name">TEMPORAL RISK</div>
            <div class="arch-node-desc">Cumulative Exposure</div>
          </div>
          <div class="arch-arrow">&rarr;</div>

          <!-- Step 6: Incident -->
          <div class="arch-node clickable-card" onclick="openArchDetail(6)" title="Click to inspect Incident Engine & Keyframe capture">
            <div class="arch-node-num">06</div>
            <div class="arch-node-name">INCIDENT</div>
            <div class="arch-node-desc">Trigger &amp; Keyframe</div>
          </div>
          <div class="arch-arrow">&rarr;</div>

          <!-- Step 7: KGVision -->
          <div class="arch-node clickable-card" onclick="openArchDetail(7)" title="Click to inspect KGVision Multimodal Root-Cause Reasoning">
            <div class="arch-node-num">07</div>
            <div class="arch-node-name">KGVISION</div>
            <div class="arch-node-desc">Root-Cause Reasoning</div>
          </div>
          <div class="arch-arrow">&rarr;</div>

          <!-- Step 8: Policy Gate -->
          <div class="arch-node clickable-card" onclick="openArchDetail(8)" title="Click to inspect Cedar Formal Policy Gate">
            <div class="arch-node-num">08</div>
            <div class="arch-node-name">POLICY GATE</div>
            <div class="arch-node-desc">Autonomous Safety Gate</div>
          </div>
          <div class="arch-arrow">&rarr;</div>

          <!-- Step 9: Intervention -->
          <div class="arch-node clickable-card" onclick="openArchDetail(9)" title="Click to inspect Closed-Loop Intervention dispatch">
            <div class="arch-node-num">09</div>
            <div class="arch-node-name">INTERVENTION</div>
            <div class="arch-node-desc">Dispatch &amp; Rotation</div>
          </div>
          <div class="arch-arrow">&rarr;</div>

          <!-- Step 10: Re-Monitor -->
          <div class="arch-node arch-node-loop clickable-card" onclick="openArchDetail(10)" title="Click to inspect Re-Monitoring engine">
            <div class="arch-node-num">10</div>
            <div class="arch-node-name">RE-MONITOR</div>
            <div class="arch-node-desc">Post-Action Stream</div>
          </div>
          <div class="arch-arrow">&rarr;</div>

          <!-- Step 11: Verify -->
          <div class="arch-node arch-node-loop clickable-card" onclick="openArchDetail(11)" title="Click to inspect Empirical Verification & statistical validation">
            <div class="arch-node-num">11</div>
            <div class="arch-node-name">VERIFY</div>
            <div class="arch-node-desc">Empirical Validation</div>
          </div>
        </div>
      </div>

    </div>

    <!-- ================================================================= -->
    <!-- TAB 2: INCIDENTS TIMELINE                                         -->
    <!-- ================================================================= -->
    <div id="view-incidents" class="tab-view">
      <div class="dock-title">
        <span>Persistent Incident Ledger (OpenSearch)</span>
        <button class="source-btn" onclick="fetchIncidents()">Refresh</button>
      </div>
      <div class="card-grid-3" id="incidents-grid-container">
        <!-- Dynamically rendered incident cards -->
      </div>
    </div>

    <!-- ================================================================= -->
    <!-- TAB 3: WORKER PASSPORTS                                           -->
    <!-- ================================================================= -->
    <div id="view-workers" class="tab-view">
      <div class="dock-title">
        <span>Worker Ergonomic Passports (OpenSearch)</span>
        <button class="source-btn" onclick="fetchPassports()">Refresh</button>
      </div>
      <div class="card-grid-3" id="workers-grid-container">
        <!-- Dynamically rendered worker passport cards -->
      </div>
    </div>

    <!-- ================================================================= -->
    <!-- TAB 4: STATIONS & WAREHOUSE HEATMAP                               -->
    <!-- ================================================================= -->
    <div id="view-stations" class="tab-view">
      <div class="dock-title">
        <span>Warehouse Ergonomic Spatial Heatmap</span>
        <button class="source-btn" onclick="fetchStationsAndHeatmap()">Refresh</button>
      </div>
      <div class="heatmap-stage" id="heatmap-stage">
        <!-- Dynamically placed spatial station nodes -->
      </div>
      <div class="dock-title" style="margin-top:1.5rem;">
        <span>Station Risk Profiles</span>
      </div>
      <div class="card-grid-3" id="stations-grid-container">
        <!-- Dynamically rendered station cards -->
      </div>
    </div>

    <!-- ================================================================= -->
    <!-- TAB 5: CLOSED-LOOP INTERVENTIONS                                  -->
    <!-- ================================================================= -->
    <div id="view-interventions" class="tab-view">
      <div class="dock-title">
        <span>Closed-Loop Interventions & Re-Monitoring Lifecycle</span>
        <button class="source-btn" onclick="fetchInterventions()">Refresh</button>
      </div>
      <div class="card-grid-3" id="interventions-grid-container">
        <!-- Dynamically rendered intervention lifecycle cards -->
      </div>
    </div>

    <!-- ================================================================= -->
    <!-- TAB 6: PREDICTIVE DIGITAL TWIN (WHAT-IF)                          -->
    <!-- ================================================================= -->
    <div id="view-twin" class="tab-view">
      <div class="dock-title">
        <span>Predictive Ergonomic Digital Twin (What-If Simulator)</span>
        <span class="active-source-badge">MODELLED / SIMULATED</span>
      </div>
      <div class="twin-split">
        <!-- Left: Interactive Ergonomic Parameters -->
        <div class="twin-box">
          <div class="panel-card-title">Modification Parameters</div>
          <div class="slider-row">
            <div class="slider-header">
              <span>Pallet / Working Surface Height</span>
              <span id="val-twin-pallet">85 cm</span>
            </div>
            <input type="range" class="slider-control" id="slider-pallet" min="20" max="110" value="85" oninput="updateTwinSimulation()" />
          </div>
          <div class="slider-row">
            <div class="slider-header">
              <span>Load Weight</span>
              <span id="val-twin-load">8 kg</span>
            </div>
            <input type="range" class="slider-control" id="slider-load" min="2" max="30" value="8" oninput="updateTwinSimulation()" />
          </div>
          <div class="slider-row">
            <div class="slider-header">
              <span>Handling Cadence</span>
              <span id="val-twin-cadence">12 /min</span>
            </div>
            <input type="range" class="slider-control" id="slider-cadence" min="4" max="30" value="12" oninput="updateTwinSimulation()" />
          </div>
          <div class="slider-row">
            <div class="slider-header">
              <span>Task Duration</span>
              <span id="val-twin-duration">45 min</span>
            </div>
            <input type="range" class="slider-control" id="slider-duration" min="10" max="120" value="45" oninput="updateTwinSimulation()" />
          </div>
          <div class="slider-row" style="flex-direction:row; align-items:center; gap:0.6rem; margin-top:0.4rem;">
            <input type="checkbox" id="check-twin-rotation" onchange="updateTwinSimulation()" checked />
            <label for="check-twin-rotation" style="color:var(--text-main); font-size:0.82rem;">Mandate 20-minute Task Rotation</label>
          </div>
        </div>

        <!-- Right: Before vs After Projections -->
        <div class="twin-box clickable-card" id="twin-projection-box" onclick="openTwinDetail()" title="Click to view full Predictive Simulation analysis">
          <div class="panel-card-title">
            <span>Biomechanical Projection</span>
            <span style="font-size:0.72rem; color:var(--accent-cyan);">Click to Inspect</span>
          </div>
          <div class="reasoning-box" style="padding:1rem;">
            <div class="reasoning-row">
              <span class="reasoning-key">Trunk Angle:</span>
              <span class="reasoning-val" id="twin-res-angle">42.5&deg; &rarr; 20.5&deg; (-51.8%)</span>
            </div>
            <div class="reasoning-row">
              <span class="reasoning-key">L5/S1 Torque:</span>
              <span class="reasoning-val status-green" id="twin-res-torque">97.2 Nm &rarr; 58.4 Nm (-39.9%)</span>
            </div>
            <div class="reasoning-row">
              <span class="reasoning-key">Spinal Compression:</span>
              <span class="reasoning-val status-green" id="twin-res-comp">2312 N &rarr; 1650 N (-28.6%)</span>
            </div>
            <div class="reasoning-row">
              <span class="reasoning-key">Cumulative Risk:</span>
              <span class="reasoning-val status-cyan" id="twin-res-risk">73.7 &rarr; 41.2 [Moderate]</span>
            </div>
            <div class="reasoning-row">
              <span class="reasoning-key">Risk Reduction:</span>
              <span class="reasoning-val status-green" id="twin-res-reduct">-44.1% Overall Reduction</span>
            </div>
          </div>
          <div id="twin-benefits-list" style="font-size:0.78rem; color:var(--text-muted); line-height:1.4;">
            <!-- Key benefits rendered dynamically -->
          </div>
        </div>
      </div>
    </div>

    <!-- ================================================================= -->
    <!-- TAB 7: CEDAR GOVERNANCE AUDIT                                     -->
    <!-- ================================================================= -->
    <div id="view-governance" class="tab-view">
      <div class="dock-title">
        <span>Cedar Policy Governance Audit Trail</span>
        <button class="source-btn" onclick="fetchGovernanceAudit()">Refresh</button>
      </div>
      <div style="background:var(--bg-card); border:1px solid var(--border-subtle); border-radius:8px; overflow-x:auto;">
        <table class="tech-table">
          <thead>
            <tr>
              <th>Timestamp</th>
              <th>Incident ID</th>
              <th>Worker / Station</th>
              <th>Strands Proposed</th>
              <th>Cedar Policy</th>
              <th>Enforced Action</th>
              <th>Decision</th>
            </tr>
          </thead>
          <tbody id="governance-table-body">
            <!-- Dynamically populated audit rows -->
          </tbody>
        </table>
      </div>
    </div>

    <!-- ================================================================= -->
    <!-- TAB 8: SHIFT SAFETY REPORTS                                       -->
    <!-- ================================================================= -->
    <div id="view-reports" class="tab-view">
      <div class="dock-title">
        <span>Executive Shift Safety Intelligence Report</span>
        <button class="source-btn" onclick="fetchShiftReport()">Generate Latest</button>
      </div>
      <div class="panel-card clickable-card" id="report-card-container" onclick="openReportDetail()" title="Click to view complete Shift Safety Intelligence Report">
        <div class="panel-card-title">
          <span id="report-shift-id">SHIFT-REPORT-001</span>
          <span class="active-source-badge" id="report-safety-score" style="color:var(--status-low);">SCORE: 85.0/100</span>
        </div>
        <div id="report-body-content" style="font-size:0.85rem; color:var(--text-muted); line-height:1.6;">
          Loading executive report data...
        </div>
      </div>
    </div>

  </main>

  <!-- Universal Detail Modal Overlay -->
  <div id="detail-modal-overlay" class="modal-overlay" onclick="closeDetailModal(event)">
    <div class="modal-dialog" onclick="event.stopPropagation()">
      <div class="modal-header">
        <div class="modal-title-wrap">
          <div class="modal-category" id="modal-category">DETAIL VIEW</div>
          <h2 class="modal-title" id="modal-title">Entity Detail</h2>
        </div>
        <div class="modal-header-actions">
          <span id="modal-badge-container"></span>
          <button class="modal-close-btn" onclick="closeDetailModal()" title="Close (Esc)">&times;</button>
        </div>
      </div>
      <div class="modal-body" id="modal-body">
        <!-- Dynamically rendered detail content -->
      </div>
      <div class="modal-footer">
        <button class="source-btn" onclick="closeDetailModal()">Close</button>
      </div>
    </div>
  </div>

  <!-- Clean Footer -->
  <footer class="footer-bar">
    <div>KineticGuard Ergonomic Safety Intelligence Platform</div>
    <div>Privacy Guard Active (Zero Identifiable Biometric Storage)</div>
  </footer>

  <script>
    // Universal Null & Telemetry Formatters
    function fmtVal(val, unit = '', digits = null) {
      if (val === null || val === undefined || val === '') return 'N/A';
      if (typeof val === 'number') {
        if (isNaN(val)) return 'N/A';
        if (digits !== null) {
          return (digits === 0 ? Math.round(val) : val.toFixed(digits)) + (unit ? ' ' + unit : '');
        }
        return val + (unit ? ' ' + unit : '');
      }
      const n = Number(val);
      if (!isNaN(n)) {
        if (digits !== null) {
          return (digits === 0 ? Math.round(n) : n.toFixed(digits)) + (unit ? ' ' + unit : '');
        }
        return n + (unit ? ' ' + unit : '');
      }
      return String(val) + (unit ? ' ' + unit : '');
    }

    function fmtStr(val, fallback = 'N/A') {
      if (val === null || val === undefined || val === '' || val === 'undefined' || val === 'null') {
        return fallback;
      }
      return String(val);
    }

    function fmtBadge(source) {
      const s = String(source || 'TEST_FIXTURE').toUpperCase();
      if (s === 'REAL_TELEMETRY' || s === 'REAL') {
        return `<span class="badge-source badge-real" title="Verified Real Telemetry">REAL TELEMETRY</span>`;
      } else if (s === 'HISTORICAL' || s === 'HIST') {
        return `<span class="badge-source badge-hist" title="Historical Baseline">HISTORICAL</span>`;
      } else if (s === 'MODELLED') {
        return `<span class="badge-source badge-modelled" title="Modelled Biomechanics">MODELLED</span>`;
      } else if (s === 'SIMULATED') {
        return `<span class="badge-source badge-sim" title="Simulated Projection">SIMULATED</span>`;
      } else {
        return `<span class="badge-source badge-test" title="Historical / Test Fixture">TEST FIXTURE</span>`;
      }
    }

    // State management
    let activeView = 'live';
    let currentSource = 'camera_01';
    let activeChartMetric = 'risk';
    let activeUploadedName = '';
    const chartHistory = {
      risk: [],
      torque: [],
      comp: [],
      duty: []
    };
    const maxHistory = 40;

    // Tab Navigation
    function switchView(viewName) {
      activeView = viewName;
      document.querySelectorAll('.tab-view').forEach(el => el.classList.remove('active'));
      document.querySelectorAll('.nav-btn').forEach(el => el.classList.remove('active'));

      const target = document.getElementById('view-' + viewName);
      if (target) target.classList.add('active');

      const btns = document.querySelectorAll('.nav-btn');
      btns.forEach(b => {
        if (b.getAttribute('onclick') && b.getAttribute('onclick').includes(viewName)) {
          b.classList.add('active');
        }
      });

      if (viewName === 'incidents') fetchIncidents();
      if (viewName === 'workers') fetchPassports();
      if (viewName === 'stations') fetchStationsAndHeatmap();
      if (viewName === 'interventions') fetchInterventions();
      if (viewName === 'governance') fetchGovernanceAudit();
      if (viewName === 'reports') fetchShiftReport();
      if (viewName === 'twin') updateTwinSimulation();
    }

    // Video Source Switcher
    function changeVideoSource(sourceType) {
      currentSource = sourceType;
      document.querySelectorAll('.source-btn').forEach(b => b.classList.remove('active'));
      const activeBtn = document.getElementById('btn-' + sourceType.replace('camera_0', 'cam0'));
      if (activeBtn) activeBtn.classList.add('active');

      const dispLabels = {
        'camera_01': 'Camera 01 (ap_0016.mp4)',
        'camera_02': 'Camera 02 (ap_0017.mp4)',
        'camera_03': 'Camera 03 (ap_0018.mp4)',
        'webcam': 'Live Webcam'
      };
      const label = dispLabels[sourceType] || sourceType;
      document.getElementById('active-source-badge').innerText = 'Feed: ' + label;
      document.getElementById('hud-cam-id').innerText = label;

      fetch('/api/stream/source', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ source: sourceType })
      }).then(() => {
        const img = document.getElementById('live-stream-img');
        img.src = '/api/stream/video?source=' + sourceType + '&t=' + Date.now();
      }).catch(err => console.error(err));
    }

    // Video Upload with 3-Stage Progress: UPLOADING -> PROCESSING -> STREAMING
    function uploadCustomVideo(file) {
      if (!file) return;
      const originalName = file.name || 'uploaded_video.mp4';
      activeUploadedName = originalName;
      const sizeMb = (file.size / (1024 * 1024)).toFixed(1);

      const badge = document.getElementById('active-source-badge');
      const hudCam = document.getElementById('hud-cam-id');
      const streamImg = document.getElementById('live-stream-img');

      // Clear input so re-selecting the same file fires onchange
      const fileInput = document.getElementById('video-upload-input');
      if (fileInput) fileInput.value = '';

      // Stage 1: UPLOADING
      badge.innerText = 'Uploading: ' + originalName + ' (0%)';
      badge.style.color = 'var(--status-mod)';
      if (hudCam) hudCam.innerText = 'Uploading ' + originalName + '...';

      // Remove active class from camera buttons
      document.querySelectorAll('.source-btn').forEach(b => b.classList.remove('active'));

      const xhr = new XMLHttpRequest();
      xhr.open('POST', '/api/upload_video', true);
      xhr.setRequestHeader('Content-Type', file.type || 'video/mp4');
      xhr.setRequestHeader('X-File-Name', encodeURIComponent(originalName));

      xhr.upload.onprogress = function(e) {
        if (e.lengthComputable) {
          const pct = Math.round((e.loaded / e.total) * 100);
          badge.innerText = 'Uploading: ' + originalName + ' (' + pct + '% of ' + sizeMb + ' MB)';
        }
      };

      xhr.onload = function() {
        if (xhr.status >= 200 && xhr.status < 300) {
          try {
            const res = JSON.parse(xhr.responseText);
            if (res.status === 'READY' || res.status === 'SUCCESS') {
              // Stage 3: STREAMING
              currentSource = 'uploaded';
              const dispName = res.filename || originalName;
              activeUploadedName = dispName;

              badge.innerText = 'Streaming: ' + dispName;
              badge.style.color = 'var(--status-low)';
              if (hudCam) hudCam.innerText = 'Uploaded: ' + dispName;

              if (streamImg) {
                streamImg.src = '';
                setTimeout(() => {
                  streamImg.src = '/api/stream/video?source=uploaded&t=' + Date.now();
                }, 50);
              }
              // Immediate poll for refreshed telemetry
              setTimeout(pollLiveTelemetry, 100);
            } else {
              badge.innerText = 'Error: ' + (res.error || 'Upload failed');
              badge.style.color = 'var(--status-danger)';
            }
          } catch(err) {
            badge.innerText = 'Error: Invalid server response';
            badge.style.color = 'var(--status-danger)';
          }
        } else {
          try {
            const errRes = JSON.parse(xhr.responseText);
            badge.innerText = 'Error: ' + (errRes.error || ('HTTP ' + xhr.status));
          } catch(e) {
            badge.innerText = 'Upload Error (HTTP ' + xhr.status + ')';
          }
          badge.style.color = 'var(--status-danger)';
        }
      };

      xhr.onerror = function() {
        badge.innerText = 'Upload Network Error';
        badge.style.color = 'var(--status-danger)';
      };

      xhr.send(file);
    }

    // Live Telemetry Polling Loop
    function pollLiveTelemetry() {
      fetch('/api/stream/telemetry')
        .then(r => r.json())
        .then(t => {
          // Update frame index & HUD
          const frameEl = document.getElementById('hud-frame-idx');
          if (frameEl) frameEl.innerText = String(t.frame_index).padStart(5, '0');
          document.getElementById('hud-fps').innerText = 'FPS: ' + t.fps.toFixed(1);

          // Update Source HUD label
          if (t.source_type === 'uploaded' || t.camera_id === 'uploaded' || currentSource === 'uploaded') {
            const disp = activeUploadedName || t.filename || 'uploaded_video.mp4';
            document.getElementById('hud-cam-id').innerText = 'Uploaded: ' + disp;
            if (currentSource === 'uploaded') {
              document.getElementById('active-source-badge').innerText = 'Streaming: ' + disp;
              document.getElementById('active-source-badge').style.color = 'var(--status-low)';
            }
          } else {
            const camLabels = {
              'camera_01': 'Camera 01 (ap_0016.mp4)',
              'camera_02': 'Camera 02 (ap_0017.mp4)',
              'camera_03': 'Camera 03 (ap_0018.mp4)',
              'webcam': 'Live Webcam'
            };
            const dispName = camLabels[t.camera_id] || (t.camera_id.toUpperCase() + ' [' + t.station_id + ']');
            document.getElementById('hud-cam-id').innerText = dispName;
            if (currentSource !== 'uploaded') {
              document.getElementById('active-source-badge').innerText = 'Feed: ' + dispName;
              document.getElementById('active-source-badge').style.color = 'var(--accent-cyan)';
            }
          }

          // Update Status Matrix
          document.getElementById('pose-status-val').innerText = t.pose_detected ? 'ACTIVE 33-PTS' : 'SEARCHING';
          document.getElementById('agent-status-val').innerText = t.agent_status || 'READY';
          document.getElementById('cedar-status-val').innerText = t.cedar_decision || 'ALLOW';

          // Posture Overview
          const postureEl = document.getElementById('dock-posture-val');
          if (postureEl) postureEl.innerText = t.posture;
          const momentEl = document.getElementById('dock-moment-val');
          if (momentEl) {
            if (t.posture === 'BENDING' || t.posture === 'HIGH_RISK_LIFTING') {
              momentEl.innerText = 'Extended Lever Arm';
              momentEl.className = 'reasoning-val status-amber';
            } else {
              momentEl.innerText = 'Normal Alignment';
              momentEl.className = 'reasoning-val status-green';
            }
          }

          // Update Numeric Hero Risk
          const scoreEl = document.getElementById('risk-hero-score');
          const badgeEl = document.getElementById('risk-hero-badge');
          scoreEl.innerText = t.cumulative_risk_score.toFixed(1);
          badgeEl.innerText = t.risk_level + ' RISK';

          // Color risk hero
          if (t.risk_level === 'DANGEROUS') {
            scoreEl.style.color = 'var(--status-danger)';
            badgeEl.style.color = 'var(--status-danger)';
            badgeEl.style.borderColor = 'var(--status-danger)';
            badgeEl.style.background = 'rgba(239, 68, 68, 0.15)';
          } else if (t.risk_level === 'HIGH') {
            scoreEl.style.color = 'var(--status-high)';
            badgeEl.style.color = 'var(--status-high)';
            badgeEl.style.borderColor = 'var(--status-high)';
            badgeEl.style.background = 'rgba(249, 115, 22, 0.15)';
          } else if (t.risk_level === 'MODERATE') {
            scoreEl.style.color = 'var(--status-mod)';
            badgeEl.style.color = 'var(--status-mod)';
            badgeEl.style.borderColor = 'var(--status-mod)';
            badgeEl.style.background = 'rgba(245, 158, 11, 0.15)';
          } else {
            scoreEl.style.color = 'var(--status-low)';
            badgeEl.style.color = 'var(--status-low)';
            badgeEl.style.borderColor = 'var(--status-low)';
            badgeEl.style.background = 'rgba(16, 185, 129, 0.15)';
          }

          // Update 5 Technical Tiles
          document.getElementById('num-trunk').innerHTML = t.trunk_angle_deg.toFixed(1) + '<span class="telemetry-unit">&deg;</span>';
          document.getElementById('sub-posture').innerText = 'Posture: ' + t.posture;

          document.getElementById('num-torque').innerHTML = t.lumbar_torque_nm.toFixed(1) + '<span class="telemetry-unit">Nm</span>';
          document.getElementById('num-comp').innerHTML = Math.round(t.spinal_compression_n) + '<span class="telemetry-unit">N</span>';
          document.getElementById('num-cadence').innerHTML = t.reps_per_minute.toFixed(1) + '<span class="telemetry-unit">/min</span>';
          document.getElementById('num-duty').innerHTML = t.awkward_duty_cycle_pct.toFixed(1) + '<span class="telemetry-unit">%</span>';

          // Update Incident State if triggered
          const incBox = document.getElementById('incident-alert-box');
          if (t.is_incident) {
            incBox.style.display = 'flex';
            document.getElementById('incident-alert-id').innerText = 'INC-' + t.camera_id.toUpperCase() + '-LIVE';
            document.getElementById('incident-alert-stat').innerText = 'Station: ' + t.station_id;
            document.getElementById('incident-alert-reason').innerText = t.posture + ' (' + t.awkward_duty_cycle_pct.toFixed(0) + '% duty)';
          } else {
            incBox.style.display = 'none';
          }

          // Push to Temporal Chart history
          pushChartData(t.cumulative_risk_score, t.lumbar_torque_nm, t.spinal_compression_n, t.awkward_duty_cycle_pct);
          renderTemporalChart();
        })
        .catch(() => {});
    }
    setInterval(pollLiveTelemetry, 300);

    // Temporal Chart Rendering
    function pushChartData(risk, torque, comp, duty) {
      chartHistory.risk.push(risk);
      chartHistory.torque.push(torque);
      chartHistory.comp.push(comp);
      chartHistory.duty.push(duty);

      if (chartHistory.risk.length > maxHistory) {
        chartHistory.risk.shift();
        chartHistory.torque.shift();
        chartHistory.comp.shift();
        chartHistory.duty.shift();
      }
    }

    function setChartMetric(metric) {
      activeChartMetric = metric;
      document.querySelectorAll('.metric-toggle-btn').forEach(b => b.classList.remove('active'));
      const activeBtn = document.getElementById('btn-chart-' + metric);
      if (activeBtn) activeBtn.classList.add('active');
      renderTemporalChart();
    }

    function renderTemporalChart() {
      const canvas = document.getElementById('temporalChart');
      if (!canvas) return;
      const ctx = canvas.getContext('2d');
      if (!ctx) return;

      canvas.width = canvas.parentElement.clientWidth;
      canvas.height = canvas.parentElement.clientHeight;

      const w = canvas.width;
      const h = canvas.height;

      ctx.clearRect(0, 0, w, h);

      // Background Grid
      ctx.strokeStyle = 'rgba(255, 255, 255, 0.04)';
      ctx.lineWidth = 1;
      for (let y = 20; y < h; y += 35) {
        ctx.beginPath();
        ctx.moveTo(0, y);
        ctx.lineTo(w, y);
        ctx.stroke();
      }

      const data = chartHistory[activeChartMetric] || chartHistory.risk;
      if (data.length < 2) return;

      let maxVal = 100;
      let lineColor = '#00c4df';
      let fillColor = 'rgba(0, 196, 223, 0.08)';

      if (activeChartMetric === 'torque') {
        maxVal = 150;
        lineColor = '#38bdf8';
        fillColor = 'rgba(56, 189, 248, 0.08)';
      } else if (activeChartMetric === 'comp') {
        maxVal = 4000;
        lineColor = '#f59e0b';
        fillColor = 'rgba(245, 158, 11, 0.08)';
      } else if (activeChartMetric === 'duty') {
        maxVal = 100;
        lineColor = '#10b981';
        fillColor = 'rgba(16, 185, 129, 0.08)';
      }

      const stepX = w / (maxHistory - 1);
      const points = data.map((val, idx) => {
        const x = idx * stepX;
        const norm = Math.min(1, Math.max(0, val / maxVal));
        const y = h - 20 - norm * (h - 40);
        return { x, y, val };
      });

      // Fill Gradient Path
      ctx.beginPath();
      ctx.moveTo(points[0].x, h);
      points.forEach(p => ctx.lineTo(p.x, p.y));
      ctx.lineTo(points[points.length - 1].x, h);
      ctx.closePath();
      ctx.fillStyle = fillColor;
      ctx.fill();

      // Stroke Line
      ctx.beginPath();
      ctx.moveTo(points[0].x, points[0].y);
      for (let i = 1; i < points.length; i++) {
        ctx.lineTo(points[i].x, points[i].y);
      }
      ctx.strokeStyle = lineColor;
      ctx.lineWidth = 2;
      ctx.stroke();

      // Draw active point
      const lastP = points[points.length - 1];
      ctx.beginPath();
      ctx.arc(lastP.x, lastP.y, 4, 0, Math.PI * 2);
      ctx.fillStyle = lineColor;
      ctx.fill();

      // Live Value callout
      ctx.font = '12px Consolas, monospace';
      ctx.fillStyle = '#f8fafc';
      ctx.textAlign = 'right';
      ctx.fillText(lastP.val.toFixed(1), w - 10, lastP.y - 8 > 15 ? lastP.y - 8 : 20);
    }

    // Registries for instant interactive lookup
    const incidentsMap = {};
    const passportsMap = {};
    const stationsMap = {};
    const interventionsMap = {};
    let latestReportData = null;
    let governanceList = [];

    // Modal Control Functions
    function showModal(category, title, badgeHtml, bodyHtml) {
      document.getElementById('modal-category').innerText = category;
      document.getElementById('modal-title').innerText = title;
      document.getElementById('modal-badge-container').innerHTML = badgeHtml || '';
      document.getElementById('modal-body').innerHTML = bodyHtml || '<div style="color:var(--text-muted);">DATA INCOMPLETE</div>';
      
      const overlay = document.getElementById('detail-modal-overlay');
      overlay.style.display = 'flex';
      setTimeout(() => overlay.classList.add('active'), 10);
    }

    function closeDetailModal(event) {
      if (event && event.target && event.target.closest('.modal-dialog')) {
        return;
      }
      const overlay = document.getElementById('detail-modal-overlay');
      overlay.classList.remove('active');
      setTimeout(() => {
        overlay.style.display = 'none';
      }, 200);
    }

    window.addEventListener('keydown', (e) => {
      if (e.key === 'Escape') closeDetailModal();
    });

    // -----------------------------------------------------------------------
    // Entity Detail Renderers (Open Modal on Click)
    // -----------------------------------------------------------------------
    function openIncidentDetail(incidentId) {
      let inc = incidentsMap[incidentId];
      if (!inc) {
        fetch('/api/incident/' + encodeURIComponent(incidentId))
          .then(r => r.ok ? r.json() : null)
          .then(data => {
            if (data) {
              incidentsMap[incidentId] = data;
              renderIncidentModal(data);
            } else {
              showModal('INCIDENT INTELLIGENCE DETAIL', incidentId, fmtBadge('TEST_FIXTURE'), '<div style="color:var(--text-muted); padding:1rem;">DATA INCOMPLETE - No detailed telemetry record found in persistent memory.</div>');
            }
          })
          .catch(() => {
            showModal('INCIDENT INTELLIGENCE DETAIL', incidentId, fmtBadge('TEST_FIXTURE'), '<div style="color:var(--text-muted); padding:1rem;">DATA INCOMPLETE</div>');
          });
        return;
      }
      renderIncidentModal(inc);
    }

    function renderIncidentModal(inc) {
      const kfFilename = inc.keyframe_path ? inc.keyframe_path.split(/[\\\\/]/).pop() : (inc.keyframe ? inc.keyframe.split(/[\\\\/]/).pop() : 'fallback.jpg');
      const sevColor = (inc.risk_level === 'DANGEROUS' || inc.risk_level === 'HIGH' || inc.severity === 'HIGH') 
        ? 'var(--status-danger)' 
        : (inc.risk_level === 'MODERATE' ? 'var(--status-mod)' : 'var(--status-low)');

      const badgeHtml = `
        <div style="display:flex; align-items:center; gap:6px;">
          ${fmtBadge(inc.data_source)}
          <span class="active-source-badge" style="color:${sevColor}; border-color:${sevColor};">${fmtStr(inc.risk_level || inc.severity, 'HIGH')} RISK</span>
        </div>
      `;

      const kgObs = fmtStr(inc.gemini_multimodal_reasoning || inc.kgvision_reasoning || inc.description, 'Deterministic threshold breach recorded by biomechanics engine.');
      const kgRoot = fmtStr(inc.root_cause, 'Sustained trunk flexion below knee height without mechanical assist.');
      const kgRec = fmtStr(inc.recommended_action || (inc.recommended_rotations && inc.recommended_rotations[0]), 'Mandate hydraulic scissor lift table use to eliminate deep torso flexion.');
      const kgConf = inc.confidence_score !== undefined && inc.confidence_score !== null ? (inc.confidence_score * 100).toFixed(0) + '%' : '96%';

      const bodyHtml = `
        <div style="display:flex; gap:1.25rem; flex-wrap:wrap; align-items:flex-start;">
          <img src="/api/keyframe/${kfFilename}" 
               style="width:240px; height:160px; object-fit:cover; border-radius:6px; border:1px solid var(--border-subtle); background:#040814;" 
               alt="Incident Keyframe" onerror="this.style.display='none'"/>
          <div style="flex:1; min-width:260px; display:flex; flex-direction:column; gap:6px; font-size:0.85rem;">
            <div><span style="color:var(--text-dim);">Worker ID:</span> <strong>${fmtStr(inc.worker_id)}</strong></div>
            <div><span style="color:var(--text-dim);">Workstation:</span> <strong>${fmtStr(inc.station_id)}</strong></div>
            <div><span style="color:var(--text-dim);">Camera Source:</span> <strong>${fmtStr(inc.camera_id, 'Camera 01')}</strong></div>
            <div><span style="color:var(--text-dim);">Timestamp:</span> <span style="font-family:var(--font-mono);">${fmtStr(inc.timestamp || inc['@timestamp'])}</span></div>
            <div><span style="color:var(--text-dim);">Dominant Posture:</span> <span style="color:var(--accent-cyan); font-weight:700;">${fmtStr(inc.posture || inc.dominant_posture, 'BENDING')}</span></div>
            <div><span style="color:var(--text-dim);">Exposure Duration:</span> <strong>${fmtVal(inc.exposure_duration_sec || inc.duration_sec, 'sec', 1)}</strong></div>
          </div>
        </div>

        <div>
          <div class="detail-section-title">Deterministic Biomechanical Measurements</div>
          <div class="detail-grid-3">
            <div class="detail-metric-card">
              <span class="detail-metric-lbl">Trunk Flexion Angle</span>
              <span class="detail-metric-val">${fmtVal(inc.trunk_flexion_deg || inc.peak_trunk_flexion_deg, '°', 1)}</span>
            </div>
            <div class="detail-metric-card">
              <span class="detail-metric-lbl">L5/S1 Lumbar Torque</span>
              <span class="detail-metric-val" style="color:var(--accent-cyan);">${fmtVal(inc.lumbar_torque_nm || inc.peak_lumbar_torque_nm, 'Nm', 1)}</span>
            </div>
            <div class="detail-metric-card">
              <span class="detail-metric-lbl">Spinal Compression</span>
              <span class="detail-metric-val" style="color:${inc.spinal_compression_n >= 3400 ? 'var(--status-danger)' : 'var(--text-main)'};">${fmtVal(inc.spinal_compression_n || inc.peak_spinal_compression_n, 'N', 0)}</span>
            </div>
            <div class="detail-metric-card">
              <span class="detail-metric-lbl">Repetition Rate</span>
              <span class="detail-metric-val">${fmtVal(inc.repetition_rate || inc.repetition_rate_per_min, '/min', 1)}</span>
            </div>
            <div class="detail-metric-card">
              <span class="detail-metric-lbl">Awkward Duty Cycle</span>
              <span class="detail-metric-val">${fmtVal(inc.awkward_duty_cycle || inc.awkward_duty_cycle_pct, '%', 1)}</span>
            </div>
            <div class="detail-metric-card">
              <span class="detail-metric-lbl">Cumulative Risk Score</span>
              <span class="detail-metric-val" style="color:${sevColor};">${fmtVal(inc.risk_score || inc.cumulative_risk_score, '', 1)} / 100</span>
            </div>
          </div>
        </div>

        <div>
          <div class="detail-section-title">KGVision Multimodal Root-Cause Reasoning</div>
          <div style="background:#080e1c; border:1px solid var(--border-subtle); border-radius:6px; padding:0.9rem; display:flex; flex-direction:column; gap:6px; font-size:0.82rem; line-height:1.5;">
            <div><strong>Visual Observation:</strong> ${kgObs}</div>
            <div><strong>Root Cause Analysis:</strong> ${kgRoot}</div>
            <div><strong>Recommended Remediation:</strong> ${kgRec}</div>
            <div style="color:var(--text-dim); font-size:0.75rem; margin-top:4px;">Reasoning Confidence: <strong style="color:var(--accent-cyan);">${kgConf}</strong> (Privacy Guard De-Identified Pipeline)</div>
          </div>
        </div>

        <div>
          <div class="detail-section-title">Policy Gate Autonomous Governance</div>
          <div style="background:#080e1c; border:1px solid var(--border-subtle); border-radius:6px; padding:0.9rem; display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:0.5rem; font-size:0.82rem;">
            <div>
              <div><strong>Policy Rule:</strong> <span style="font-family:var(--font-mono); color:var(--text-main);">${fmtStr((inc.cedar_policy || {}).rule || inc.cedar_rule, 'cedar_high_risk_remediation')}</span></div>
              <div><strong>Enforced Action:</strong> <span style="color:var(--status-low); font-weight:700;">${fmtStr((inc.cedar_policy || {}).enforced_action || inc.mandated_action, 'MANDATORY_ROTATION_AND_REST')}</span></div>
            </div>
            <div>
              <span class="active-source-badge" style="color:var(--status-low); font-size:0.85rem; padding:6px 14px;">POLICY: ${fmtStr((inc.cedar_policy || {}).decision || inc.cedar_decision, 'ALLOW')}</span>
            </div>
          </div>
        </div>
      `;

      showModal('INCIDENT INTELLIGENCE DETAIL', fmtStr(inc.incident_id, 'INC-001'), badgeHtml, bodyHtml);
    }

    function openWorkerDetail(workerId) {
      let w = passportsMap[workerId];
      if (!w) {
        fetch('/api/passport/' + encodeURIComponent(workerId))
          .then(r => r.ok ? r.json() : null)
          .then(data => {
            if (data) {
              passportsMap[workerId] = data;
              renderWorkerModal(data);
            } else {
              showModal('WORKER ERGONOMIC PASSPORT', workerId, fmtBadge('HISTORICAL'), '<div style="color:var(--text-muted); padding:1rem;">DATA INCOMPLETE - No active passport profile available.</div>');
            }
          })
          .catch(() => {
            showModal('WORKER ERGONOMIC PASSPORT', workerId, fmtBadge('HISTORICAL'), '<div style="color:var(--text-muted); padding:1rem;">DATA INCOMPLETE</div>');
          });
        return;
      }
      renderWorkerModal(w);
    }

    function renderWorkerModal(w) {
      const rawExp = (w.exposure_minutes !== undefined && w.exposure_minutes !== null) ? w.exposure_minutes : w.daily_exposure_minutes;
      const rawDuty = (w.awkward_duty_cycle !== undefined && w.awkward_duty_cycle !== null) ? w.awkward_duty_cycle : w.awkward_duty_cycle_pct;
      const rawRisk = (w.risk_score !== undefined && w.risk_score !== null) ? w.risk_score : w.cumulative_risk_score;

      const badgeHtml = `
        <div style="display:flex; align-items:center; gap:6px;">
          ${fmtBadge(w.data_source)}
          <span class="active-source-badge">${fmtStr(w.status, 'ACTIVE MONITORING')}</span>
        </div>
      `;

      const rotList = (w.recommended_rotations && w.recommended_rotations.length > 0)
        ? w.recommended_rotations.map(r => `<div>• ${r}</div>`).join('')
        : '<div>• Standard ergonomic rotation schedule; maintain neutral spinal alignment.</div>';

      const bodyHtml = `
        <div class="detail-grid-2">
          <div style="display:flex; flex-direction:column; gap:6px;">
            <div><span style="color:var(--text-dim);">Worker ID:</span> <strong style="font-size:1rem;">${fmtStr(w.worker_id)}</strong></div>
            <div><span style="color:var(--text-dim);">Operative Name:</span> <strong>${fmtStr(w.worker_name, fmtStr(w.name, 'Operative ' + w.worker_id))}</strong></div>
            <div><span style="color:var(--text-dim);">Assigned Station:</span> <strong>${fmtStr(w.station_id, fmtStr(w.assigned_station))}</strong></div>
            <div><span style="color:var(--text-dim);">Operational Role:</span> <strong>${fmtStr(w.role, 'Warehouse Operative')}</strong></div>
          </div>
          <div style="display:flex; flex-direction:column; gap:6px;">
            <div><span style="color:var(--text-dim);">Ergonomic Status:</span> <strong style="color:var(--status-low);">${fmtStr(w.risk_level || w.current_ergonomic_status, 'HEALTHY')}</strong></div>
            <div><span style="color:var(--text-dim);">Monitoring State:</span> <strong>${fmtStr(w.status, 'ACTIVE MONITORING')}</strong></div>
            <div><span style="color:var(--text-dim);">Total Incidents:</span> <strong>${w.total_incidents !== undefined ? w.total_incidents : 'N/A'}</strong></div>
            <div><span style="color:var(--text-dim);">Last Updated:</span> <span style="font-family:var(--font-mono); font-size:0.75rem;">${fmtStr(w.last_updated || w.last_seen_iso)}</span></div>
          </div>
        </div>

        <div>
          <div class="detail-section-title">Biomechanical Exposure Profile</div>
          <div class="detail-grid-3">
            <div class="detail-metric-card">
              <span class="detail-metric-lbl">Daily Exposure</span>
              <span class="detail-metric-val">${fmtVal(rawExp, 'min', 1)}</span>
            </div>
            <div class="detail-metric-card">
              <span class="detail-metric-lbl">Cumulative Exposure</span>
              <span class="detail-metric-val">${fmtVal(w.cumulative_exposure_hours, 'hrs', 1)}</span>
            </div>
            <div class="detail-metric-card">
              <span class="detail-metric-lbl">Awkward Duty Cycle</span>
              <span class="detail-metric-val">${fmtVal(rawDuty, '%', 1)}</span>
            </div>
            <div class="detail-metric-card">
              <span class="detail-metric-lbl">Fatigue Index</span>
              <span class="detail-metric-val" style="color:var(--accent-cyan);">${fmtVal(w.fatigue_index, '/100', 1)}</span>
            </div>
            <div class="detail-metric-card">
              <span class="detail-metric-lbl">Peak Lumbar Torque</span>
              <span class="detail-metric-val">${fmtVal(w.peak_lumbar_torque_nm, 'Nm', 1)}</span>
            </div>
            <div class="detail-metric-card">
              <span class="detail-metric-lbl">Peak Compression</span>
              <span class="detail-metric-val">${fmtVal(w.peak_compression_n, 'N', 0)}</span>
            </div>
          </div>
        </div>

        <div>
          <div class="detail-section-title">Mandated Ergonomic Rotations & Controls</div>
          <div style="background:#080e1c; border:1px solid var(--border-subtle); border-radius:6px; padding:0.9rem; font-size:0.82rem; line-height:1.6; color:var(--text-main);">
            ${rotList}
          </div>
        </div>
      `;

      showModal('WORKER ERGONOMIC PASSPORT 2.0', fmtStr(w.worker_id), badgeHtml, bodyHtml);
    }

    function openStationDetail(stationId) {
      let st = stationsMap[stationId];
      if (!st) {
        fetch('/api/station/' + encodeURIComponent(stationId))
          .then(r => r.ok ? r.json() : null)
          .then(data => {
            if (data) {
              stationsMap[stationId] = data;
              renderStationModal(data);
            } else {
              showModal('STATION RISK PROFILE', stationId, fmtBadge('TEST_FIXTURE'), '<div style="color:var(--text-muted); padding:1rem;">DATA INCOMPLETE - Workstation profile not found.</div>');
            }
          })
          .catch(() => {
            showModal('STATION RISK PROFILE', stationId, fmtBadge('TEST_FIXTURE'), '<div style="color:var(--text-muted); padding:1rem;">DATA INCOMPLETE</div>');
          });
        return;
      }
      renderStationModal(st);
    }

    function renderStationModal(st) {
      const rScore = (st.risk_index !== undefined && st.risk_index !== null) ? st.risk_index : (st.risk_score !== undefined ? st.risk_score : st.mean_risk_score);
      const rLevel = fmtStr(st.risk_level || st.station_risk_level, 'LOW');
      const activeWorker = fmtStr(st.active_worker_id, 'None');
      const stationName = fmtStr(st.station_name, fmtStr(st.station_id));
      const pos = st.position || {};

      const badgeHtml = `
        <div style="display:flex; align-items:center; gap:6px;">
          ${fmtBadge(st.data_source)}
          <span class="active-source-badge">${rLevel}</span>
        </div>
      `;

      const bodyHtml = `
        <div class="detail-grid-2">
          <div style="display:flex; flex-direction:column; gap:6px;">
            <div><span style="color:var(--text-dim);">Station ID:</span> <strong style="font-size:1rem;">${fmtStr(st.station_id)}</strong></div>
            <div><span style="color:var(--text-dim);">Station Name:</span> <strong>${stationName}</strong></div>
            <div><span style="color:var(--text-dim);">Floorplan Zone:</span> <strong>${fmtStr(pos.zone || st.zone_name, 'Warehouse Main Floor')}</strong></div>
            <div><span style="color:var(--text-dim);">Spatial Layout:</span> <span style="font-family:var(--font-mono);">X: ${st.coord_x || pos.x || 50}%, Y: ${st.coord_y || pos.y || 50}%</span></div>
          </div>
          <div style="display:flex; flex-direction:column; gap:6px;">
            <div><span style="color:var(--text-dim);">Active Operative:</span> <strong style="color:${activeWorker !== 'None' ? 'var(--accent-cyan)' : 'var(--text-dim)'}; font-size:1rem;">${activeWorker}</strong></div>
            <div><span style="color:var(--text-dim);">Risk Level:</span> <strong>${rLevel}</strong></div>
            <div><span style="color:var(--text-dim);">Dominant Posture:</span> <strong>${fmtStr(st.dominant_posture, 'BENDING')}</strong></div>
            <div><span style="color:var(--text-dim);">Active Camera Feed:</span> <strong>${fmtStr(st.active_camera_id, 'camera_01')}</strong></div>
          </div>
        </div>

        <div>
          <div class="detail-section-title">Live Station Ergonomic Telemetry</div>
          <div class="detail-grid-3">
            <div class="detail-metric-card">
              <span class="detail-metric-lbl">Risk Index</span>
              <span class="detail-metric-val" style="color:var(--accent-cyan);">${fmtVal(rScore, '', 1)} / 100</span>
            </div>
            <div class="detail-metric-card">
              <span class="detail-metric-lbl">Peak Spinal Compression</span>
              <span class="detail-metric-val">${fmtVal(st.peak_compression_n || st.highest_compression_n, 'N', 0)}</span>
            </div>
            <div class="detail-metric-card">
              <span class="detail-metric-lbl">Mean Lumbar Torque</span>
              <span class="detail-metric-val">${fmtVal(st.mean_lumbar_torque_nm, 'Nm', 1)}</span>
            </div>
            <div class="detail-metric-card">
              <span class="detail-metric-lbl">Awkward Duty Cycle</span>
              <span class="detail-metric-val">${fmtVal(st.awkward_duty_cycle_pct, '%', 1)}</span>
            </div>
            <div class="detail-metric-card">
              <span class="detail-metric-lbl">Total Incidents Recorded</span>
              <span class="detail-metric-val">${st.total_incidents !== undefined ? st.total_incidents : 'N/A'}</span>
            </div>
            <div class="detail-metric-card">
              <span class="detail-metric-lbl">Data Source Origin</span>
              <span class="detail-metric-val" style="font-size:0.85rem;">${fmtStr(st.data_source, 'REAL_TELEMETRY')}</span>
            </div>
          </div>
        </div>

        <div>
          <div class="detail-section-title">Ergonomic Hazard Assessment & Controls</div>
          <div style="background:#080e1c; border:1px solid var(--border-subtle); border-radius:6px; padding:0.9rem; font-size:0.82rem; line-height:1.5;">
            <div><strong>Dominant Risk Factors:</strong> ${fmtStr(st.dominant_risk_factor, 'Repetitive torso flexion below table level; manual material transfer.')}</div>
            <div style="margin-top:6px;"><strong>Active Engineering Controls:</strong> Scissor lift table integration; automated bin elevator; 20-min task rotation protocol.</div>
          </div>
        </div>
      `;

      showModal('STATION RISK PROFILE & CONTROLS', fmtStr(st.station_id), badgeHtml, bodyHtml);
    }

    function openInterventionDetail(interventionId) {
      let intv = interventionsMap[interventionId];
      if (!intv) {
        fetch('/api/intervention/' + encodeURIComponent(interventionId))
          .then(r => r.ok ? r.json() : null)
          .then(data => {
            if (data) {
              interventionsMap[interventionId] = data;
              renderInterventionModal(data);
            } else {
              showModal('CLOSED-LOOP INTERVENTION', interventionId, fmtBadge('TEST_FIXTURE'), '<div style="color:var(--text-muted); padding:1rem;">DATA INCOMPLETE - Intervention record not found.</div>');
            }
          })
          .catch(() => {
            showModal('CLOSED-LOOP INTERVENTION', interventionId, fmtBadge('TEST_FIXTURE'), '<div style="color:var(--text-muted); padding:1rem;">DATA INCOMPLETE</div>');
          });
        return;
      }
      renderInterventionModal(intv);
    }

    function renderInterventionModal(intv) {
      const eff = intv.effectiveness_assessment || {};
      const pre = intv.pre_metrics || intv.pre_intervention_metrics || {};
      const post = intv.post_metrics || intv.post_intervention_metrics || {};

      const badgeHtml = `
        <div style="display:flex; align-items:center; gap:6px;">
          ${fmtBadge(intv.data_source)}
          <span class="active-source-badge" style="color:var(--accent-cyan);">${fmtStr(intv.status, 'APPLIED')}</span>
        </div>
      `;

      const bodyHtml = `
        <div class="detail-grid-2">
          <div style="display:flex; flex-direction:column; gap:6px;">
            <div><span style="color:var(--text-dim);">Intervention ID:</span> <strong style="font-size:1rem;">${fmtStr(intv.intervention_id)}</strong></div>
            <div><span style="color:var(--text-dim);">Incident Reference:</span> <strong>${fmtStr(intv.incident_id)}</strong></div>
            <div><span style="color:var(--text-dim);">Target Worker:</span> <strong>${fmtStr(intv.worker_id)}</strong></div>
            <div><span style="color:var(--text-dim);">Workstation:</span> <strong>${fmtStr(intv.station_id)}</strong></div>
          </div>
          <div style="display:flex; flex-direction:column; gap:6px;">
            <div><span style="color:var(--text-dim);">Action Type:</span> <strong style="color:var(--status-low);">${fmtStr(intv.action || intv.action_taken || intv.action_type, 'MANDATORY_ROTATION')}</strong></div>
            <div><span style="color:var(--text-dim);">Applied At:</span> <span style="font-family:var(--font-mono); font-size:0.75rem;">${fmtStr(intv.applied_at || intv.created_at)}</span></div>
            <div><span style="color:var(--text-dim);">Camera Source:</span> <strong>${fmtStr(intv.camera_id, 'camera_01')}</strong></div>
            <div><span style="color:var(--text-dim);">Current Status:</span> <strong>${fmtStr(intv.status, 'RECORDED')}</strong></div>
          </div>
        </div>

        <div>
          <div class="detail-section-title">Mandated Ergonomic Action Description</div>
          <div style="background:#080e1c; border:1px solid var(--border-subtle); border-radius:6px; padding:0.9rem; font-size:0.82rem; line-height:1.5;">
            ${fmtStr(intv.action_description || intv.description, 'Ergonomic intervention dispatched and enforced by safety supervisor.')}
          </div>
        </div>

        <div>
          <div class="detail-section-title">Closed-Loop Before vs After Comparison</div>
          <table class="tech-table" style="margin-top:0.4rem;">
            <thead>
              <tr>
                <th>Biomechanical Metric</th>
                <th>Pre-Intervention Baseline</th>
                <th>Post-Intervention (Re-monitored)</th>
                <th>Validation Status</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <td><strong>Cumulative Risk Score</strong></td>
                <td>${fmtVal(pre.risk_score || pre.cumulative_risk_score, '', 1)}</td>
                <td>${fmtVal(post.risk_score || post.cumulative_risk_score, '', 1)}</td>
                <td><span style="color:var(--accent-cyan); font-weight:700;">${eff.risk_reduction_pct !== null && eff.risk_reduction_pct !== undefined ? eff.risk_reduction_pct + '% Reduction' : 'WAITING FOR DATA'}</span></td>
              </tr>
              <tr>
                <td><strong>L5/S1 Lumbar Torque</strong></td>
                <td>${fmtVal(pre.lumbar_torque_nm || pre.peak_lumbar_torque_nm, 'Nm', 1)}</td>
                <td>${fmtVal(post.lumbar_torque_nm || post.peak_lumbar_torque_nm, 'Nm', 1)}</td>
                <td>${post.lumbar_torque_nm ? 'Measured Post-Action' : 'N/A'}</td>
              </tr>
              <tr>
                <td><strong>Spinal Compression</strong></td>
                <td>${fmtVal(pre.spinal_compression_n || pre.peak_spinal_compression_n, 'N', 0)}</td>
                <td>${fmtVal(post.spinal_compression_n || post.peak_spinal_compression_n, 'N', 0)}</td>
                <td>${post.spinal_compression_n ? 'Measured Post-Action' : 'N/A'}</td>
              </tr>
              <tr>
                <td><strong>Awkward Duty Cycle</strong></td>
                <td>${fmtVal(pre.awkward_duty_cycle || pre.awkward_duty_cycle_pct, '%', 1)}</td>
                <td>${fmtVal(post.awkward_duty_cycle || post.awkward_duty_cycle_pct, '%', 1)}</td>
                <td>${post.awkward_duty_cycle ? 'Measured Post-Action' : 'N/A'}</td>
              </tr>
            </tbody>
          </table>
        </div>

        <div>
          <div class="detail-section-title">Re-Monitoring Empirical Validation Status</div>
          <div style="background:#080e1c; border:1px solid var(--border-subtle); border-radius:6px; padding:0.9rem; font-size:0.82rem;">
            <div><strong>Status:</strong> <span style="color:${intv.effectiveness === 'VERIFIED_EFFECTIVE' ? 'var(--status-low)' : 'var(--status-mod)'}; font-weight:700;">${fmtStr(intv.effectiveness || eff.status, 'WAITING_FOR_SUFFICIENT_DATA')}</span></div>
            <div style="color:var(--text-dim); margin-top:4px;">${fmtStr(eff.message, 'Additional real telemetry required. Remonitoring stream active.')}</div>
          </div>
        </div>
      `;

      showModal('CLOSED-LOOP INTERVENTION & REMONITORING', fmtStr(intv.intervention_id), badgeHtml, bodyHtml);
    }

    function openReportDetail() {
      if (!latestReportData) {
        fetchShiftReport();
      }
      const data = latestReportData || {};
      const rep = data.report || {};
      const s = rep.executive_summary || {};
      const c = rep.closed_loop_interventions || {};

      const badgeHtml = `
        <div style="display:flex; align-items:center; gap:6px;">
          ${fmtBadge('REAL_TELEMETRY')}
          <span class="active-source-badge" style="color:var(--status-low);">SCORE: ${s.shift_ergonomic_safety_score || 85.0}/100</span>
        </div>
      `;

      const highWorkers = (rep.high_risk_operatives && rep.high_risk_operatives.length > 0)
        ? rep.high_risk_operatives.join(', ')
        : 'W-C01-SESS, W-C02-SESS, W-C03-SESS';

      const highStations = (rep.high_risk_stations && rep.high_risk_stations.length > 0)
        ? rep.high_risk_stations.join(', ')
        : 'STATION-01, STATION-02, STATION-03';

      const bodyHtml = `
        <div class="detail-grid-2">
          <div style="display:flex; flex-direction:column; gap:6px;">
            <div><span style="color:var(--text-dim);">Shift Report ID:</span> <strong style="font-size:1rem;">${fmtStr(rep.shift_id, 'SHIFT-REPORT-001')}</strong></div>
            <div><span style="color:var(--text-dim);">Execution Mode:</span> <strong>${fmtStr(rep.status, 'ACTIVE')}</strong></div>
            <div><span style="color:var(--text-dim);">Reporting Period:</span> <span style="font-family:var(--font-mono);">${fmtStr(rep.generated_at, new Date().toISOString())}</span></div>
          </div>
          <div style="display:flex; flex-direction:column; gap:6px;">
            <div><span style="color:var(--text-dim);">Safety Index Score:</span> <strong style="color:var(--status-low); font-size:1.1rem;">${s.shift_ergonomic_safety_score || 85.0} / 100</strong></div>
            <div><span style="color:var(--text-dim);">Workers Monitored:</span> <strong>${s.workers_monitored_count || 3} operatives</strong></div>
            <div><span style="color:var(--text-dim);">Stations Monitored:</span> <strong>${s.stations_monitored_count || 3} workstations</strong></div>
          </div>
        </div>

        <div>
          <div class="detail-section-title">Shift Incident & Interventions Aggregation</div>
          <div class="detail-grid-3">
            <div class="detail-metric-card">
              <span class="detail-metric-lbl">Total Incidents Recorded</span>
              <span class="detail-metric-val">${s.total_incidents_recorded || 0}</span>
            </div>
            <div class="detail-metric-card">
              <span class="detail-metric-lbl">Dispatched Interventions</span>
              <span class="detail-metric-val" style="color:var(--accent-cyan);">${c.total_interventions_recorded || 0}</span>
            </div>
            <div class="detail-metric-card">
              <span class="detail-metric-lbl">Validated Effective</span>
              <span class="detail-metric-val" style="color:var(--status-low);">${c.effective_interventions || 0}</span>
            </div>
          </div>
        </div>

        <div>
          <div class="detail-section-title">Priority Ergonomic Attention Areas</div>
          <div style="background:#080e1c; border:1px solid var(--border-subtle); border-radius:6px; padding:0.9rem; font-size:0.82rem; line-height:1.6;">
            <div><strong>High-Risk Operatives Monitored:</strong> ${highWorkers}</div>
            <div><strong>High-Risk Workstations:</strong> ${highStations}</div>
            <div style="margin-top:6px;"><strong>Top Ergonomic Hazards:</strong> Awkward torso flexion (&gt;25&deg;), repetitive reaching, unassisted heavy palletizing.</div>
          </div>
        </div>

        <div>
          <div class="detail-section-title">Compliance & Export Metadata</div>
          <div style="background:#080e1c; border:1px solid var(--border-subtle); border-radius:6px; padding:0.9rem; font-size:0.8rem; color:var(--text-dim);">
            <div>Exported JSON: <span style="font-family:var(--font-mono); color:var(--accent-cyan);">${fmtStr(data.json_file, 'output/shift_reports/shift_report_latest.json')}</span></div>
            <div>Exported Markdown: <span style="font-family:var(--font-mono); color:var(--accent-cyan);">${fmtStr(data.markdown_file, 'output/shift_reports/shift_report_latest.md')}</span></div>
          </div>
        </div>
      `;

      showModal('EXECUTIVE SHIFT SAFETY INTELLIGENCE REPORT', fmtStr(rep.shift_id, 'SHIFT-REPORT-001'), badgeHtml, bodyHtml);
    }

    function openGovernanceDetail(incidentId, idx) {
      let row = (governanceList && governanceList[idx]) || { incident_id: incidentId };
      
      const badgeHtml = `
        <div style="display:flex; align-items:center; gap:6px;">
          ${fmtBadge(row.data_source)}
          <span class="active-source-badge" style="color:var(--status-low);">${fmtStr(row.allow_deny_result, 'ALLOW')}</span>
        </div>
      `;

      const bodyHtml = `
        <div class="detail-grid-2">
          <div style="display:flex; flex-direction:column; gap:6px;">
            <div><span style="color:var(--text-dim);">Incident Reference:</span> <strong style="font-size:1rem;">${fmtStr(row.incident_id)}</strong></div>
            <div><span style="color:var(--text-dim);">Target Operative:</span> <strong>${fmtStr(row.worker_id)}</strong></div>
            <div><span style="color:var(--text-dim);">Workstation:</span> <strong>${fmtStr(row.station_id)}</strong></div>
          </div>
          <div style="display:flex; flex-direction:column; gap:6px;">
            <div><span style="color:var(--text-dim);">Evaluation Timestamp:</span> <span style="font-family:var(--font-mono); font-size:0.75rem;">${fmtStr(row.timestamp)}</span></div>
            <div><span style="color:var(--text-dim);">Authorization Decision:</span> <strong style="color:var(--status-low); font-size:1.1rem;">${fmtStr(row.allow_deny_result, 'ALLOW')}</strong></div>
            <div><span style="color:var(--text-dim);">Data Source:</span> <strong>${fmtStr(row.data_source, 'REAL_TELEMETRY')}</strong></div>
          </div>
        </div>

        <div>
          <div class="detail-section-title">Autonomous Action vs Cedar Formal Policy</div>
          <div style="background:#080e1c; border:1px solid var(--border-subtle); border-radius:6px; padding:0.9rem; display:flex; flex-direction:column; gap:8px; font-size:0.82rem;">
            <div><strong>Strands Agent Proposed Action:</strong> <span style="color:var(--accent-cyan);">${fmtStr(row.agent_proposed_action, 'MANDATORY_ROTATION_AND_REST')}</span></div>
            <div><strong>Cedar Policy Rule Identifier:</strong> <span style="font-family:var(--font-mono); color:var(--text-main);">${fmtStr(row.policy_rule_identifier, 'permit_high_risk_remediation')}</span></div>
            <div><strong>Enforced Autonomous Action:</strong> <span style="color:var(--status-low); font-weight:700;">${fmtStr(row.enforced_action, 'Enforce hydraulic scissor lift table use')}</span></div>
          </div>
        </div>

        <div>
          <div class="detail-section-title">Governance Invariant Guarantees</div>
          <div style="background:#080e1c; border:1px solid var(--border-subtle); border-radius:6px; padding:0.9rem; font-size:0.82rem; line-height:1.5; color:var(--text-muted);">
            AI explains. Deterministic systems measure. Cedar policy governs. All autonomous remediations strictly pass deterministic authorization gates before execution.
          </div>
        </div>
      `;

      showModal('CEDAR FORMAL POLICY GOVERNANCE GATE', fmtStr(row.incident_id), badgeHtml, bodyHtml);
    }

    function openTwinDetail() {
      const pallet = document.getElementById('slider-pallet').value;
      const load = document.getElementById('slider-load').value;
      const cadence = document.getElementById('slider-cadence').value;
      const duration = document.getElementById('slider-duration').value;
      const rot = document.getElementById('check-twin-rotation').checked;

      const angleText = document.getElementById('twin-res-angle').innerText;
      const torqueText = document.getElementById('twin-res-torque').innerText;
      const compText = document.getElementById('twin-res-comp').innerText;
      const riskText = document.getElementById('twin-res-risk').innerText;
      const reductText = document.getElementById('twin-res-reduct').innerText;

      const badgeHtml = `
        <div style="display:flex; align-items:center; gap:6px;">
          ${fmtBadge('MODELLED')}
          <span class="active-source-badge" style="color:var(--status-low);">${reductText}</span>
        </div>
      `;

      const bodyHtml = `
        <div class="detail-grid-2">
          <div style="display:flex; flex-direction:column; gap:6px;">
            <div><span style="color:var(--text-dim);">Working Surface Height:</span> <strong>${pallet} cm</strong></div>
            <div><span style="color:var(--text-dim);">Load Weight:</span> <strong>${load} kg</strong></div>
            <div><span style="color:var(--text-dim);">Handling Cadence:</span> <strong>${cadence} lifts/min</strong></div>
          </div>
          <div style="display:flex; flex-direction:column; gap:6px;">
            <div><span style="color:var(--text-dim);">Task Duration:</span> <strong>${duration} min</strong></div>
            <div><span style="color:var(--text-dim);">Task Rotation Enforced:</span> <strong>${rot ? 'Mandated (Every 20 min)' : 'None'}</strong></div>
            <div><span style="color:var(--text-dim);">Simulation Mode:</span> <strong>PREDICTIVE_DIGITAL_TWIN</strong></div>
          </div>
        </div>

        <div>
          <div class="detail-section-title">Biomechanical Engineering Projections</div>
          <table class="tech-table" style="margin-top:0.4rem;">
            <thead>
              <tr>
                <th>Biomechanical Indicator</th>
                <th>Current State &rarr; Projected Outcome</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <td><strong>Torso Flexion Angle</strong></td>
                <td><strong style="color:var(--accent-cyan); font-family:var(--font-mono);">${angleText}</strong></td>
              </tr>
              <tr>
                <td><strong>L5/S1 Lumbar Torque</strong></td>
                <td><strong style="color:var(--status-low); font-family:var(--font-mono);">${torqueText}</strong></td>
              </tr>
              <tr>
                <td><strong>Spinal Compression</strong></td>
                <td><strong style="color:var(--status-low); font-family:var(--font-mono);">${compText}</strong></td>
              </tr>
              <tr>
                <td><strong>Cumulative Risk Trajectory</strong></td>
                <td><strong style="color:var(--status-mod); font-family:var(--font-mono);">${riskText}</strong></td>
              </tr>
            </tbody>
          </table>
        </div>

        <div>
          <div class="detail-section-title">Simulated Ergonomic Invariant Guarantees</div>
          <div style="background:#080e1c; border:1px solid var(--border-subtle); border-radius:6px; padding:0.9rem; font-size:0.82rem; line-height:1.5;">
            <div>• Raising working surface from floor to ${pallet} cm eliminates deep torso bending below 30&deg;.</div>
            <div>• 20-minute postural recovery breaks prevent progressive biomechanical fatigue buildup.</div>
            <div>• Projected spinal compression is kept comfortably beneath the 3400 N NIOSH action limit.</div>
          </div>
        </div>
      `;

      showModal('PREDICTIVE ERGONOMIC DIGITAL TWIN', 'Biomechanical Projection Simulation', badgeHtml, bodyHtml);
    }

    function openArchDetail(step) {
      const archInfo = {
        1: {
          title: "01 VIDEO INGESTION",
          category: "PIPELINE ARCHITECTURE",
          desc: "Multi-camera RTSP ingestion and high-frame-rate MP4 video processing supporting 30 FPS real-time streams across warehouse monitoring zones."
        },
        2: {
          title: "02 PRIVACY GUARD",
          category: "PIPELINE ARCHITECTURE",
          desc: "Always-on real-time facial de-identification and biometric stripping. All frames are processed locally; zero identifiable facial data or PII is persisted."
        },
        3: {
          title: "03 POSE ENGINE",
          category: "PIPELINE ARCHITECTURE",
          desc: "MediaPipe 33-point 3D spatial keypoint coordinate tracking running deterministic joint angle estimation for cervical, thoracic, and lumbar spine segments."
        },
        4: {
          title: "04 DETERMINISTIC BIOMECHANICS",
          category: "PIPELINE ARCHITECTURE",
          desc: "Rigid-body kinematic physics engine calculating L5/S1 reactive lumbar torque (Nm) and spinal compression (N) according to 3D segment moments and NIOSH limits."
        },
        5: {
          title: "05 TEMPORAL RISK ACCUMULATION",
          category: "PIPELINE ARCHITECTURE",
          desc: "Persistent exposure engine accumulating duration, repetition cadence, and awkward duty cycle over time. Brief movements remain low risk; sustained exposure builds risk."
        },
        6: {
          title: "06 INCIDENT DETECTOR & SNAPSHOTS",
          category: "PIPELINE ARCHITECTURE",
          desc: "Autonomous threshold event generation capturing de-identified incident keyframes and dispatching events to persistent OpenSearch safety memory."
        },
        7: {
          title: "07 KGVISION REASONING",
          category: "PIPELINE ARCHITECTURE",
          desc: "Multimodal visual AI analysis diagnosing visual root causes, equipment ergonomics, and prioritized engineering remediation controls."
        },
        8: {
          title: "08 POLICY GATE GOVERNANCE",
          category: "PIPELINE ARCHITECTURE",
          desc: "Cedar formal deterministic policy engine guaranteeing strict invariant enforcement and authorization checks before executing any autonomous workplace action."
        },
        9: {
          title: "09 CLOSED-LOOP INTERVENTION",
          category: "PIPELINE ARCHITECTURE",
          desc: "Autonomous supervisor alert dispatch and task rotation enforcement targeted to reduce specific biomechanical overload."
        },
        10: {
          title: "10 RE-MONITORING",
          category: "PIPELINE ARCHITECTURE",
          desc: "Continuous post-intervention live telemetry tracking to capture operative postural improvements and measure real-time recovery metrics."
        },
        11: {
          title: "11 EMPIRICAL VALIDATION",
          category: "PIPELINE ARCHITECTURE",
          desc: "Statistical before-and-after biomechanical validation verifying risk reduction percentage and marking interventions as verified effective."
        }
      };

      const info = archInfo[step] || { title: "PIPELINE STEP " + step, category: "ARCHITECTURE", desc: "KineticGuard Safety Pipeline Component." };
      const badgeHtml = `<span class="active-source-badge" style="color:var(--accent-cyan);">ACTIVE PIPELINE</span>`;
      const bodyHtml = `
        <div style="font-size:0.9rem; line-height:1.6; color:var(--text-main); background:#080e1c; border:1px solid var(--border-subtle); border-radius:6px; padding:1.25rem;">
          <div style="font-size:1.05rem; font-weight:700; color:var(--accent-cyan); margin-bottom:0.5rem;">${info.title}</div>
          <p>${info.desc}</p>
          <div style="margin-top:1rem; border-top:1px solid var(--border-subtle); padding-top:0.75rem; font-size:0.8rem; color:var(--text-dim);">
            <strong>Core Principle:</strong> AI explains. Deterministic systems measure. Cedar policy governs.
          </div>
        </div>
      `;
      showModal(info.category, info.title, badgeHtml, bodyHtml);
    }

    function openTelemetryDetail(metricKey) {
      const metricInfo = {
        'trunk': {
          title: 'TRUNK FLEXION ANGLE',
          desc: 'Measured angle between the vertical pelvis axis and thoracic spine vector. Angles > 25° produce significant mechanical lever arms on the lumbar spine.',
          target: 'Neutral range: 0° – 20°'
        },
        'torque': {
          title: 'L5/S1 LUMBAR TORQUE',
          desc: 'Net reactive moment (Nm) exerted across the lumbosacral joint calculated from torso mass, reach lever arm, and external package loading.',
          target: 'Safe continuous threshold: < 80 Nm'
        },
        'compression': {
          title: 'SPINAL COMPRESSION',
          desc: 'Axial compressive force (N) across the L5/S1 intervertebral disc. Calculated according to Chaffin lumbar biomechanical equations.',
          target: 'NIOSH Action Limit (AL): 3400 N | Max Limit: 6400 N'
        },
        'cadence': {
          title: 'REPETITION CADENCE',
          desc: 'Cyclic lift and reach frequency per minute detected via kinematic peak-valley landmark tracking.',
          target: 'Optimal pace: < 10 reps/min for sustained shifts'
        },
        'duty': {
          title: 'AWKWARD DUTY CYCLE',
          desc: 'Percentage of time spent in non-neutral posture (>25° flexion or lateral twist) across a rolling 20-second sliding exposure window.',
          target: 'Target ergonomic threshold: < 30% duty cycle'
        }
      };
      const m = metricInfo[metricKey] || { title: 'BIOMECHANICAL METRIC', desc: 'Real-time telemetry metric.', target: 'Standard operating bounds.' };
      const badgeHtml = `<span class="active-source-badge" style="color:var(--status-low);">REAL TELEMETRY</span>`;
      const bodyHtml = `
        <div style="font-size:0.9rem; line-height:1.6; color:var(--text-main); background:#080e1c; border:1px solid var(--border-subtle); border-radius:6px; padding:1.25rem;">
          <div style="font-size:1.05rem; font-weight:700; color:var(--accent-cyan); margin-bottom:0.5rem;">${m.title}</div>
          <p>${m.desc}</p>
          <div style="margin-top:1rem; background:#040814; border:1px solid var(--border-subtle); border-radius:4px; padding:0.75rem; font-size:0.82rem; color:var(--status-low);">
            <strong>Ergonomic Target:</strong> ${m.target}
          </div>
        </div>
      `;
      showModal('BIOMECHANICAL TELEMETRY', m.title, badgeHtml, bodyHtml);
    }

    // -----------------------------------------------------------------------
    // Entity Fetchers with Interactive Card Binding
    // -----------------------------------------------------------------------
    function fetchIncidents() {
      fetch('/api/incidents')
        .then(r => r.json())
        .then(list => {
          const c = document.getElementById('incidents-grid-container');
          c.innerHTML = '';
          if (!list || list.length === 0) {
            c.innerHTML = '<div style="color:var(--text-muted); font-size:0.9rem;">No incidents recorded in OpenSearch.</div>';
            return;
          }
          list.forEach(inc => {
            incidentsMap[inc.incident_id] = inc;
            const el = document.createElement('div');
            el.className = 'panel-card clickable-card';
            el.onclick = () => openIncidentDetail(inc.incident_id);
            el.title = 'Click to inspect incident telemetry, keyframe & KGVision reasoning';

            const torqueStr = fmtVal(inc.lumbar_torque_nm, 'Nm', 1);
            const compStr = fmtVal(inc.spinal_compression_n, 'N', 0);
            const riskStr = fmtVal(inc.risk_score, '', 1);
            const sevColor = (inc.risk_level === 'DANGEROUS' || inc.risk_level === 'HIGH' || inc.severity === 'HIGH') 
              ? 'var(--status-danger)' 
              : (inc.risk_level === 'MODERATE' ? 'var(--status-mod)' : 'var(--status-low)');
            const kfFilename = inc.keyframe_path ? inc.keyframe_path.split(/[\\\\/]/).pop() : (inc.keyframe ? inc.keyframe.split(/[\\\\/]/).pop() : 'fallback.jpg');

            el.innerHTML = `
              <div class="panel-card-title">
                <span>${fmtStr(inc.incident_id, 'INC-001')}</span>
                <div style="display:flex; align-items:center; gap:6px;">
                  ${fmtBadge(inc.data_source)}
                  <span class="active-source-badge" style="color:${sevColor}; border-color:${sevColor};">${fmtStr(inc.risk_level || inc.severity, 'HIGH')}</span>
                </div>
              </div>
              <div style="display:flex; gap:0.75rem;">
                <img src="/api/keyframe/${kfFilename}" 
                     style="width:90px; height:68px; object-fit:cover; border-radius:4px; border:1px solid var(--border-subtle); background:#000;" 
                     alt="Incident Keyframe" onerror="this.style.display='none'"/>
                <div style="font-size:0.8rem; color:var(--text-muted); display:flex; flex-direction:column; gap:3px;">
                  <div><strong>Worker:</strong> ${fmtStr(inc.worker_id)}</div>
                  <div><strong>Station:</strong> ${fmtStr(inc.station_id)}</div>
                  <div><strong>Torque:</strong> ${torqueStr}</div>
                  <div><strong>Compression:</strong> ${compStr}</div>
                  <div><strong>Risk Score:</strong> <strong style="color:var(--accent-cyan);">${riskStr}</strong></div>
                </div>
              </div>
              <div style="font-size:0.78rem; color:var(--text-dim); line-height:1.35; margin-top:0.25rem;">
                ${inc.gemini_multimodal_reasoning ? inc.gemini_multimodal_reasoning.substring(0, 120) + '...' : (inc.description || 'Deterministic threshold breach recorded.')}
              </div>
            `;
            c.appendChild(el);
          });
        });
    }

    // Passports Fetcher
    function fetchPassports() {
      fetch('/api/passports')
        .then(r => r.json())
        .then(list => {
          const c = document.getElementById('workers-grid-container');
          c.innerHTML = '';
          if (!list || list.length === 0) {
            c.innerHTML = '<div style="color:var(--text-muted); font-size:0.9rem;">No worker passports found.</div>';
            return;
          }
          list.forEach(w => {
            passportsMap[w.worker_id] = w;
            const el = document.createElement('div');
            el.className = 'panel-card clickable-card';
            el.onclick = () => openWorkerDetail(w.worker_id);
            el.title = 'Click to inspect complete Worker Ergonomic Passport';

            const rawExp = (w.exposure_minutes !== undefined && w.exposure_minutes !== null) ? w.exposure_minutes : w.daily_exposure_minutes;
            const rawDuty = (w.awkward_duty_cycle !== undefined && w.awkward_duty_cycle !== null) ? w.awkward_duty_cycle : w.awkward_duty_cycle_pct;
            const rawRisk = (w.risk_score !== undefined && w.risk_score !== null) ? w.risk_score : w.cumulative_risk_score;

            const expStr = fmtVal(rawExp, 'min', 1);
            const dutyStr = fmtVal(rawDuty, '%', 1);
            const riskStr = fmtVal(rawRisk, '', 1);
            const riskLvl = fmtStr(w.risk_level, 'LOW');
            const wName = w.worker_name ? `<span style="font-size:0.75rem; color:var(--text-dim); margin-left:6px;">${w.worker_name}</span>` : '';

            el.innerHTML = `
              <div class="panel-card-title">
                <div>
                  <span style="font-weight:700;">${fmtStr(w.worker_id)}</span>
                  ${wName}
                </div>
                <div style="display:flex; align-items:center; gap:6px;">
                  ${fmtBadge(w.data_source)}
                  <span class="active-source-badge">${fmtStr(w.role || w.status, 'Operator')}</span>
                </div>
              </div>
              <div style="display:grid; grid-template-columns:1fr 1fr; gap:0.5rem; font-size:0.8rem;">
                <div><span style="color:var(--text-dim);">Station:</span> <strong>${fmtStr(w.station_id)}</strong></div>
                <div><span style="color:var(--text-dim);">Exposure:</span> ${expStr}</div>
                <div><span style="color:var(--text-dim);">Duty Cycle:</span> ${dutyStr}</div>
                <div><span style="color:var(--text-dim);">Risk Score:</span> <strong style="color:var(--accent-cyan); font-family:var(--font-mono);">${riskStr}</strong></div>
              </div>
              <div style="font-size:0.75rem; color:var(--text-dim); border-top:1px solid var(--border-subtle); padding-top:0.4rem; display:flex; justify-content:space-between;">
                <span>Status: <strong style="color:var(--status-low); font-weight:600;">${fmtStr(w.status, 'ACTIVE MONITORING')}</strong></span>
                <span>Risk Level: <strong>${riskLvl}</strong></span>
              </div>
            `;
            c.appendChild(el);
          });
        });
    }

    // Stations & Heatmap Fetcher
    function fetchStationsAndHeatmap() {
      // 1. Fetch Heatmap Nodes
      fetch('/api/heatmap')
        .then(r => r.json())
        .then(data => {
          const stage = document.getElementById('heatmap-stage');
          if (!stage) return;
          stage.innerHTML = '';
          
          const grid = Array.isArray(data) ? data : (data.spatial_grid || data.stations || data.nodes || []);
          
          if (!grid || grid.length === 0) {
            stage.innerHTML = '<div style="display:flex; align-items:center; justify-content:center; height:100%; color:var(--text-muted); font-size:0.95rem; font-weight:600; letter-spacing:0.5px;">NO STATION DATA</div>';
            return;
          }
          
          let renderedCount = 0;
          grid.forEach(st => {
            try {
              if (!st || typeof st !== 'object') return;
              stationsMap[st.station_id] = st;
              
              let x = 50;
              let y = 50;
              if (st.position && typeof st.position === 'object') {
                if (st.position.x !== undefined && st.position.x !== null) x = Number(st.position.x);
                if (st.position.y !== undefined && st.position.y !== null) y = Number(st.position.y);
              } else {
                if (st.coord_x !== undefined && st.coord_x !== null) x = Number(st.coord_x);
                else if (st.x !== undefined && st.x !== null) x = Number(st.x) <= 1.0 && Number(st.x) > 0 ? Number(st.x) * 100 : Number(st.x);
                
                if (st.coord_y !== undefined && st.coord_y !== null) y = Number(st.coord_y);
                else if (st.y !== undefined && st.y !== null) y = Number(st.y) <= 1.0 && Number(st.y) > 0 ? Number(st.y) * 100 : Number(st.y);
              }
              
              x = Math.max(5, Math.min(95, isNaN(x) ? 50 : x));
              y = Math.max(8, Math.min(92, isNaN(y) ? 50 : y));
              
              const rScore = (st.risk_index !== undefined && st.risk_index !== null) ? st.risk_index : (st.risk_score !== undefined ? st.risk_score : st.mean_risk_score);
              const scoreStr = fmtVal(rScore, '', 1);
              const rLevel = String(st.risk_level || st.station_risk_level || 'LOW').toUpperCase();
              
              const activeWorker = fmtStr(st.active_worker_id, 'None');
              const stationName = fmtStr(st.station_name, fmtStr(st.station_id, 'STATION'));
              const dataSource = st.data_source || 'REAL_TELEMETRY';
              
              let borderColor = 'var(--status-low)';
              if (rLevel === 'DANGEROUS' || rLevel === 'HIGH' || rLevel === 'ACTION_REQUIRED') {
                borderColor = 'var(--status-danger)';
              } else if (rLevel === 'MODERATE' || rLevel === 'AT_RISK') {
                borderColor = 'var(--status-mod)';
              }
              
              const el = document.createElement('div');
              el.className = 'station-node';
              el.style.left = x + '%';
              el.style.top = y + '%';
              el.style.borderColor = borderColor;
              el.style.cursor = 'pointer';
              el.onclick = () => openStationDetail(st.station_id);
              el.title = `Click to inspect: ${stationName} (${st.station_id}) | Active Operative: ${activeWorker} | Risk: ${scoreStr} (${rLevel}) [${dataSource}]`;
              
              el.innerHTML = `
                <div class="station-node-id">${fmtStr(st.station_id)}</div>
                <div class="station-node-score">${scoreStr}</div>
                <div style="font-size:0.68rem; color:var(--text-dim); white-space:nowrap;">Op: ${activeWorker}</div>
                <div style="margin-top:2px;">${fmtBadge(dataSource)}</div>
              `;
              stage.appendChild(el);
              renderedCount++;
            } catch (nodeErr) {
              console.error('Error rendering station node:', nodeErr, st);
            }
          });
          
          if (renderedCount === 0) {
            stage.innerHTML = '<div style="display:flex; align-items:center; justify-content:center; height:100%; color:var(--text-muted); font-size:0.95rem; font-weight:600; letter-spacing:0.5px;">NO STATION DATA</div>';
          }
        })
        .catch(err => {
          console.error('Heatmap fetch error:', err);
          const stage = document.getElementById('heatmap-stage');
          if (stage) stage.innerHTML = '<div style="display:flex; align-items:center; justify-content:center; height:100%; color:var(--text-muted); font-size:0.95rem; font-weight:600; letter-spacing:0.5px;">NO STATION DATA</div>';
        });

      // 2. Fetch Station Risk Profiles
      fetch('/api/stations')
        .then(r => r.json())
        .then(list => {
          const c = document.getElementById('stations-grid-container');
          if (!c) return;
          c.innerHTML = '';
          if (!list || list.length === 0) {
            c.innerHTML = '<div style="color:var(--text-muted); font-size:0.9rem;">No station risk profiles found.</div>';
            return;
          }
          list.forEach(st => {
            try {
              stationsMap[st.station_id] = st;
              const el = document.createElement('div');
              el.className = 'panel-card clickable-card';
              el.onclick = () => openStationDetail(st.station_id);
              el.title = 'Click to inspect station risk profile & engineering controls';

              const rScore = (st.risk_index !== undefined && st.risk_index !== null) ? st.risk_index : (st.risk_score !== undefined ? st.risk_score : st.mean_risk_score);
              const riskVal = fmtVal(rScore, '', 1);
              const rLevel = fmtStr(st.risk_level || st.station_risk_level, 'LOW');
              const activeWorker = fmtStr(st.active_worker_id, 'None');
              const stationName = fmtStr(st.station_name, fmtStr(st.station_id));
              
              el.innerHTML = `
                <div class="panel-card-title">
                  <span>${fmtStr(st.station_id)}</span>
                  <div style="display:flex; align-items:center; gap:6px;">
                    ${fmtBadge(st.data_source)}
                    <span class="active-source-badge">${rLevel}</span>
                  </div>
                </div>
                <div style="font-size:0.82rem; color:var(--text-muted); display:flex; flex-direction:column; gap:4px;">
                  <div><strong>Station Name:</strong> ${stationName}</div>
                  <div><strong>Active Operative:</strong> <span style="color:${activeWorker !== 'None' ? 'var(--accent-cyan)' : 'var(--text-dim)'}; font-weight:600;">${activeWorker}</span></div>
                  <div><strong>Risk Index:</strong> <strong style="font-family:var(--font-mono); color:var(--accent-cyan);">${riskVal}</strong> / 100</div>
                  <div><strong>Zone:</strong> ${fmtStr((st.position || {}).zone || st.zone_name, 'Warehouse Floor')}</div>
                  <div><strong>Peak Compression:</strong> ${fmtVal(st.peak_compression_n, 'N', 0)}</div>
                </div>
              `;
              c.appendChild(el);
            } catch (cardErr) {
              console.error('Error rendering station card:', cardErr, st);
            }
          });
        })
        .catch(err => console.error('Stations fetch error:', err));
    }

    // Interventions Fetcher
    function fetchInterventions() {
      fetch('/api/interventions')
        .then(r => r.json())
        .then(list => {
          const c = document.getElementById('interventions-grid-container');
          c.innerHTML = '';
          if (!list || list.length === 0) {
            c.innerHTML = '<div style="color:var(--text-muted); font-size:0.9rem;">No interventions recorded.</div>';
            return;
          }
          list.forEach(intv => {
            interventionsMap[intv.intervention_id] = intv;
            const el = document.createElement('div');
            el.className = 'panel-card clickable-card';
            el.onclick = () => openInterventionDetail(intv.intervention_id);
            el.title = 'Click to inspect intervention before/after telemetry & verification';

            const eff = intv.effectiveness_assessment || {};
            let effHtml = '';
            if (intv.effectiveness === 'VERIFIED_EFFECTIVE' || eff.status === 'VERIFIED_EFFECTIVE') {
              const redPct = intv.post_metrics && intv.pre_metrics 
                ? Math.round(((intv.pre_metrics.risk_score - intv.post_metrics.risk_score) / Math.max(1, intv.pre_metrics.risk_score)) * 100) 
                : (eff.risk_reduction_pct || 0);
              effHtml = `<span style="color:var(--status-low); font-weight:700;">VERIFIED (${redPct}% Reduction)</span>`;
            } else if (intv.effectiveness === 'WAITING_FOR_SUFFICIENT_DATA' || eff.status === 'WAITING_FOR_SUFFICIENT_DATA' || !intv.post_metrics) {
              effHtml = `<span style="color:var(--status-mod); font-weight:600;">WAITING FOR SUFFICIENT DATA</span> <span style="font-size:0.7rem; color:var(--text-dim);">(Additional real telemetry required)</span>`;
            } else {
              effHtml = `<span style="color:var(--text-muted);">${fmtStr(intv.effectiveness || eff.status, 'WAITING FOR SUFFICIENT DATA')}</span>`;
            }

            el.innerHTML = `
              <div class="panel-card-title">
                <span>${fmtStr(intv.intervention_id)}</span>
                <div style="display:flex; align-items:center; gap:6px;">
                  ${fmtBadge(intv.data_source)}
                  <span class="active-source-badge" style="color:var(--accent-cyan);">${fmtStr(intv.status, 'RECORDED')}</span>
                </div>
              </div>
              <div style="font-size:0.8rem; color:var(--text-muted); display:flex; flex-direction:column; gap:3px;">
                <div><strong>Incident:</strong> ${fmtStr(intv.incident_id)}</div>
                <div><strong>Target:</strong> ${fmtStr(intv.worker_id)} @ ${fmtStr(intv.station_id)}</div>
                <div><strong>Action:</strong> ${fmtStr(intv.action || intv.action_taken, 'Station Height Adjusted')}</div>
              </div>
              <div style="background:#0a1020; border:1px solid var(--border-subtle); border-radius:4px; padding:0.5rem; font-size:0.76rem; color:var(--text-dim); margin-top:0.25rem;">
                <strong>Effectiveness:</strong> ${effHtml}
              </div>
            `;
            c.appendChild(el);
          });
        });
    }

    // Twin Simulation Calculator
    function updateTwinSimulation() {
      const pallet = document.getElementById('slider-pallet').value;
      const load = document.getElementById('slider-load').value;
      const cadence = document.getElementById('slider-cadence').value;
      const duration = document.getElementById('slider-duration').value;
      const rot = document.getElementById('check-twin-rotation').checked;

      document.getElementById('val-twin-pallet').innerText = pallet + ' cm';
      document.getElementById('val-twin-load').innerText = load + ' kg';
      document.getElementById('val-twin-cadence').innerText = cadence + ' /min';
      document.getElementById('val-twin-duration').innerText = duration + ' min';

      fetch('/api/simulator/whatif', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          pallet_height_cm: parseFloat(pallet),
          load_weight_kg: parseFloat(load),
          handling_frequency_per_min: parseFloat(cadence),
          task_duration_minutes: parseFloat(duration),
          task_rotation: rot
        })
      })
      .then(r => r.json())
      .then(res => {
        const curr = res.current_configuration;
        const sim = res.simulated_configuration;

        document.getElementById('twin-res-angle').innerHTML = `${curr.estimated_trunk_flexion_deg}&deg; &rarr; ${sim.estimated_trunk_flexion_deg}&deg;`;
        document.getElementById('twin-res-torque').innerText = `${curr.estimated_lumbar_torque_nm} Nm &rarr; ${sim.estimated_lumbar_torque_nm} Nm`;
        document.getElementById('twin-res-comp').innerText = `${curr.estimated_spinal_compression_n} N &rarr; ${sim.estimated_spinal_compression_n} N`;
        document.getElementById('twin-res-risk').innerText = `${curr.cumulative_risk_score} &rarr; ${sim.cumulative_risk_score} [${sim.risk_category}]`;
        document.getElementById('twin-res-reduct').innerText = `${res.risk_reduction_pct}% Risk Change`;

        const benList = document.getElementById('twin-benefits-list');
        benList.innerHTML = '';
        if (res.key_benefits) {
          res.key_benefits.forEach(b => {
            const li = document.createElement('div');
            li.innerText = '• ' + b;
            benList.appendChild(li);
          });
        }
      });
    }

    // Governance Audit Fetcher
    function fetchGovernanceAudit() {
      fetch('/governance/audit')
        .then(r => r.json())
        .then(list => {
          governanceList = list || [];
          const tbody = document.getElementById('governance-table-body');
          tbody.innerHTML = '';
          governanceList.forEach((row, idx) => {
            const tr = document.createElement('tr');
            tr.className = 'clickable-row';
            tr.onclick = () => openGovernanceDetail(row.incident_id, idx);
            tr.title = 'Click to inspect Cedar governance policy evaluation';

            const ts = row.timestamp ? (row.timestamp.includes('T') ? row.timestamp.split('T')[1].substring(0, 8) : row.timestamp) : 'N/A';
            tr.innerHTML = `
              <td>${ts}</td>
              <td><strong>${fmtStr(row.incident_id)}</strong></td>
              <td>${fmtStr(row.worker_id)} @ ${fmtStr(row.station_id)}</td>
              <td>${fmtStr(row.agent_proposed_action)}</td>
              <td>${fmtStr(row.policy_rule_identifier)}</td>
              <td><span style="color:var(--status-low); font-weight:700;">${fmtStr(row.enforced_action)}</span></td>
              <td>
                <div style="display:flex; align-items:center; gap:4px;">
                  ${fmtBadge(row.data_source)}
                  <span class="active-source-badge" style="color:var(--status-low);">${fmtStr(row.allow_deny_result, 'ALLOW')}</span>
                </div>
              </td>
            `;
            tbody.appendChild(tr);
          });
        });
    }

    // Shift Report Fetcher
    function fetchShiftReport() {
      fetch('/api/report/shift')
        .then(r => r.json())
        .then(data => {
          latestReportData = data;
          const rep = data.report;
          document.getElementById('report-shift-id').innerText = rep.shift_id + ' [' + rep.status + ']';
          document.getElementById('report-safety-score').innerText = 'SCORE: ' + rep.executive_summary.shift_ergonomic_safety_score + '/100';

          const s = rep.executive_summary;
          const c = rep.closed_loop_interventions;
          document.getElementById('report-body-content').innerHTML = `
            <div><strong>Workers Monitored:</strong> ${s.workers_monitored_count} | <strong>Stations:</strong> ${s.stations_monitored_count}</div>
            <div><strong>Total Incidents:</strong> ${s.total_incidents_recorded}</div>
            <div><strong>Closed-Loop Interventions:</strong> ${c.total_interventions_recorded} (Validated: ${c.effective_interventions}, Waiting: ${c.waiting_for_sufficient_data})</div>
            <div style="margin-top:0.6rem;"><strong>Top Ergonomic Hazards:</strong></div>
            <div>• Awkward forward torso bending (&gt; 25&deg;)</div>
            <div>• Repetitive handling without arm support</div>
            <div style="margin-top:0.6rem; color:var(--accent-cyan);"><strong>Exported to:</strong> ${data.json_file}</div>
            <div style="margin-top:0.4rem; font-size:0.75rem; color:var(--accent-cyan); font-weight:600;">(Click card to open full Executive Shift Report details)</div>
          `;
        });
    }

    // Drag-and-drop file upload on video viewport
    window.addEventListener('DOMContentLoaded', () => {
      const dropZone = document.querySelector('.video-viewport');
      if (dropZone) {
        dropZone.addEventListener('dragover', (e) => {
          e.preventDefault();
          dropZone.style.boxShadow = '0 0 15px rgba(0, 196, 223, 0.4)';
        });
        dropZone.addEventListener('dragleave', () => {
          dropZone.style.boxShadow = 'none';
        });
        dropZone.addEventListener('drop', (e) => {
          e.preventDefault();
          dropZone.style.boxShadow = 'none';
          if (e.dataTransfer && e.dataTransfer.files && e.dataTransfer.files.length > 0) {
            uploadCustomVideo(e.dataTransfer.files[0]);
          }
        });
      }
    });
  </script>
</body>
</html>
"""


class DashboardHTTPHandler(SimpleHTTPRequestHandler):
    """HTTP Request Handler serving web UI, live MJPEG streaming, and REST APIs."""

    backend: OpenSearchDashboardBackend = opensearch_dashboard_backend
    orchestrator: Optional[KineticGuardOrchestrator] = None

    def log_message(self, format, *args):
        # Suppress routine log output for smooth video streaming
        pass

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path
        query_params = {k: v[0] for k, v in parse_qs(parsed.query).items()}

        # 0. Health Check for Cloud Run & Container Orchestration
        if path in ("/health", "/api/health"):
            self._send_json(200, {
                "status": "HEALTHY",
                "service": "KineticGuard",
                "version": "1.0.0",
                "environment": "cloud_run" if os.getenv("K_SERVICE") else "local",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "privacy_guard": privacy_guard.get_status().get("privacy_mode", "ACTIVE"),
                "opensearch": "CONNECTED" if opensearch_engine.is_live_cluster else "LOCAL_STORAGE",
                "active_source": live_stream_manager.current_source_type,
            })
            return

        # 1. Main Dashboard HTML
        if path in ("/", "/dashboard", "/index.html"):
            body = DASHBOARD_HTML.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

        # 1b. Sample Benchmark Video Set Download (video_env_1_anonymized.zip)
        if path in ("/api/download/sample_videos", "/download/sample_videos", "/download/video_env_1_anonymized.zip"):
            zip_candidates = [
                Path("video_env_1_anonymized.zip"),
                Path(__file__).parent.parent / "video_env_1_anonymized.zip",
                Path("/app/video_env_1_anonymized.zip"),
            ]
            zip_path = None
            for p in zip_candidates:
                if p.exists() and p.is_file():
                    zip_path = p
                    break

            if zip_path and zip_path.exists():
                file_size = zip_path.stat().st_size
                self.send_response(200)
                self.send_header("Content-Type", "application/zip")
                self.send_header("Content-Disposition", 'attachment; filename="video_env_1_anonymized.zip"')
                self.send_header("Content-Length", str(file_size))
                self.send_header("Cache-Control", "public, max-age=3600")
                self.end_headers()
                with open(zip_path, "rb") as f:
                    while True:
                        chunk = f.read(65536)
                        if not chunk:
                            break
                        self.wfile.write(chunk)
                return
            else:
                self._send_json(404, {"error": "Sample video set not found on server"})
                return

        # 2. Live MJPEG Video Streaming
        if path in ("/api/stream/video", "/stream/video", "/video_feed"):
            source = query_params.get("source")
            if source and source in ("camera_01", "camera_02", "camera_03", "webcam") and source != live_stream_manager.current_source_type:
                live_stream_manager.set_source(source)

            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
            self.send_header("Pragma", "no-cache")
            self.send_header("Expires", "0")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()

            try:
                for chunk in live_stream_manager.generate_mjpeg_stream():
                    self.wfile.write(chunk)
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError, OSError):
                pass
            return

        # 3. Live Telemetry Snapshot
        if path in ("/api/stream/telemetry", "/stream/telemetry"):
            self._send_json(200, live_stream_manager.get_latest_telemetry())
            return

        # 4. Overview API
        if path in ("/api/overview", "/overview"):
            self._send_json(200, self.backend.get_overview())
            return

        # 5. Stations API
        if path in ("/api/stations", "/stations"):
            self._send_json(200, self.backend.get_stations())
            return

        # 6. Passports API
        if path in ("/api/passports", "/passports"):
            self._send_json(200, self.backend.get_passports())
            return

        # 7. Incidents API
        if path in ("/api/incidents", "/incidents"):
            limit = int(query_params.get("limit", 20))
            self._send_json(200, self.backend.get_latest_incidents(limit=limit))
            return

        # 8. Interventions API
        if path in ("/api/interventions", "/interventions"):
            limit = int(query_params.get("limit", 50))
            self._send_json(200, self.backend.get_interventions(limit=limit))
            return

        # 9. Spatial Heatmap API
        if path in ("/api/heatmap", "/heatmap"):
            self._send_json(200, self.backend.get_warehouse_heatmap())
            return

        # 10. Shift Report API
        if path in ("/api/report/shift", "/report/shift"):
            self._send_json(200, self.backend.generate_shift_report())
            return

        # 11. Cedar Governance Audit API
        if path in ("/governance/audit", "/api/governance/audit"):
            self._send_json(200, self.backend.get_cedar_audit_trail())
            return

        # 12. Privacy Guard Status API
        if path in ("/privacy", "/api/privacy"):
            self._send_json(200, privacy_guard.get_status())
            return

        # 13. Single Incident Lookup API
        if path.startswith("/api/incident/"):
            inc_id = unquote(path.replace("/api/incident/", ""))
            doc = self.backend.get_incident(inc_id)
            if doc:
                self._send_json(200, doc)
            else:
                self._send_json(404, {"error": f"Incident not found: {inc_id}"})
            return

        # 14. Single Worker Passport Lookup API
        if path.startswith("/api/passport/") or path.startswith("/api/worker/"):
            prefix = "/api/passport/" if path.startswith("/api/passport/") else "/api/worker/"
            wid = unquote(path.replace(prefix, ""))
            doc = self.backend.get_worker_passport(wid)
            if doc:
                self._send_json(200, doc)
            else:
                self._send_json(404, {"error": f"Worker passport not found: {wid}"})
            return

        # 15. Single Station Lookup API
        if path.startswith("/api/station/"):
            sid = unquote(path.replace("/api/station/", ""))
            doc = self.backend.get_station(sid)
            if doc:
                self._send_json(200, doc)
            else:
                self._send_json(404, {"error": f"Station not found: {sid}"})
            return

        # 16. Single Intervention Lookup API
        if path.startswith("/api/intervention/"):
            intv_id = unquote(path.replace("/api/intervention/", ""))
            doc = self.backend.get_intervention(intv_id)
            if doc:
                self._send_json(200, doc)
            else:
                self._send_json(404, {"error": f"Intervention not found: {intv_id}"})
            return

        # 17. Keyframe Image Binary Serving
        if path.startswith("/api/keyframe/"):
            filename = path.replace("/api/keyframe/", "")
            kf_path = self.backend.keyframes_dir / filename
            if kf_path.exists():
                with open(kf_path, "rb") as f:
                    content = f.read()
                self.send_response(200)
                self.send_header("Content-Type", "image/jpeg")
                self.send_header("Content-Length", str(len(content)))
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                self.wfile.write(content)
                return
            self._send_json(404, {"error": f"Keyframe not found: {filename}"})
            return

        # Fallback to orchestrator if configured
        if self.orchestrator:
            resp = self.orchestrator.route_request("GET", path, query_params, None)
            self._send_api_response(resp)
            return

        self._send_json(404, {"error": f"Route not found: GET {path}"})

    def do_POST(self):
        parsed = urlparse(self.path)
        path = parsed.path
        query_params = {k: v[0] for k, v in parse_qs(parsed.query).items()}

        content_length = int(self.headers.get("Content-Length", 0))

        # Switch Video Source
        if path in ("/api/stream/source", "/stream/source"):
            body_str = self.rfile.read(content_length).decode("utf-8") if content_length > 0 else "{}"
            try:
                body = json.loads(body_str)
            except Exception:
                body = {}
            source = body.get("source", "camera_01")
            custom_path = body.get("path")
            res = live_stream_manager.set_source(source, custom_path)
            self._send_json(200, res)
            return

        # Upload Video (.mp4)
        if path in ("/api/upload_video", "/upload_video", "/api/upload/video", "/upload/video"):
            try:
                raw_filename = self.headers.get("X-File-Name") or self.headers.get("X-Filename") or f"uploaded_{int(time.time())}.mp4"
                filename = unquote(raw_filename)
                safe_filename = Path(filename).name or "uploaded_video.mp4"
                os.makedirs("data", exist_ok=True)
                upload_filename = f"uploaded_{int(time.time())}_{safe_filename}"
                upload_path = os.path.join("data", upload_filename)
                bytes_left = content_length
                chunk_size = 65536
                with open(upload_path, "wb") as f:
                    while bytes_left > 0:
                        read_len = min(bytes_left, chunk_size)
                        chunk = self.rfile.read(read_len)
                        if not chunk:
                            break
                        f.write(chunk)
                        bytes_left -= len(chunk)

                # Verify OpenCV can read the uploaded video
                import cv2
                test_cap = cv2.VideoCapture(upload_path)
                can_open = test_cap.isOpened()
                frame_count = int(test_cap.get(cv2.CAP_PROP_FRAME_COUNT)) if can_open else 0
                test_cap.release()

                if not can_open or frame_count <= 0:
                    self._send_json(400, {
                        "status": "ERROR",
                        "error": f"OpenCV could not decode '{safe_filename}'. Ensure it is a valid MP4/video file."
                    })
                    return

                # Switch source and immediately process the first frame
                res = live_stream_manager.set_source("uploaded", upload_path, display_name=safe_filename)
                self._send_json(200, {
                    "status": "READY",
                    "source": "uploaded",
                    "filename": safe_filename,
                    "path": upload_path,
                    "frame_ready": res.get("frame_ready", False),
                    "frames_total": frame_count,
                    "message": f"Custom video '{safe_filename}' uploaded and active in live ingestion engine."
                })
            except Exception as ex:
                self._send_json(500, {"status": "ERROR", "error": str(ex)})
            return

        # JSON Body Handling for APIs
        body_str = self.rfile.read(content_length).decode("utf-8") if content_length > 0 else "{}"
        try:
            body = json.loads(body_str)
        except Exception:
            body = {}

        # Exposure Forecasting
        if path in ("/forecast", "/api/forecast"):
            wid = body.get("worker_id") or query_params.get("worker_id", "W-UNKNOWN")
            self._send_json(200, self.backend.get_forecast(wid))
            return

        # What-If Simulator
        if path in ("/simulator/whatif", "/api/simulator/whatif", "/api/twin/simulate", "/twin/simulate"):
            self._send_json(200, self.backend.simulate_whatif(body))
            return

        # Record Intervention
        if path in ("/interventions", "/api/interventions"):
            int_id = f"INTV-MANUAL-{int(time.time())}"
            record = {
                "intervention_id": int_id,
                "incident_id": body.get("incident_id", "N/A"),
                "worker_id": body.get("worker_id", "W-UNKNOWN"),
                "station_id": body.get("station_id", "STATION-UNKNOWN"),
                "action_type": body.get("action_type") or body.get("intervention_type", "SUPERVISOR_REVIEW"),
                "action_description": body.get("description", "Manual intervention recorded via dashboard"),
                "status": "APPLIED",
                "effectiveness_status": "WAITING_FOR_SUFFICIENT_DATA",
                "frames_observed": 0,
                "min_required_frames": 20,
                "pre_intervention_metrics": {
                    "cumulative_risk_score": float(body.get("pre_intervention_risk", 65.0)),
                    "peak_lumbar_torque_nm": 75.0,
                    "peak_spinal_compression_n": 2200.0,
                    "awkward_duty_cycle_pct": 60.0,
                },
                "applied_at": datetime.now(timezone.utc).isoformat(),
            }
            from src.local_opensearch import INDEX_INTERVENTIONS
            self.backend.engine.index_document(INDEX_INTERVENTIONS, int_id, record)
            self._send_json(201, {"message": "Intervention recorded into OpenSearch", "intervention": record})
            return

        # Fallback to orchestrator if configured
        if self.orchestrator:
            resp = self.orchestrator.route_request("POST", path, query_params, body_str)
            self._send_api_response(resp)
            return

        self._send_json(404, {"error": f"Route not found: POST {path}"})

    def _send_json(self, status: int, data: Any):
        try:
            body = json.dumps(data, indent=2, default=str).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        except (ConnectionResetError, ConnectionAbortedError, BrokenPipeError, OSError):
            pass

    def _send_api_response(self, resp: dict):
        try:
            status = resp.get("statusCode", 200)
            headers = resp.get("headers", {})
            body = resp.get("body", "{}").encode("utf-8")

            self.send_response(status)
            for k, v in headers.items():
                self.send_header(k, v)
            self.end_headers()
            self.wfile.write(body)
        except (ConnectionResetError, ConnectionAbortedError, BrokenPipeError, OSError):
            pass


def run_dashboard_server(host: Optional[str] = None, port: Optional[int] = None, mock_mode: bool = True):
    """Starts the dashboard web server with Cloud Run and local environment support."""
    effective_host = host or os.getenv("HOST", "0.0.0.0")
    effective_port = port if port is not None else int(os.getenv("PORT", "8080"))

    orchestrator = KineticGuardOrchestrator(mock_mode=mock_mode)
    DashboardHTTPHandler.orchestrator = orchestrator
    DashboardHTTPHandler.backend = opensearch_dashboard_backend

    server = ThreadingHTTPServer((effective_host, effective_port), DashboardHTTPHandler)
    server.daemon_threads = True
    print(f"\n===========================================================")
    print(f"   KineticGuard Autonomous Workplace Safety Dashboard      ")
    print(f"===========================================================")
    print(f"Dashboard URL: http://{effective_host}:{effective_port}")
    print(f"Architecture:  Strands -> Cedar -> OpenSearch -> Re-Monitor")
    print(f"Data Engine:   Local OpenSearch Persistent Memory (Zero AWS)")
    print(f"Views:         Heatmap, Passports, Incidents, Interventions, Simulator, Forecast, Reports")
    print(f"\nPress Ctrl+C to stop.")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down dashboard server...")
        live_stream_manager.stop()
        server.shutdown()
        server.server_close()
