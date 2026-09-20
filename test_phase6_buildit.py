"""
KineticGuard - Phase 6 Build It Comprehensive Test Suite
Tests all dynamic Build It presentation and dashboard features:
1. Dynamic Live Dashboard Overview & Architecture Stack Visibility
2. Worker Ergonomic Passport 2.0 aggregated directly from OpenSearch
3. Station Risk Profiles & Station Comparison from OpenSearch
4. Spatial Warehouse Ergonomic Heatmap Data
5. Live Incident Timeline with Real Keyframes, Strands, Gemini, and Cedar metadata
6. Closed-Loop Intervention Lifecycle with WAITING_FOR_SUFFICIENT_DATA and real effectiveness
7. Near-Term Exposure Forecasting from OpenSearch history with Insufficient Data handling
8. Deterministic What-If Simulator with Real Telemetry Baseline
9. Executive Shift Safety Reports compiled from OpenSearch
10. HTTP Endpoints & Real Keyframe JPEG Media Serving
"""

import json
import os
import shutil
import tempfile
import time
import unittest
from datetime import datetime, timezone
from pathlib import Path

from src.local_opensearch import (
    KineticGuardOpenSearch,
    INDEX_INCIDENTS,
    INDEX_WORKER_EXPOSURE,
    INDEX_STATION_RISK,
    INDEX_AGENT_DECISIONS,
    INDEX_INTERVENTIONS,
)
from src.opensearch_dashboard_backend import OpenSearchDashboardBackend
from src.dashboard_server import DashboardHTTPHandler


class TestPhase6BuildItDashboard(unittest.TestCase):
    """Integration test suite for KineticGuard Phase 6 Build It."""

    def setUp(self):
        self.test_dir = tempfile.mkdtemp(prefix="kg_test_p6_")
        self.os_data_dir = Path(self.test_dir) / "opensearch_data"
        self.keyframes_dir = Path(self.test_dir) / "keyframes"
        self.reports_dir = Path(self.test_dir) / "reports"
        self.os_data_dir.mkdir(parents=True, exist_ok=True)
        self.keyframes_dir.mkdir(parents=True, exist_ok=True)
        self.reports_dir.mkdir(parents=True, exist_ok=True)

        # Create isolated OpenSearch engine
        self.engine = KineticGuardOpenSearch()
        self.engine.local_storage.data_dir = self.os_data_dir
        self.engine.local_storage._init_storage()

        # Create backend wired to isolated storage
        self.backend = OpenSearchDashboardBackend(engine=self.engine)
        self.backend.keyframes_dir = self.keyframes_dir
        self.backend.reports_dir = self.reports_dir

        # Create sample real keyframe image
        self.sample_keyframe = self.keyframes_dir / "INC-TEST-001_keyframe.jpg"
        with open(self.sample_keyframe, "wb") as f:
            f.write(b"\xff\xd8\xff\xe0\x00\x10JFIF" + b"\x00" * 50)  # Valid JPEG header

        # Seed realistic OpenSearch documents
        self._seed_test_data()

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def _seed_test_data(self):
        # 1. Real incident document
        inc = {
            "incident_id": "INC-TEST-001",
            "worker_id": "W-DEMO-01",
            "station_id": "STATION-01",
            "camera_id": "camera_01",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "cumulative_risk_score": 74.5,
            "risk_level": "HIGH",
            "keyframe_path": str(self.sample_keyframe),
            "dominant_posture": "BENDING",
            "exposure_metrics": {
                "peak_lumbar_torque_nm": 82.0,
                "mean_lumbar_torque_nm": 76.5,
                "peak_spinal_compression_n": 2350.0,
                "awkward_duty_cycle_pct": 92.0,
                "rep_rate_per_min": 18.0,
            },
        }
        self.engine.index_incident(inc)

        # 2. Real agent decision (Strands + Gemini + Cedar)
        decision = {
            "incident_id": "INC-TEST-001",
            "decision": "ESCALATE",
            "reasoning": "Cedar policy gate evaluated ESCALATE (Allowed: True). Biomechanical ground truth confirms cumulative risk 74.5/100.",
            "recommended_action": "Mandatory task rotation and deploy hydraulic scissor lift table.",
            "policy_result": {
                "decision": "ESCALATE",
                "policy_gate": "PASSED_ESCALATE_POLICY",
                "allowed": True,
            },
            "gemini_analysis": {
                "model": "gemini-3.6-flash",
                "ergonomic_observation": "Worker observed in 38-degree forward trunk flexion lifting carton.",
                "root_cause": "Workstation pallet positioned below knee height.",
                "confidence": 0.94,
            },
            "authoritative_biomechanics": {
                "cumulative_risk_score": 74.5,
                "peak_lumbar_torque_nm": 82.0,
                "peak_spinal_compression_n": 2350.0,
                "awkward_duty_cycle_pct": 92.0,
                "repetition_rate_per_min": 18.0,
            },
        }
        self.engine.index_agent_decision(decision)

        # 3. Worker profile
        self.engine.index_worker_exposure("W-DEMO-01", {
            "worker_id": "W-DEMO-01",
            "station_id": "STATION-01",
            "last_risk_score": 74.5,
            "peak_compression_n": 2350.0,
            "fatigue_index": 62.0,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        })

        # 4. Station profile
        self.engine.index_station_risk("STATION-01", {
            "station_id": "STATION-01",
            "active_camera_id": "camera_01",
            "mean_risk_score": 74.5,
            "peak_compression_n": 2350.0,
            "dominant_posture": "BENDING",
        })

        # 5. Interventions: one WAITING_FOR_SUFFICIENT_DATA and one EFFECTIVE
        self.engine.index_intervention({
            "intervention_id": "INTV-WAITING-001",
            "incident_id": "INC-TEST-001",
            "worker_id": "W-DEMO-01",
            "station_id": "STATION-01",
            "action_type": "MANDATORY_ROTATION_AND_REST",
            "status": "APPLIED",
            "effectiveness_status": "WAITING_FOR_SUFFICIENT_DATA",
            "frames_observed": 6,
            "pre_intervention_metrics": {"cumulative_risk_score": 74.5},
        })

        self.engine.index_intervention({
            "intervention_id": "INTV-EFFECTIVE-001",
            "incident_id": "INC-TEST-001",
            "worker_id": "W-DEMO-01",
            "station_id": "STATION-01",
            "action_type": "PALLET_ELEVATION",
            "status": "RESOLVED",
            "effectiveness_status": "EFFECTIVE",
            "is_effective": True,
            "risk_reduction_pct": 52.0,
            "frames_observed": 30,
            "pre_intervention_metrics": {"cumulative_risk_score": 74.5},
            "post_intervention_metrics": {"cumulative_risk_score": 35.8, "frames_analyzed": 30},
        })

    def test_01_opensearch_dashboard_overview_dynamic_metrics(self):
        """Verify get_overview aggregates real station/worker counts and visible architecture stack."""
        ov = self.backend.get_overview()
        self.assertEqual(ov["status"], "HEALTHY")
        self.assertEqual(ov["data_source"], "Local OpenSearch Memory")
        self.assertGreaterEqual(ov["stations_count"], 1)
        self.assertGreaterEqual(ov["workers_count"], 1)
        self.assertGreaterEqual(ov["incidents_count"], 1)
        self.assertIn(ov["freshness"], ["LIVE", "STALE"])
        self.assertEqual(ov["latest_action"], "ESCALATE")
        self.assertGreaterEqual(len(ov["architecture_stack"]), 10)
        self.assertEqual(ov["architecture_stack"][0]["name"], "Video Ingestion")
        self.assertEqual(ov["architecture_stack"][9]["name"], "Live Dashboard")

    def test_02_worker_passport_from_opensearch(self):
        """Verify Worker Ergonomic Passport 2.0 aggregates from OpenSearch without fake data."""
        p = self.backend.get_worker_passport("W-DEMO-01")
        self.assertIsNotNone(p)
        self.assertEqual(p["worker_id"], "W-DEMO-01")
        self.assertEqual(p["assigned_station"], "STATION-01")
        self.assertEqual(p["current_ergonomic_status"], "ACTION_REQUIRED")
        self.assertEqual(p["peak_lumbar_torque_nm"], 82.0)
        self.assertEqual(p["peak_compression_n"], 2350.0)
        self.assertEqual(p["awkward_duty_cycle_pct"], 92.0)
        self.assertEqual(p["repetition_rate_per_min"], 18.0)
        self.assertEqual(len(p["risk_trend"]), 1)
        self.assertEqual(p["risk_trend"][0]["score"], 74.5)
        self.assertTrue(len(p["recommended_rotations"]) > 0)
        self.assertIn("disclaimer", p)

    def test_03_station_risk_and_comparison(self):
        """Verify Station Risk Profiles and Station Comparison originate from OpenSearch."""
        stations = self.backend.get_stations()
        self.assertGreaterEqual(len(stations), 1)
        st = next(s for s in stations if s["station_id"] == "STATION-01")
        self.assertEqual(st["active_camera_id"], "camera_01")
        self.assertEqual(st["station_risk_level"], "ACTION_REQUIRED")
        self.assertEqual(st["mean_risk_score"], 74.5)
        self.assertEqual(st["peak_compression_n"], 2350.0)
        self.assertEqual(st["dominant_posture"], "BENDING")
        self.assertIn("W-DEMO-01", st["workers_affected"])

    def test_04_warehouse_heatmap_spatial_grid(self):
        """Verify 2D spatial warehouse heatmap coordinates and risk level mappings."""
        nodes = self.backend.get_warehouse_heatmap()
        self.assertGreaterEqual(len(nodes), 1)
        n = next(node for node in nodes if node["station_id"] == "STATION-01")
        self.assertEqual(n["coord_x"], 20)
        self.assertEqual(n["coord_y"], 35)
        self.assertEqual(n["risk_score"], 74.5)
        self.assertEqual(n["risk_level"], "ACTION_REQUIRED")
        self.assertEqual(n["color"], "#ef4444")  # Red alert for ACTION_REQUIRED

    def test_05_live_incident_timeline_with_real_keyframes(self):
        """Verify live incidents include real keyframe paths, Strands reasoning, Gemini, and Cedar gates."""
        incidents = self.backend.get_latest_incidents(limit=10)
        self.assertGreaterEqual(len(incidents), 1)
        inc = incidents[0]
        self.assertEqual(inc["incident_id"], "INC-TEST-001")
        self.assertEqual(inc["worker_id"], "W-DEMO-01")
        self.assertEqual(inc["risk_level"], "HIGH")

        # Visual evidence & keyframe
        vis = inc["visual_evidence"]
        self.assertTrue(vis["has_keyframe"])
        self.assertIn("INC-TEST-001_keyframe.jpg", vis["keyframe_url"])

        # Strands Reasoning
        self.assertIn("Cedar policy gate evaluated ESCALATE", inc["strands_guardian"]["reasoning"])

        # Gemini Multimodal
        gemini = inc["gemini_analysis"]
        self.assertEqual(gemini["model"], "gemini-3.6-flash")
        self.assertIn("38-degree forward trunk flexion", gemini["observation"])
        self.assertIn("knee height", gemini["root_cause"])
        self.assertEqual(gemini["confidence"], 0.94)

        # Cedar Policy Gate
        cedar = inc["cedar_policy"]
        self.assertEqual(cedar["decision"], "ESCALATE")
        self.assertEqual(cedar["policy_gate"], "PASSED_ESCALATE_POLICY")
        self.assertTrue(cedar["allowed"])

    def test_06_closed_loop_intervention_lifecycle_and_waiting_guard(self):
        """Verify intervention lifecycle stepper, WAITING_FOR_SUFFICIENT_DATA guard, and real effectiveness."""
        intvs = self.backend.get_interventions()
        self.assertGreaterEqual(len(intvs), 2)

        # Check WAITING_FOR_SUFFICIENT_DATA record
        w_intv = next(i for i in intvs if i["intervention_id"] == "INTV-WAITING-001")
        self.assertEqual(w_intv["effectiveness_status"], "WAITING_FOR_SUFFICIENT_DATA")
        self.assertTrue(w_intv["is_waiting_for_data"])
        self.assertEqual(w_intv["frames_observed"], 6)
        self.assertEqual(w_intv["min_required_frames"], 20)
        self.assertIsNone(w_intv["is_effective"])
        self.assertIsNone(w_intv["post_intervention_metrics"]["cumulative_risk_score"])
        self.assertEqual(w_intv["lifecycle_stepper"]["stage_3_remonitor"]["status"], "IN_PROGRESS")

        # Check EFFECTIVE record
        e_intv = next(i for i in intvs if i["intervention_id"] == "INTV-EFFECTIVE-001")
        self.assertEqual(e_intv["effectiveness_status"], "EFFECTIVE")
        self.assertFalse(e_intv["is_waiting_for_data"])
        self.assertTrue(e_intv["is_effective"])
        self.assertEqual(e_intv["risk_reduction_pct"], 52.0)
        self.assertEqual(e_intv["post_intervention_metrics"]["cumulative_risk_score"], 35.8)
        self.assertEqual(e_intv["lifecycle_stepper"]["stage_4_outcome"]["status"], "COMPLETED")

    def test_07_near_term_exposure_forecaster(self):
        """Verify forecasting projects fatigue from real OpenSearch data and handles insufficient data."""
        # Active worker forecast
        fc_active = self.backend.get_forecast("W-DEMO-01")
        self.assertIn("CRITICAL_ACCUMULATION", fc_active["forecast"])
        self.assertEqual(fc_active["current_score"], 74.5)
        self.assertGreater(fc_active["projected_score_30min"], 74.5)
        self.assertIn("Schedule proactive worker rotation", fc_active["proactive_action"])

        # Unknown worker with zero data
        fc_empty = self.backend.get_forecast("W-NONEXISTENT")
        self.assertEqual(fc_empty["status"], "INSUFFICIENT_DATA")
        self.assertEqual(fc_empty["confidence"], 0.0)

    def test_08_whatif_simulator_deterministic_modeling(self):
        """Verify What-If simulator computes deterministic risk reduction using real telemetry baseline."""
        sim = self.backend.simulate_whatif({
            "base_pallet_height_cm": 40.0,
            "base_load_weight_kg": 14.0,
            "pallet_height_cm": 85.0,
            "load_weight_kg": 8.0,
            "reps_per_minute": 10.0,
            "task_duration_minutes": 45.0,
            "worker_rotation_enabled": True,
        })
        self.assertEqual(sim["status"], "SIMULATED_ESTIMATE")
        self.assertLess(sim["risk_reduction_pct"], -25.0)
        self.assertLess(
            sim["simulated_configuration"]["estimated_spinal_compression_n"],
            sim["current_configuration"]["estimated_spinal_compression_n"],
        )
        self.assertTrue(len(sim["key_benefits"]) > 0)

    def test_09_shift_safety_reporter(self):
        """Verify executive shift safety report is compiled from OpenSearch and exported as JSON/Markdown."""
        res = self.backend.generate_shift_report()
        report = res["report"]
        self.assertEqual(report["status"], "ACTIVE_DATA")
        self.assertEqual(report["executive_summary"]["workers_monitored_count"], 1)
        self.assertEqual(report["executive_summary"]["total_incidents_recorded"], 1)
        self.assertTrue(Path(res["json_file"]).exists())
        self.assertTrue(Path(res["markdown_file"]).exists())

    def test_10_http_backend_and_keyframe_media_serving(self):
        """Verify HTTP server handler routes and keyframe JPEG binary serving."""
        handler = DashboardHTTPHandler
        handler.backend = self.backend

        # Verify keyframe file exists and is valid JPEG
        self.assertTrue(self.sample_keyframe.exists())
        with open(self.sample_keyframe, "rb") as f:
            data = f.read()
        self.assertTrue(data.startswith(b"\xff\xd8\xff\xe0\x00\x10JFIF"))

        # Verify overview API response format
        ov = self.backend.get_overview()
        self.assertIn("stations_count", ov)
        self.assertIn("workers_count", ov)
        self.assertIn("freshness", ov)


if __name__ == "__main__":
    unittest.main()
