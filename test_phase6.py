"""
KineticGuard - Comprehensive Phase 6 Test Suite
Tests all dynamic Autonomous Safety Intelligence capabilities:
1. Worker Ergonomic Passport 2.0 Engine
2. Station Risk Intelligence Engine
3. Safety Guardian Autonomous Decision Layer
4. Closed-Loop Intervention System (Record, Re-monitor, Before vs After)
5. Near-Term Exposure Forecasting (including Insufficient Data handling)
6. Deterministic What-If Simulator
7. Executive Shift Safety Intelligence Reporter
8. REST API Endpoints & Overview Aggregator
"""

import json
import os
import shutil
import tempfile
import time
import unittest
from datetime import datetime, timezone

from src.dynamo_manager import DynamoDBErgonomicsManager
from src.safety_guardian import SafetyGuardianAgent
from src.intervention_manager import InterventionManager
from src.exposure_forecaster import ExposureForecaster
from src.whatif_simulator import WhatIfSimulator
from src.shift_reporter import ShiftReporter
from src.aws_orchestrator import KineticGuardOrchestrator


class TestPhase6AutonomousIntelligence(unittest.TestCase):
    """Unit and integration tests for Phase 6."""

    def setUp(self):
        self.test_dir = tempfile.mkdtemp(prefix="kg_test_phase6_")
        self.dynamo_dir = os.path.join(self.test_dir, "dynamodb")
        self.reports_dir = os.path.join(self.test_dir, "reports")
        os.makedirs(self.dynamo_dir, exist_ok=True)
        os.makedirs(self.reports_dir, exist_ok=True)

        self.dynamo = DynamoDBErgonomicsManager(mock_mode=True, mock_state_dir=self.dynamo_dir)
        self.guardian = SafetyGuardianAgent()
        self.interventions = InterventionManager(
            mock_mode=True,
            mock_file=os.path.join(self.dynamo_dir, "interventions.json"),
        )
        self.forecaster = ExposureForecaster()
        self.simulator = WhatIfSimulator()
        self.reporter = ShiftReporter(mock_mode=True)
        self.reporter.dynamo = self.dynamo
        self.reporter.interventions = self.interventions
        self.reporter.output_dir = os.path.join(self.reports_dir)

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_01_worker_passport_dynamic_accumulation(self):
        """Verify worker passport accumulates fatigue, torque, repetitions, and risk trend without fake data."""
        worker_id = "W-TEST-001"
        station_id = "STATION-PACK-01"

        # First incident
        inc1 = {
            "incident_id": "INC-TEST-01",
            "worker_id": worker_id,
            "station_id": station_id,
            "authoritative_metrics": {
                "cumulative_risk_score": 62.0,
                "risk_level": "HIGH",
                "peak_lumbar_torque_nm": 88.5,
                "peak_spinal_compression_n": 2450.0,
                "repetition_count": 5,
                "repetition_rate_per_min": 12.0,
                "duration_sec": 2.5,
            },
        }
        p1 = self.dynamo.update_worker_passport(inc1)
        self.assertEqual(p1["worker_id"], worker_id)
        self.assertEqual(p1["total_incidents"], 1)
        self.assertEqual(p1["high_risk_incidents"], 1)
        self.assertEqual(p1["peak_lumbar_torque_nm"], 88.5)
        self.assertEqual(len(p1["risk_trend"]), 1)
        self.assertGreater(p1["fatigue_index"], 0.0)

        # Second incident (higher strain)
        inc2 = {
            "incident_id": "INC-TEST-02",
            "worker_id": worker_id,
            "station_id": station_id,
            "authoritative_metrics": {
                "cumulative_risk_score": 78.0,
                "risk_level": "HIGH",
                "peak_lumbar_torque_nm": 95.0,
                "peak_spinal_compression_n": 2900.0,
                "repetition_count": 8,
                "repetition_rate_per_min": 15.0,
                "duration_sec": 3.0,
            },
        }
        p2 = self.dynamo.update_worker_passport(inc2)
        self.assertEqual(p2["total_incidents"], 2)
        self.assertEqual(p2["high_risk_incidents"], 2)
        self.assertEqual(p2["peak_lumbar_torque_nm"], 95.0)
        self.assertEqual(len(p2["risk_trend"]), 2)
        self.assertGreater(p2["fatigue_index"], p1["fatigue_index"])
        self.assertEqual(p2["current_ergonomic_status"], "ACTION_REQUIRED")

        # Test listing passports
        all_p = self.dynamo.list_all_passports()
        self.assertEqual(len(all_p), 1)
        self.assertEqual(all_p[0]["worker_id"], worker_id)

    def test_02_station_risk_intelligence(self):
        """Verify station intelligence aggregates affected workers and dominant risk factors."""
        st_id = "STATION-SORT-A"
        inc = {
            "incident_id": "INC-ST-01",
            "station_id": st_id,
            "camera_id": "camera_01",
            "worker_id": "W-CAM01-AA",
            "authoritative_metrics": {
                "cumulative_risk_score": 72.0,
                "peak_lumbar_torque_nm": 92.0,
                "peak_spinal_compression_n": 2800.0,
                "awkward_duty_cycle_pct": 86.0,
                "duration_sec": 2.0,
            },
            "vlm_analysis": {
                "risk_factors": ["Excessive trunk forward flexion (> 40 deg)", "Awkward lumbar moment"]
            },
        }
        st = self.dynamo.update_station_stats(inc)
        self.assertEqual(st["station_id"], st_id)
        self.assertEqual(st["total_incidents"], 1)
        self.assertIn("W-CAM01-AA", st["workers_affected"])
        self.assertEqual(st["station_risk_level"], "HIGH")
        self.assertIn("Excessive trunk forward flexion", st["dominant_risk_factor"])

    def test_03_safety_guardian_decisions(self):
        """Verify Safety Guardian makes deterministic, policy-bound safety decisions."""
        # 1. Critical threshold breach -> ESCALATE
        inc_crit = {
            "incident_id": "INC-CRIT",
            "worker_id": "W-TEST",
            "station_id": "ST-01",
            "authoritative_metrics": {
                "cumulative_risk_score": 85.0,
                "risk_level": "DANGEROUS",
                "peak_spinal_compression_n": 3650.0,  # Exceeds NIOSH 3400 N limit
                "peak_lumbar_torque_nm": 110.0,
                "awkward_duty_cycle_pct": 92.0,
                "repetition_rate_per_min": 18.0,
            },
        }
        dec_crit = self.guardian.evaluate(incident=inc_crit)
        self.assertEqual(dec_crit["decision"], "ESCALATE")
        self.assertIn("CRITICAL SAFETY THRESHOLD BREACH", dec_crit["reason"])
        self.assertIn("Halt manual picking operations", dec_crit["recommended_action"])

        # 2. Moderate risk -> NOTIFY_SUPERVISOR
        inc_mod = {
            "incident_id": "INC-MOD",
            "worker_id": "W-TEST",
            "station_id": "ST-01",
            "authoritative_metrics": {
                "cumulative_risk_score": 45.0,
                "risk_level": "MODERATE",
                "peak_spinal_compression_n": 1800.0,
                "peak_lumbar_torque_nm": 65.0,
                "awkward_duty_cycle_pct": 55.0,
                "repetition_rate_per_min": 8.0,
            },
        }
        dec_mod = self.guardian.evaluate(incident=inc_mod)
        self.assertEqual(dec_mod["decision"], "NOTIFY_SUPERVISOR")

    def test_04_closed_loop_intervention_lifecycle(self):
        """Verify recording an intervention, waiting, re-monitoring, and effectiveness classification."""
        # Record intervention
        rec = self.interventions.record_intervention(
            worker_id="W-OP-01",
            station_id="STATION-01",
            triggering_incident_id="INC-100",
            intervention_type="PALLET_ELEVATION",
            pre_intervention_risk=75.0,
            description="Elevated pallet from 45 cm to 85 cm via hydraulic lift",
        )
        int_id = rec["intervention_id"]
        self.assertIn(rec["effectiveness_status"], ["PENDING", "WAITING FOR SUFFICIENT POST-INTERVENTION DATA"])
        self.assertEqual(rec["pre_intervention_risk"], 75.0)

        # Re-monitor post-intervention (risk dropped to 42.0: -44.0% change -> EFFECTIVE)
        evaluated = self.interventions.evaluate_effectiveness(
            intervention_id=int_id,
            post_intervention_risk=42.0,
            evaluator_notes="Biomechanical torque reduced by 44%",
        )
        self.assertEqual(evaluated["effectiveness_status"], "EFFECTIVE")
        self.assertEqual(evaluated["post_intervention_risk"], 42.0)
        self.assertAlmostEqual(evaluated["risk_change_pct"], -44.0, places=1)

    def test_05_exposure_forecaster(self):
        """Verify exposure forecaster calculates near-term fatigue ramp and handles insufficient data."""
        # Insufficient data
        fc_empty = self.forecaster.forecast_exposure(
            current_score=0.0,
            duty_cycle_pct=0.0,
            repetition_rate_per_min=0.0,
            fatigue_index=0.0,
            risk_trend=[],
            worker_id="W-NONE",
        )
        self.assertEqual(fc_empty["status"], "INSUFFICIENT_DATA")

        # Active telemetry forecast
        fc_active = self.forecaster.forecast_exposure(
            current_score=72.0,
            duty_cycle_pct=88.0,
            repetition_rate_per_min=16.0,
            fatigue_index=65.0,
            risk_trend=[{"score": 60.0}, {"score": 72.0}],
            worker_id="W-ACTIVE",
            station_id="STATION-01",
        )
        self.assertIn("CRITICAL_ACCUMULATION", fc_active["forecast"])
        self.assertGreater(fc_active["projected_score_30min"], 72.0)
        self.assertIn("Schedule proactive worker rotation", fc_active["proactive_action"])

    def test_06_whatif_simulator(self):
        """Verify deterministic What-If simulator computes pallet height and load reduction benefits."""
        base = {
            "load_weight_kg": 15.0,
            "pallet_height_cm": 40.0,
            "reps_per_minute": 16.0,
            "task_duration_minutes": 60.0,
            "worker_rotation_enabled": False,
        }
        mod = {
            "load_weight_kg": 8.0,
            "pallet_height_cm": 85.0,
            "reps_per_minute": 10.0,
            "task_duration_minutes": 45.0,
            "worker_rotation_enabled": True,
        }
        sim = self.simulator.simulate(current_baseline=base, modifications=mod)
        self.assertEqual(sim["status"], "SIMULATED_ESTIMATE")
        self.assertLess(sim["risk_reduction_pct"], -30.0)
        self.assertLess(
            sim["simulated_configuration"]["estimated_spinal_compression_n"],
            sim["current_configuration"]["estimated_spinal_compression_n"],
        )
        self.assertTrue(len(sim["key_benefits"]) > 0)

    def test_07_shift_safety_reporter(self):
        """Verify shift reporter generates accurate summary and handles empty states cleanly."""
        # Empty state
        rep_empty = self.reporter.generate_report()["report"]
        self.assertEqual(rep_empty["status"], "INSUFFICIENT_DATA")
        self.assertEqual(rep_empty["executive_summary"]["workers_monitored_count"], 0)

        # Populate state
        inc = {
            "incident_id": "INC-REP-01",
            "worker_id": "W-REP-01",
            "station_id": "STATION-REP-01",
            "authoritative_metrics": {
                "cumulative_risk_score": 68.0,
                "risk_level": "HIGH",
                "peak_lumbar_torque_nm": 85.0,
                "peak_spinal_compression_n": 2400.0,
                "repetition_count": 6,
                "repetition_rate_per_min": 12.0,
                "duration_sec": 2.0,
            },
            "vlm_analysis": {"risk_factors": ["Excessive trunk forward flexion (> 40 deg)"]},
        }
        self.dynamo.save_incident(inc)
        rep_active = self.reporter.generate_report()["report"]
        self.assertEqual(rep_active["status"], "ACTIVE_DATA")
        self.assertEqual(rep_active["executive_summary"]["workers_monitored_count"], 1)
        self.assertEqual(rep_active["executive_summary"]["total_incidents_recorded"], 1)

    def test_08_orchestrator_routes_and_overview(self):
        """Verify KineticGuardOrchestrator routes all REST endpoints without errors."""
        orchestrator = KineticGuardOrchestrator(mock_mode=True)
        
        # Overview
        ov = orchestrator.route_request("GET", "/overview", {}, None)
        self.assertEqual(ov["statusCode"], 200)
        ov_data = json.loads(ov["body"])
        self.assertIn("stations_count", ov_data)
        self.assertIn("freshness", ov_data)

        # Health
        h = orchestrator.route_request("GET", "/health", {}, None)
        self.assertEqual(h["statusCode"], 200)

        # Passports and workers
        w = orchestrator.route_request("GET", "/workers", {}, None)
        self.assertEqual(w["statusCode"], 200)


if __name__ == "__main__":
    unittest.main()
