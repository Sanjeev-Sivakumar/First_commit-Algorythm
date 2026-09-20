"""
Comprehensive Normalization & Telemetry Regression Test Suite for KineticGuard
Verifies Requirements 1 through 21:
1. Canonical data source classification: REAL_TELEMETRY, TEST_FIXTURE, HISTORICAL, MODELLED, SIMULATED.
2. Missing telemetry fields return None (rendered as N/A), while genuine 0.0 remains 0.0.
3. Worker passports schema quality: never display 0 min / 0% / 0 for missing telemetry.
4. Historical worker records: W-C01-OS-HIST -> HISTORICAL / DATA INCOMPLETE.
5. Inconsistent worker records: W-ESC-01, W-LINK-77, W-LOG-01 -> TEST_FIXTURE.
6. Station active operative aggregation: STATION-01 -> W-C01-SESS, inactive -> None.
7. Station risk aggregation excludes test fixtures from live operational risk.
8. Spatial warehouse heatmap data and station risk profiles use identical canonical aggregation.
9. WAITING_FOR_SUFFICIENT_DATA lifecycle is strictly preserved.
10. Digital Twin outputs are strictly tagged MODELLED / SIMULATED.
11. AI reasoning cannot overwrite deterministic biomechanical measurements.
"""

import unittest
import json
from pathlib import Path
from src.data_normalizer import (
    safe_num,
    classify_data_source,
    normalize_station_id,
    normalize_incident,
    normalize_worker_passport,
    normalize_station_profile,
    normalize_intervention,
    DATA_SOURCE_REAL,
    DATA_SOURCE_TEST,
    DATA_SOURCE_HISTORICAL,
    DATA_SOURCE_MODELLED,
    DATA_SOURCE_SIMULATED,
)
from src.opensearch_dashboard_backend import opensearch_dashboard_backend


class TestDataNormalization(unittest.TestCase):

    def test_01_missing_metrics_do_not_become_zero(self):
        """Requirement 2: Missing biomechanical fields return None (not 0.0)."""
        doc = {
            "incident_id": "INC-TEST-001",
            "worker_id": "W-TEST-01",
            # torque, compression, risk_score completely absent
        }
        normalized = normalize_incident(doc)

        self.assertIsNone(normalized["lumbar_torque_nm"], "Missing lumbar torque must be None, not 0")
        self.assertIsNone(normalized["spinal_compression_n"], "Missing spinal compression must be None, not 0")
        self.assertIsNone(normalized["risk_score"], "Missing risk score must be None, not 0")
        self.assertIsNone(normalized["trunk_flexion_deg"], "Missing trunk flexion must be None, not 0")
        self.assertIsNone(normalized["repetition_rate"], "Missing repetition rate must be None, not 0")
        self.assertIsNone(normalized["awkward_duty_cycle"], "Missing duty cycle must be None, not 0")

    def test_02_real_zero_remains_zero(self):
        """Requirement 2: Real numeric 0.0 remains 0.0, not None."""
        self.assertEqual(safe_num(0.0), 0.0)
        self.assertEqual(safe_num(0), 0.0)
        self.assertEqual(safe_num("0.0"), 0.0)
        self.assertEqual(safe_num(0, precision=0), 0)

        doc = {
            "incident_id": "INC-ZERO-001",
            "lumbar_torque_nm": 0.0,
            "spinal_compression_n": 0,
            "risk_score": 0.0,
            "trunk_flexion_deg": 0.0,
        }
        normalized = normalize_incident(doc)
        self.assertEqual(normalized["lumbar_torque_nm"], 0.0)
        self.assertEqual(normalized["spinal_compression_n"], 0)
        self.assertEqual(normalized["risk_score"], 0.0)
        self.assertEqual(normalized["trunk_flexion_deg"], 0.0)

    def test_03_specific_incidents_fallback_zero_eliminated(self):
        """Requirement 2: Verify INC-TEST-OS-001, INC-DYN-999, INC-CRIT-001, INC-MILD-001."""
        # When torque is absent, it must be None
        for inc_id in ("INC-TEST-OS-001", "INC-DYN-999", "INC-CRIT-001", "INC-MILD-001"):
            doc = {"incident_id": inc_id}
            norm = normalize_incident(doc)
            self.assertIsNone(norm["lumbar_torque_nm"])
            self.assertEqual(norm["data_source"], DATA_SOURCE_TEST)

    def test_04_data_source_classification(self):
        """Requirement 1: Verify data_source classification for test, historical, and real records."""
        # Test fixtures
        self.assertEqual(classify_data_source({"incident_id": "INC-GOV-TEST-001"}), DATA_SOURCE_TEST)
        self.assertEqual(classify_data_source({"incident_id": "INC-TEST-PRIVACY-OS-001"}), DATA_SOURCE_TEST)
        self.assertEqual(classify_data_source({"incident_id": "INC-P5-E2E-001"}), DATA_SOURCE_TEST)
        self.assertEqual(classify_data_source({"incident_id": "INC-SCHEMA-001"}), DATA_SOURCE_TEST)
        self.assertEqual(classify_data_source({"worker_id": "W-ESC-01"}), DATA_SOURCE_TEST)
        self.assertEqual(classify_data_source({"worker_id": "W-LINK-77"}), DATA_SOURCE_TEST)
        self.assertEqual(classify_data_source({"worker_id": "W-LOG-01"}), DATA_SOURCE_TEST)

        # Historical records
        self.assertEqual(classify_data_source({"worker_id": "W-C01-OS-HIST"}), DATA_SOURCE_HISTORICAL)
        self.assertEqual(classify_data_source({"incident_id": "INC-ARCHIVE-99"}), DATA_SOURCE_HISTORICAL)

        # Real telemetry records
        self.assertEqual(classify_data_source({"incident_id": "INC-20260920-CAMERA_01-001", "camera_id": "camera_01"}), DATA_SOURCE_REAL)
        self.assertEqual(classify_data_source({"worker_id": "W-C01-SESS", "session_id": "sess_1789900905_camera_01"}), DATA_SOURCE_REAL)

    def test_05_historical_worker_passport_data_quality(self):
        """Requirement 4: Historical worker W-C01-OS-HIST marked HISTORICAL and no fabricated zeros."""
        passport = normalize_worker_passport(
            worker_id="W-C01-OS-HIST",
            profile={"worker_id": "W-C01-OS-HIST"},
            incidents=[],
            interventions=[]
        )
        self.assertEqual(passport["data_source"], DATA_SOURCE_HISTORICAL)
        self.assertEqual(passport["status"], "HISTORICAL / DATA INCOMPLETE")
        self.assertEqual(passport["risk_level"], "HISTORICAL")
        self.assertIsNone(passport["exposure_minutes"])
        self.assertIsNone(passport["awkward_duty_cycle"])
        self.assertIsNone(passport["risk_score"])

    def test_06_inconsistent_worker_records_classified_correctly(self):
        """Requirement 5: Test workers W-ESC-01, W-LINK-77, W-LOG-01 classified as TEST_FIXTURE."""
        for wid in ("W-ESC-01", "W-LINK-77", "W-LOG-01"):
            passport = normalize_worker_passport(
                worker_id=wid,
                profile={"worker_id": wid, "station_id": "STATION-01"},
                incidents=[],
                interventions=[]
            )
            self.assertEqual(passport["data_source"], DATA_SOURCE_TEST)
            self.assertEqual(passport["status"], "TEST_FIXTURE")
            self.assertIsNone(passport["exposure_minutes"])
            self.assertIsNone(passport["awkward_duty_cycle"])

    def test_07_real_worker_passports(self):
        """Requirement 3: Real workers W-C01-SESS, W-C02-SESS, W-C03-SESS."""
        p1 = normalize_worker_passport(
            worker_id="W-C01-SESS",
            profile={"worker_id": "W-C01-SESS", "station_id": "STATION-01", "data_source": DATA_SOURCE_REAL},
            incidents=[{
                "incident_id": "INC-20260920-CAMERA_01-001",
                "camera_id": "camera_01",
                "station_id": "STATION-01",
                "cumulative_risk_score": 71.3,
                "exposure_metrics": {
                    "peak_lumbar_torque_nm": 77.8,
                    "peak_spinal_compression_n": 1955.0,
                    "awkward_duty_cycle_pct": 93.2,
                }
            }],
            interventions=[]
        )
        self.assertEqual(p1["data_source"], DATA_SOURCE_REAL)
        self.assertEqual(p1["status"], "ACTIVE MONITORING")
        self.assertEqual(p1["station_id"], "STATION-01")
        self.assertEqual(p1["risk_score"], 71.3)
        self.assertEqual(p1["peak_lumbar_torque_nm"], 77.8)
        self.assertEqual(p1["awkward_duty_cycle"], 93.2)

    def test_08_station_active_operative_resolution(self):
        """Requirement 6: Station profile derives Active Operative from current REAL_TELEMETRY workers."""
        active_passports = [
            {"worker_id": "W-C01-SESS", "station_id": "STATION-01", "data_source": DATA_SOURCE_REAL},
            {"worker_id": "W-C02-SESS", "station_id": "STATION-02", "data_source": DATA_SOURCE_REAL},
            {"worker_id": "W-C03-SESS", "station_id": "STATION-03", "data_source": DATA_SOURCE_REAL},
            {"worker_id": "W-GOV-01", "station_id": "STATION-01", "data_source": DATA_SOURCE_TEST},  # test worker should NOT become active operative
        ]

        st1 = normalize_station_profile("STATION-01", {}, [], active_passports)
        self.assertEqual(st1["active_worker_id"], "W-C01-SESS")

        st2 = normalize_station_profile("STATION-02", {}, [], active_passports)
        self.assertEqual(st2["active_worker_id"], "W-C02-SESS")

        st3 = normalize_station_profile("STATION-03", {}, [], active_passports)
        self.assertEqual(st3["active_worker_id"], "W-C03-SESS")

        # Inactive station with no real workers
        st_inactive = normalize_station_profile("STATION-NORTH", {}, [], active_passports)
        self.assertIsNone(st_inactive["active_worker_id"])

    def test_09_station_risk_aggregation_excludes_test_fixtures(self):
        """Requirement 7: Live station risk aggregates strictly from REAL_TELEMETRY incidents."""
        incidents = [
            # Real telemetry incident
            {
                "incident_id": "INC-20260920-CAMERA_01-001",
                "camera_id": "camera_01",
                "station_id": "STATION-01",
                "cumulative_risk_score": 45.0,
                "data_source": DATA_SOURCE_REAL,
            },
            # Synthetic test fixture with extreme score
            {
                "incident_id": "INC-GOV-TEST-001",
                "station_id": "STATION-01",
                "cumulative_risk_score": 99.0,
                "data_source": DATA_SOURCE_TEST,
            },
        ]
        st = normalize_station_profile("STATION-01", {}, incidents, [])
        # Live risk must be 45.0 from the real incident, not distorted by 99.0 test fixture
        self.assertEqual(st["risk_index"], 45.0)
        self.assertEqual(st["data_source"], DATA_SOURCE_REAL)

    def test_10_station_canonical_schema_and_position(self):
        """Requirement 8 & 9: Station profile schema has position, station_name, active_worker_id, risk_index."""
        st = normalize_station_profile("STATION-01", {"station_name": "Ingestion Dock 01"}, [], [])
        self.assertEqual(st["station_id"], "STATION-01")
        self.assertIn("position", st)
        self.assertEqual(st["position"]["x"], 20)
        self.assertEqual(st["position"]["y"], 35)
        self.assertIn("zone", st["position"])
        self.assertEqual(st["coord_x"], 20)
        self.assertEqual(st["coord_y"], 35)

    def test_11_waiting_for_sufficient_data_preserved(self):
        """Requirement 11: WAITING_FOR_SUFFICIENT_DATA lifecycle preserved when telemetry insufficient."""
        intv_doc = {
            "intervention_id": "INTV-001",
            "incident_id": "INC-001",
            "worker_id": "W-101",
            "station_id": "STATION-01",
            "status": "APPLIED",
            "effectiveness_status": "WAITING_FOR_SUFFICIENT_DATA",
            "frames_observed": 5,
            "min_required_frames": 20,
            "pre_intervention_metrics": {
                "cumulative_risk_score": 75.0,
                "peak_lumbar_torque_nm": 82.0,
            },
        }
        normalized = normalize_intervention(intv_doc)
        self.assertEqual(normalized["effectiveness"], "WAITING_FOR_SUFFICIENT_DATA")
        self.assertIsNone(normalized["post_metrics"], "post_metrics must be None when waiting for data, not 0")
        self.assertIsNotNone(normalized["pre_metrics"])
        self.assertEqual(normalized["pre_metrics"]["risk_score"], 75.0)

    def test_12_digital_twin_modelled_simulated(self):
        """Requirement 12: Digital Twin outputs are strictly tagged MODELLED / SIMULATED."""
        sim_res = opensearch_dashboard_backend.simulate_whatif({
            "pallet_height_cm": 85.0,
            "load_weight_kg": 8.0,
            "handling_frequency_per_min": 12.0,
            "task_duration_minutes": 45.0,
            "task_rotation": True,
        })
        self.assertEqual(sim_res["data_source"], "MODELLED")
        self.assertEqual(sim_res["simulation_mode"], "PREDICTIVE_DIGITAL_TWIN")
        self.assertIn("simulated_configuration", sim_res)

    def test_13_ai_cannot_overwrite_deterministic_telemetry(self):
        """Requirement 18: Deterministic engine remains authoritative over AI explanations."""
        doc = {
            "incident_id": "INC-002",
            "worker_id": "W-102",
            "station_id": "STATION-02",
            "risk_score": 82.5,
            "lumbar_torque_nm": 94.2,
            "spinal_compression_n": 2750.0,
            "authoritative_biomechanics": {
                "peak_lumbar_torque_nm": 94.2,
                "peak_spinal_compression_n": 2750.0,
                "peak_trunk_flexion_deg": 48.0,
            },
            "gemini_multimodal_reasoning": "Worker shows safe posture with 10 Nm torque.",  # conflicting AI text
        }
        normalized = normalize_incident(doc)
        self.assertEqual(normalized["lumbar_torque_nm"], 94.2)
        self.assertEqual(normalized["spinal_compression_n"], 2750)
        self.assertEqual(normalized["trunk_flexion_deg"], 48.0)
        self.assertEqual(normalized["risk_score"], 82.5)


    def test_14_station_provenance_and_non_live_stations(self):
        """Audit Case 1 & 2: STATION-NORTH and STATION-OS-01 classified as TEST_FIXTURE, active_worker_id=None."""
        doc_north = {
            "station_id": "STATION-NORTH",
            "mean_risk_score": 75.0,
            "total_incidents": 3,
        }
        st_north = normalize_station_profile("STATION-NORTH", doc_north, [], [])
        self.assertEqual(st_north["data_source"], DATA_SOURCE_TEST)
        self.assertIsNone(st_north["active_worker_id"], "Non-live station must have Active Operative: None")

        doc_os01 = {
            "station_id": "STATION-OS-01",
            "mean_risk_score": 64.5,
            "highest_compression_n": 2900.0,
            "total_incidents": 8,
        }
        st_os01 = normalize_station_profile("STATION-OS-01", doc_os01, [], [])
        self.assertEqual(st_os01["data_source"], DATA_SOURCE_TEST)
        self.assertIsNone(st_os01["active_worker_id"], "Test fixture station must have Active Operative: None")

    def test_15_test_workers_provenance_and_inactive_state(self):
        """Audit Case 3: W-OP-00, W-OP-01, W-OP-02 classified as TEST_FIXTURE, never ACTIVE MONITORING without telemetry."""
        for i in range(3):
            wid = f"W-OP-0{i}"
            w_doc = {
                "worker_id": wid,
                "department": "Logistics",
                "fatigue_index": 30.0 + i * 15.0,
            }
            passport = normalize_worker_passport(wid, w_doc, [], [])
            self.assertEqual(passport["data_source"], DATA_SOURCE_TEST)
            self.assertEqual(passport["status"], "TEST_FIXTURE")
            self.assertNotEqual(passport["status"], "ACTIVE MONITORING", "Worker without session telemetry must not be ACTIVE MONITORING")

    def test_16_intervention_effectiveness_insufficient_frames(self):
        """Audit Case 4: INTV-INC-20260920-CAMERA_01-001 with 5/20 frames reverts to WAITING_FOR_SUFFICIENT_DATA."""
        intv_doc = {
            "intervention_id": "INTV-INC-20260920-CAMERA_01-001",
            "incident_id": "INC-20260920-CAMERA_01-001",
            "worker_id": "W-C01-SESS",
            "station_id": "STATION-01",
            "camera_id": "camera_01",
            "effectiveness_status": "EFFECTIVE",
            "frames_observed": 5,
            "min_required_frames": 20,
            "pre_intervention_metrics": {
                "cumulative_risk_score": 71.3,
                "peak_lumbar_torque_nm": 75.0,
            },
            "post_intervention_metrics": {
                "cumulative_risk_score": 31.3,
                "peak_lumbar_torque_nm": 110.7,
            },
            "risk_reduction_pct": 56.2,
        }
        normalized = normalize_intervention(intv_doc)
        self.assertEqual(normalized["effectiveness"], "WAITING_FOR_SUFFICIENT_DATA")
        self.assertIsNone(normalized["post_metrics"])
        self.assertIsNone(normalized["effectiveness_assessment"]["risk_reduction_pct"])
        self.assertTrue(normalized["is_waiting_for_data"])

    def test_17_live_station_and_worker_rules(self):
        """Audit Case 5, 6, 7: Real workers (W-C01-SESS) bind to real stations (STATION-01) with live telemetry."""
        active_passports = [
            {"worker_id": "W-C01-SESS", "station_id": "STATION-01", "data_source": DATA_SOURCE_REAL},
            {"worker_id": "W-C02-SESS", "station_id": "STATION-02", "data_source": DATA_SOURCE_REAL},
            {"worker_id": "W-C03-SESS", "station_id": "STATION-03", "data_source": DATA_SOURCE_REAL},
        ]
        st1 = normalize_station_profile("STATION-01", {}, [], active_passports)
        self.assertEqual(st1["data_source"], DATA_SOURCE_REAL)
        self.assertEqual(st1["active_worker_id"], "W-C01-SESS")

    def test_18_front_page_architecture_block(self):
        """Audit Case 8: Front page contains visual architecture block with product-facing names."""
        from src.dashboard_server import DASHBOARD_HTML
        self.assertIn("KINETICGUARD ARCHITECTURE", DASHBOARD_HTML)
        self.assertIn("AI explains. Deterministic systems measure. Policy governs.", DASHBOARD_HTML)
        self.assertIn("PRIVACY GUARD", DASHBOARD_HTML)
        self.assertIn("POSE ENGINE", DASHBOARD_HTML)
        self.assertIn("BIOMECHANICS", DASHBOARD_HTML)
        self.assertIn("TEMPORAL RISK", DASHBOARD_HTML)
        self.assertIn("KGVISION", DASHBOARD_HTML)
        self.assertIn("POLICY GATE", DASHBOARD_HTML)
        self.assertIn("INTERVENTION", DASHBOARD_HTML)
        self.assertIn("RE-MONITOR", DASHBOARD_HTML)
        self.assertIn("VERIFY", DASHBOARD_HTML)
        self.assertIn("CLOSED-LOOP: VERIFY", DASHBOARD_HTML)


if __name__ == "__main__":
    unittest.main()
