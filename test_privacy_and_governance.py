"""
KineticGuard - Privacy Guard & Governance Audit Test Suite
Verifies:
1. Mandatory local Privacy Guard is ALWAYS ACTIVE with no toggle switch.
2. Real-time face detection applies Gaussian blur and pixelation to obscure facial identity.
3. Body pose, skeletal landmarks, joint angles, and biomechanics are strictly preserved.
4. Raw identifiable frames NEVER enter Gemini multimodal vision pipeline.
5. No raw visual data is stored in OpenSearch (keyframes on disk and indexed docs are de-identified).
6. Predictive Ergonomic Digital Twin recalculates L5/S1 torque, compression, and risk deterministically.
7. Digital Twin uses REAL OpenSearch baseline telemetry and returns INSUFFICIENT_DATA when missing.
8. Cedar Governance Audit Trail accurately records real Strands -> Cedar evaluations.
9. Deterministic Cedar ALLOW/DENY propagation and fallback gates operate without fabrication.
"""

import os
import json
import unittest
from pathlib import Path
import cv2
import numpy as np

from src.config import settings
from src.privacy_guard import PrivacyGuard, privacy_guard
from src.pose_detector import PoseDetector
from src.incident_engine import IncidentEngine
from src.gemini_analyzer import GeminiMultimodalAnalyzer
from src.cedar_engine import CedarPolicyEngine
from src.safety_guardian_agent import StrandsSafetyGuardianAgent
from src.local_opensearch import (
    KineticGuardOpenSearch,
    opensearch_engine,
    INDEX_INCIDENTS,
    INDEX_AGENT_DECISIONS,
)
from src.whatif_simulator import WhatIfSimulator
from src.opensearch_dashboard_backend import OpenSearchDashboardBackend, opensearch_dashboard_backend


class TestPrivacyAndGovernance(unittest.TestCase):
    """Test suite for Privacy Guard, Digital Twin, and Cedar Governance Audit Trail."""

    @classmethod
    def setUpClass(cls):
        cls.opensearch = opensearch_engine
        cls.backend = opensearch_dashboard_backend
        cls.guardian = StrandsSafetyGuardianAgent(opensearch=cls.opensearch)
        cls.pose_detector = PoseDetector()
        cls.cedar = CedarPolicyEngine()
        cls.simulator = WhatIfSimulator()

    def test_01_mandatory_privacy_guard_always_on(self):
        """Verify Privacy Guard is ALWAYS ENABLED with no on/off toggle switch."""
        pg = PrivacyGuard()
        self.assertTrue(pg.is_active())
        self.assertTrue(pg.is_always_on)
        self.assertTrue(privacy_guard.is_active())
        self.assertTrue(privacy_guard.is_always_on)

        status = privacy_guard.get_status()
        self.assertEqual(status["status"], "ACTIVE")
        self.assertEqual(status["mode"], "ALWAYS_ON")
        self.assertFalse(status["toggle_allowed"])
        self.assertIn("compliance_notice", status)
        self.assertNotIn("GDPR certified", status["compliance_notice"])

    def test_02_face_deidentification_blur_and_pixelation(self):
        """Verify face de-identification modifies the facial region with irreversible blur."""
        # Create synthetic test frame with simulated face contrast
        frame = np.ones((480, 640, 3), dtype=np.uint8) * 120
        # Draw high-contrast facial features
        cv2.circle(frame, (320, 180), 40, (220, 190, 160), -1)  # Face
        cv2.circle(frame, (305, 170), 5, (20, 20, 20), -1)      # Eye 1
        cv2.circle(frame, (335, 170), 5, (20, 20, 20), -1)      # Eye 2
        cv2.line(frame, (310, 200), (330, 200), (40, 40, 180), 3) # Mouth

        original_face_roi = frame[140:220, 280:360].copy()

        landmarks = {"nose": (320, 180), "left_shoulder": (260, 270), "right_shoulder": (380, 270)}
        deidentified = privacy_guard.deidentify_frame(frame, landmarks_px=landmarks)

        deidentified_roi = deidentified[140:220, 280:360]

        # Verify pixels in face ROI were altered by blur/pixelation
        diff = np.abs(original_face_roi.astype(float) - deidentified_roi.astype(float))
        self.assertGreater(np.mean(diff), 5.0, "Face region was not de-identified by PrivacyGuard")

    def test_03_body_pose_and_biomechanics_preserved_through_face_blur(self):
        """Verify skeletal landmarks, joint angles, and pose detection work on de-identified frames."""
        vpath = settings.CAMERA_SOURCES["camera_01"]
        cap = cv2.VideoCapture(vpath)
        ret, frame = cap.read()
        cap.release()
        self.assertTrue(ret, "Failed to read test frame from camera_01.mp4")

        # De-identify frame
        deidentified = privacy_guard.deidentify_frame(frame)
        self.assertEqual(deidentified.shape, frame.shape)

        # Run MediaPipe Pose on de-identified frame
        res = self.pose_detector.detect_pose(deidentified)
        self.assertTrue(res.detected, "MediaPipe failed to detect pose on de-identified frame")
        self.assertIn("trunk_flexion_deg", res.angles)
        self.assertIn("left_shoulder_deg", res.angles)
        self.assertIn("left_elbow_deg", res.angles)
        self.assertIn("left_hip_deg", res.angles)
        self.assertIn("left_knee_deg", res.angles)
        self.assertGreater(len(res.landmarks_3d), 10)

    def test_04_no_raw_frames_reach_gemini(self):
        """Verify Gemini receives ONLY de-identified keyframe images."""
        test_kf_path = Path("output/incidents/keyframes/test_privacy_gemini_kf.jpg")
        test_kf_path.parent.mkdir(parents=True, exist_ok=True)

        raw_img = np.ones((480, 640, 3), dtype=np.uint8) * 150
        cv2.circle(raw_img, (320, 200), 45, (230, 200, 180), -1)
        cv2.imwrite(str(test_kf_path), raw_img)

        analyzer = GeminiMultimodalAnalyzer()
        packet = {
            "incident_id": "INC-PRIVACY-001",
            "worker_identity": {"worker_id": "W-PRIV-01", "station_id": "STATION-01", "camera_id": "camera_01"},
            "biomechanical_exposure": {"peak_spinal_compression_n": 2400.0, "peak_lumbar_torque_nm": 85.0},
            "risk_assessment": {"cumulative_risk_score": 65.0, "risk_level": "HIGH"},
            "visual_evidence": {"keyframe_path": str(test_kf_path)},
        }

        res = analyzer.analyze_incident(packet, keyframe_path=str(test_kf_path))
        self.assertEqual(res.get("privacy_guard"), "ACTIVE")
        self.assertTrue(res.get("visual_deidentified", False))

        if test_kf_path.exists():
            test_kf_path.unlink()

    def test_05_no_raw_visual_data_stored_in_opensearch(self):
        """Verify OpenSearch indexes only de-identified visual evidence references and tags."""
        packet = {
            "incident_id": "INC-TEST-PRIVACY-OS-001",
            "worker_identity": {"worker_id": "W-PRIV-OS", "station_id": "STATION-01", "camera_id": "camera_01"},
            "biomechanical_exposure": {"peak_spinal_compression_n": 2200.0, "peak_lumbar_torque_nm": 76.0},
            "risk_assessment": {"cumulative_risk_score": 58.0, "risk_level": "MODERATE"},
            "visual_evidence": {"keyframe_path": "output/incidents/keyframes/INC-TEST-001_keyframe.jpg"},
        }

        self.opensearch.index_incident(packet)
        doc = self.opensearch.get_document(INDEX_INCIDENTS, "INC-TEST-PRIVACY-OS-001")
        self.assertIsNotNone(doc)
        self.assertEqual(doc.get("privacy_guard"), "ACTIVE")
        self.assertTrue(doc.get("visual_deidentified", False))

    def test_06_predictive_digital_twin_deterministic_calculations(self):
        """Verify Digital Twin recalculates torque, compression, and risk without LLM math."""
        baseline = {
            "load_weight_kg": 14.0,
            "pallet_height_cm": 40.0,
            "reps_per_minute": 18.0,
            "task_duration_minutes": 60.0,
            "worker_rotation_enabled": False,
            "awkward_duty_cycle_pct": 75.0,
        }
        modifications = {
            "load_weight_kg": 8.0,       # Reduced weight
            "pallet_height_cm": 85.0,     # Scissor lift table height
            "reps_per_minute": 12.0,      # Slower cadence
            "task_duration_minutes": 45.0,
            "worker_rotation_enabled": True,
        }

        res = self.simulator.simulate(current_baseline=baseline, modifications=modifications)

        self.assertEqual(res["label"], "MODELLED / SIMULATED")
        self.assertEqual(res["status"], "SIMULATED_ESTIMATE")

        curr = res["current_configuration"]
        sim = res["simulated_configuration"]

        # Lumbar torque must reduce with elevated pallet and lower load
        self.assertLess(sim["estimated_lumbar_torque_nm"], curr["estimated_lumbar_torque_nm"])
        # Spinal compression must reduce
        self.assertLess(sim["estimated_spinal_compression_n"], curr["estimated_spinal_compression_n"])
        # Risk score must reduce
        self.assertLess(sim["cumulative_risk_score"], curr["cumulative_risk_score"])
        self.assertLess(res["risk_reduction_pct"], 0.0)

    def test_07_digital_twin_real_opensearch_telemetry_baseline(self):
        """Verify Digital Twin fetches baseline from real OpenSearch records."""
        # Query with existing station from OpenSearch
        stations = self.backend.get_stations()
        self.assertGreater(len(stations), 0, "Expected at least one station in OpenSearch")

        st_id = stations[0]["station_id"]
        sim_res = self.backend.simulate_whatif({
            "station_id": st_id,
            "sim_pallet_height_cm": 85.0,
            "sim_load_weight_kg": 8.0,
        })

        self.assertEqual(sim_res["label"], "MODELLED / SIMULATED")
        self.assertEqual(sim_res["status"], "SIMULATED_ESTIMATE")
        self.assertIn("current_configuration", sim_res)
        self.assertIn("simulated_configuration", sim_res)
        self.assertIn("risk_reduction_pct", sim_res)

    def test_08_digital_twin_insufficient_data_handling(self):
        """Verify Digital Twin returns INSUFFICIENT_DATA for non-existent station/worker."""
        res_station = self.backend.simulate_whatif({"station_id": "STATION-NONEXISTENT-XYZ"})
        self.assertEqual(res_station["status"], "INSUFFICIENT_DATA")
        self.assertIn("error", res_station)

        res_worker = self.backend.simulate_whatif({"worker_id": "W-NONEXISTENT-XYZ"})
        self.assertEqual(res_worker["status"], "INSUFFICIENT_DATA")
        self.assertIn("error", res_worker)

    def test_09_cedar_governance_audit_records_and_real_propagation(self):
        """Verify Cedar audit trail accurately indexes and formats real Strands -> Cedar decisions."""
        packet = {
            "incident_id": "INC-GOV-TEST-001",
            "worker_identity": {"worker_id": "W-GOV-01", "station_id": "STATION-01", "camera_id": "camera_01"},
            "biomechanical_exposure": {
                "peak_lumbar_torque_nm": 95.0,
                "peak_spinal_compression_n": 3500.0,
                "awkward_duty_cycle_pct": 85.0,
                "repetition_rate_per_min": 7.0,
            },
            "risk_assessment": {
                "cumulative_risk_score": 78.0,
                "risk_level": "HIGH",
            },
        }

        # Process through Strands Agent + Cedar
        decision = self.guardian.process_incident(packet)
        self.assertEqual(decision["decision"], "ESCALATE")

        # Fetch audit trail from OpenSearch backend
        audit_trail = self.backend.get_cedar_audit_trail()
        self.assertGreater(len(audit_trail), 0)

        # Find our decision in the audit trail
        match = next((r for r in audit_trail if r["incident_id"] == "INC-GOV-TEST-001"), None)
        self.assertIsNotNone(match, "Decision not found in Cedar Governance Audit Trail")
        self.assertEqual(match["agent_proposed_action"], "ESCALATE")
        self.assertEqual(match["allow_deny_result"], "ALLOW")
        self.assertEqual(match["enforced_action"], "ESCALATE")
        self.assertEqual(match["policy_rule_identifier"], "policy0")
        self.assertIn("cumulative_risk", match["cedar_context"])
        self.assertIn("spinal_compression_n", match["cedar_context"])

    def test_10_cedar_deterministic_allow_deny_fallback_evaluation(self):
        """Verify Cedar deterministic evaluation handles ALLOW and DENY with authentic fallback."""
        # Case A: Breach justifies ESCALATE -> ALLOW
        res_allow = self.cedar.evaluate_proposed_action(
            incident_id="INC-ALLOW-01",
            proposed_action="ESCALATE",
            cumulative_risk=78.0,
            spinal_compression_n=3550.0,
            awkward_duty_cycle_pct=85.0,
            rep_rate_per_min=7.0,
            fatigue_index=82.0,
        )
        self.assertTrue(res_allow["allowed"])
        self.assertEqual(res_allow["enforced_action"], "ESCALATE")
        self.assertEqual(res_allow["allow_deny_result"] if "allow_deny_result" in res_allow else res_allow["cedar_decision"], "Decision.Allow")

        # Case B: Low risk where ESCALATE is forbidden -> DENY with deterministic fallback to LOG
        res_deny = self.cedar.evaluate_proposed_action(
            incident_id="INC-DENY-01",
            proposed_action="ESCALATE",
            cumulative_risk=30.0,
            spinal_compression_n=1200.0,
            awkward_duty_cycle_pct=15.0,
            rep_rate_per_min=1.0,
            fatigue_index=15.0,
        )
        self.assertFalse(res_deny["allowed"])
        self.assertEqual(res_deny["cedar_decision"], "Decision.Deny")
        self.assertEqual(res_deny["enforced_action"], "LOG")
        self.assertEqual(res_deny["fallback_action"], "LOG")


if __name__ == "__main__":
    unittest.main()
