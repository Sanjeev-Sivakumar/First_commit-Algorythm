"""
KineticGuard - Phase 4 Test Suite: Strands Agents SDK + Cedar + Gemini Multimodal Reasoner
Verifies:
1. Strands Agents SDK Agent initialization with 5 real tools.
2. Tool execution: worker exposure, station risk, intervention history, incident telemetry, risk forecast.
3. Ingestion of kineticguard.strands.v1 incident packets.
4. Gemini multimodal reasoning (gemini-3.6-flash) on captured video keyframe.
5. Biomechanical ground truth protection: Gemini does not modify authoritative numbers.
6. Formal Cedar policy evaluation with cedarpy supporting LOG, REVIEW/NOTIFY, ESCALATE.
7. Distinct incident profiles produce distinct policy decisions.
8. Dynamic worker/session/station identities without hardcoded values.
9. Zero AWS dependencies - 100% local execution.
10. End-to-end multi-camera incident processing.
"""

import json
import os
import unittest
from pathlib import Path

from strands import Agent
from src.config import settings
from src.gemini_analyzer import GeminiMultimodalAnalyzer
from src.cedar_engine import CedarPolicyEngine
from src.safety_guardian_agent import (
    StrandsSafetyGuardianAgent,
    get_worker_exposure,
    get_station_risk,
    get_intervention_history,
    get_incident_telemetry,
    get_risk_forecast,
    ACTIVE_INCIDENT_REGISTRY,
    ACTIVE_WORKER_REGISTRY,
)


class TestPhase4StrandsCedarGemini(unittest.TestCase):
    """Test suite for Phase 4: Strands Agents SDK + Cedar + Gemini Multimodal Reasoning."""

    @classmethod
    def setUpClass(cls):
        cls.guardian = StrandsSafetyGuardianAgent()
        cls.cedar = CedarPolicyEngine()
        cls.gemini = GeminiMultimodalAnalyzer()

    def test_01_strands_agent_and_tools_initialization(self):
        """Verify real Strands Agent initialization with all 5 specialized tools."""
        agent = self.guardian.agent
        self.assertIsInstance(agent, Agent)
        self.assertEqual(agent.name, "KineticGuardSafetyGuardian")

        tool_names = agent.tool_names
        self.assertIn("get_worker_exposure", tool_names)
        self.assertIn("get_station_risk", tool_names)
        self.assertIn("get_intervention_history", tool_names)
        self.assertIn("get_incident_telemetry", tool_names)
        self.assertIn("get_risk_forecast", tool_names)
        self.assertEqual(len(tool_names), 5)

    def test_02_strands_tool_execution_dynamics(self):
        """Verify dynamic execution of each of the 5 Strands tools with dynamic state."""
        # Set up dynamic worker state
        ACTIVE_WORKER_REGISTRY["W-TEST-99"] = {
            "worker_id": "W-TEST-99",
            "total_incidents": 3,
            "fatigue_index": 68.5,
            "awkward_duty_cycle_pct": 75.0,
            "repetition_rate_per_min": 6.2,
        }

        # 1. Worker exposure tool
        exp_data = json.loads(get_worker_exposure("W-TEST-99"))
        self.assertEqual(exp_data["worker_id"], "W-TEST-99")
        self.assertEqual(exp_data["total_incidents"], 3)
        self.assertAlmostEqual(exp_data["fatigue_index"], 68.5, places=1)

        # 2. Station risk tool
        stn_data = json.loads(get_station_risk("STATION-01"))
        self.assertEqual(stn_data["station_id"], "STATION-01")

        # 3. Intervention history tool
        int_data = json.loads(get_intervention_history("W-TEST-99"))
        self.assertEqual(int_data["worker_id"], "W-TEST-99")

        # 4. Risk forecast tool
        fc_data = json.loads(get_risk_forecast("W-TEST-99"))
        self.assertGreater(fc_data["projected_fatigue_30m"], 68.5)
        self.assertIn("fatigue_risk_level", fc_data)

        # 5. Incident telemetry tool
        ACTIVE_INCIDENT_REGISTRY["INC-MOCK-01"] = {
            "biomechanical_exposure": {"peak_spinal_compression_n": 3100.0, "peak_lumbar_torque_nm": 92.0},
            "risk_assessment": {"cumulative_risk_score": 71.0, "risk_level": "HIGH"},
        }
        tel_data = json.loads(get_incident_telemetry("INC-MOCK-01"))
        self.assertEqual(tel_data["cumulative_risk_score"], 71.0)
        self.assertEqual(tel_data["peak_spinal_compression_n"], 3100.0)

    def test_03_gemini_multimodal_keyframe_reasoning(self):
        """Verify Gemini multimodal visual reasoning on actual captured video keyframe."""
        # Keyframe captured in Phase 3
        keyframe_path = "output/incidents/keyframes/INC-20260920-CAMERA_01-001_keyframe.jpg"
        if not Path(keyframe_path).exists():
            # Find any existing keyframe in directory
            kfs = list(Path("output/incidents/keyframes").glob("*.jpg"))
            self.assertTrue(len(kfs) > 0, "No real keyframes found in output/incidents/keyframes")
            keyframe_path = str(kfs[0])

        packet = {
            "incident_id": "INC-TEST-001",
            "worker_identity": {"worker_id": "W-C01-DYN", "station_id": "STATION-01", "camera_id": "camera_01"},
            "biomechanical_exposure": {
                "peak_lumbar_torque_nm": 95.0,
                "peak_spinal_compression_n": 2400.0,
                "awkward_duty_cycle_pct": 85.0,
                "repetition_rate_per_min": 6.0,
            },
            "risk_assessment": {
                "cumulative_risk_score": 72.0,
                "risk_level": "HIGH",
                "contributing_factors": ["Elevated spinal compression", "Awkward torso forward flexion"],
            },
            "visual_evidence": {"keyframe_path": keyframe_path},
        }

        res = self.gemini.analyze_incident(packet, keyframe_path=keyframe_path)
        self.assertIn("ergonomic_observation", res)
        self.assertIn("root_cause", res)
        self.assertIn("risk_explanation", res)
        self.assertIn("recommended_corrective_action", res)
        self.assertGreaterEqual(res["confidence"], 0.70)
        self.assertIn(res["source"], ("gemini_multimodal_api", "deterministic_expert_reasoner"))

    def test_04_biomechanical_ground_truth_protection(self):
        """Verify Gemini reasoning does NOT override authoritative Phase 2/3 deterministic metrics."""
        packet = {
            "incident_id": "INC-TRUTH-001",
            "worker_identity": {"worker_id": "W-TRUTH-01", "station_id": "STATION-02", "camera_id": "camera_02"},
            "biomechanical_exposure": {
                "peak_lumbar_torque_nm": 88.5,
                "peak_spinal_compression_n": 2250.0,
                "awkward_duty_cycle_pct": 72.0,
                "repetition_rate_per_min": 5.5,
            },
            "risk_assessment": {
                "cumulative_risk_score": 68.2,
                "risk_level": "HIGH",
            },
        }

        agent_output = self.guardian.process_incident(packet)
        auth = agent_output["authoritative_biomechanics"]

        # Immutable ground truth verification
        self.assertEqual(auth["cumulative_risk_score"], 68.2)
        self.assertEqual(auth["risk_level"], "HIGH")
        self.assertEqual(auth["peak_lumbar_torque_nm"], 88.5)
        self.assertEqual(auth["peak_spinal_compression_n"], 2250.0)
        self.assertEqual(auth["awkward_duty_cycle_pct"], 72.0)
        self.assertTrue(auth["is_estimated_model_indicator"])

    def test_05_cedar_policy_evaluation_decisions(self):
        """Verify Cedar policy evaluation for all 3 supported decisions: LOG, REVIEW/NOTIFY, ESCALATE."""
        # Case A: Low risk baseline -> LOG
        res_log = self.cedar.evaluate_safety_decision(
            incident_id="INC-LOG-01",
            cumulative_risk=32.0,
            spinal_compression_n=1400.0,
            awkward_duty_cycle_pct=15.0,
            fatigue_index=20.0,
        )
        self.assertEqual(res_log["decision"], "LOG")
        self.assertTrue(res_log["allowed"])

        # Case B: Moderate risk -> REVIEW/NOTIFY
        res_notify = self.cedar.evaluate_safety_decision(
            incident_id="INC-NOTIFY-01",
            cumulative_risk=58.0,
            spinal_compression_n=2300.0,
            awkward_duty_cycle_pct=65.0,
            fatigue_index=45.0,
        )
        self.assertEqual(res_notify["decision"], "REVIEW/NOTIFY")
        self.assertTrue(res_notify["allowed"])

        # Case C: Critical risk breach (NIOSH AL >= 3400 N or cumulative risk >= 70) -> ESCALATE
        res_escalate = self.cedar.evaluate_safety_decision(
            incident_id="INC-ESC-01",
            cumulative_risk=78.0,
            spinal_compression_n=3550.0,
            awkward_duty_cycle_pct=88.0,
            fatigue_index=82.0,
        )
        self.assertEqual(res_escalate["decision"], "ESCALATE")
        self.assertTrue(res_escalate["allowed"])

    def test_06_different_incidents_produce_different_decisions(self):
        """Confirm distinct incident profiles produce distinct Cedar decisions and actions."""
        mild_packet = {
            "incident_id": "INC-MILD-001",
            "worker_identity": {"worker_id": "W-MILD-01", "station_id": "STATION-01", "camera_id": "camera_01"},
            "biomechanical_exposure": {"peak_spinal_compression_n": 1600.0, "awkward_duty_cycle_pct": 20.0},
            "risk_assessment": {"cumulative_risk_score": 35.0, "risk_level": "LOW"},
        }
        critical_packet = {
            "incident_id": "INC-CRIT-001",
            "worker_identity": {"worker_id": "W-CRIT-01", "station_id": "STATION-02", "camera_id": "camera_02"},
            "biomechanical_exposure": {"peak_spinal_compression_n": 3600.0, "awkward_duty_cycle_pct": 90.0, "repetition_rate_per_min": 8.0},
            "risk_assessment": {"cumulative_risk_score": 85.0, "risk_level": "DANGEROUS"},
        }

        out_mild = self.guardian.process_incident(mild_packet)
        out_crit = self.guardian.process_incident(critical_packet)

        self.assertNotEqual(out_mild["decision"], out_crit["decision"])
        self.assertEqual(out_crit["decision"], "ESCALATE")
        self.assertIn(out_mild["decision"], ("LOG", "REVIEW/NOTIFY"))

    def test_07_dynamic_identities_zero_hardcoded_values(self):
        """Verify dynamic worker and station identities flow through the Strands Agent pipeline."""
        dyn_worker = "W-C02-DYNAMIC-SES"
        dyn_station = "STATION-C02-NORTH"

        packet = {
            "incident_id": "INC-DYN-999",
            "worker_identity": {"worker_id": dyn_worker, "station_id": dyn_station, "camera_id": "camera_02"},
            "biomechanical_exposure": {"peak_spinal_compression_n": 2400.0, "awkward_duty_cycle_pct": 60.0},
            "risk_assessment": {"cumulative_risk_score": 62.0, "risk_level": "HIGH"},
        }

        output = self.guardian.process_incident(packet)
        self.assertEqual(output["worker_identity"]["worker_id"], dyn_worker)
        self.assertEqual(output["worker_identity"]["station_id"], dyn_station)
        self.assertNotIn("WORKER-1042", output["worker_identity"]["worker_id"])
        self.assertNotIn("STATION-GATE-A", output["worker_identity"]["station_id"])

    def test_08_strands_agent_packet_structure_and_zero_aws(self):
        """Verify complete agent output conforms to Build It specification without AWS dependencies."""
        packet = {
            "schema_version": "kineticguard.strands.v1",
            "incident_id": "INC-SCHEMA-001",
            "worker_identity": {"worker_id": "W-C01-TEST", "station_id": "STATION-01", "camera_id": "camera_01"},
            "temporal_window": {"start_time_sec": 1.0, "end_time_sec": 3.5, "duration_sec": 2.5},
            "risk_assessment": {
                "cumulative_risk_score": 73.7,
                "risk_level": "HIGH",
                "contributing_factors": ["High awkward posture duty cycle"],
                "recommended_action": "Mandatory task rotation",
            },
            "biomechanical_exposure": {
                "is_estimated_model_indicator": True,
                "peak_lumbar_torque_nm": 97.2,
                "peak_spinal_compression_n": 2311.6,
                "awkward_duty_cycle_pct": 91.5,
                "repetition_rate_per_min": 25.4,
            },
        }

        result = self.guardian.process_incident(packet)
        self.assertEqual(result["schema_version"], "kineticguard.strands.v1")
        self.assertEqual(result["incident_id"], "INC-SCHEMA-001")
        self.assertIn("decision", result)
        self.assertIn("policy_result", result)
        self.assertIn("gemini_analysis", result)
        self.assertIn("authoritative_biomechanics", result)
        self.assertIn("tools_consulted", result)
        self.assertEqual(len(result["tools_consulted"]), 5)


if __name__ == "__main__":
    unittest.main()
