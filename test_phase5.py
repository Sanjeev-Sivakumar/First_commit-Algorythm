"""
KineticGuard - Phase 5 Test Suite: Local OpenSearch, Event Dispatch & Closed-Loop Re-Monitoring
Verifies:
1. Local OpenSearch persistent storage engine, index management, document CRUD, and query DSL.
2. Indexing of incident, worker exposure, station risk, agent decision, and intervention records.
3. Strands tools querying historical data from OpenSearch.
4. Local event dispatcher routing for LOG, REVIEW/NOTIFY, and ESCALATE decisions.
5. Real intervention generation linked to incident, worker, and station identities.
6. Re-monitoring engine returning WAITING_FOR_SUFFICIENT_DATA when post-intervention frames are insufficient.
7. Re-monitoring engine calculating real pre vs post comparative delta when sufficient real frames are provided.
8. Zero fabricated effectiveness, zero static/fake worker data, and zero AWS cloud dependencies.
"""

import json
import os
import unittest
from pathlib import Path

from src.local_opensearch import (
    KineticGuardOpenSearch,
    INDEX_INCIDENTS,
    INDEX_WORKER_EXPOSURE,
    INDEX_STATION_RISK,
    INDEX_AGENT_DECISIONS,
    INDEX_INTERVENTIONS,
    opensearch_engine,
)
from src.event_dispatcher import SafetyEventDispatcher
from src.remonitor_engine import ClosedLoopRemonitorEngine
from src.safety_guardian_agent import (
    StrandsSafetyGuardianAgent,
    get_worker_exposure,
    get_station_risk,
    get_intervention_history,
)


class TestPhase5OpenSearchClosedLoop(unittest.TestCase):
    """Test suite for Phase 5: Local OpenSearch, Event Dispatch, and Re-Monitoring."""

    @classmethod
    def setUpClass(cls):
        cls.opensearch = opensearch_engine
        cls.dispatcher = SafetyEventDispatcher(opensearch=cls.opensearch)
        cls.remonitor = ClosedLoopRemonitorEngine(opensearch=cls.opensearch)
        cls.guardian = StrandsSafetyGuardianAgent(opensearch=cls.opensearch)

    def test_01_local_opensearch_indexing_and_retrieval(self):
        """Verify document indexing, persistence to disk, and retrieval via OpenSearch engine."""
        doc = {
            "incident_id": "INC-TEST-OS-001",
            "worker_id": "W-TEST-01",
            "station_id": "STATION-01",
            "cumulative_risk_score": 74.2,
            "risk_level": "HIGH",
        }
        res = self.opensearch.index_document(INDEX_INCIDENTS, "INC-TEST-OS-001", doc)
        self.assertEqual(res["result"], "created")

        retrieved = self.opensearch.get_document(INDEX_INCIDENTS, "INC-TEST-OS-001")
        self.assertIsNotNone(retrieved)
        self.assertEqual(retrieved["incident_id"], "INC-TEST-OS-001")
        self.assertEqual(retrieved["cumulative_risk_score"], 74.2)

        # Confirm physical persistence to disk
        expected_file = self.opensearch.data_dir / INDEX_INCIDENTS / "INC-TEST-OS-001.json"
        self.assertTrue(expected_file.exists(), f"File {expected_file} was not persisted to disk")

    def test_02_opensearch_dsl_search_queries(self):
        """Verify OpenSearch query DSL execution with term filtering and sorting."""
        # Index multiple test documents
        for i in range(3):
            self.opensearch.index_document(
                INDEX_WORKER_EXPOSURE,
                f"W-OP-0{i}",
                {"worker_id": f"W-OP-0{i}", "department": "Logistics", "fatigue_index": 30.0 + i * 15.0},
            )

        query = {"query": {"term": {"department": "Logistics"}}}
        results = self.opensearch.search(INDEX_WORKER_EXPOSURE, query)
        self.assertGreaterEqual(len(results), 3)
        self.assertTrue(all(r["department"] == "Logistics" for r in results))

    def test_03_strands_tools_read_historical_data_from_opensearch(self):
        """Verify that Strands tools retrieve historical records directly from OpenSearch."""
        # Index worker profile and station risk in OpenSearch
        test_worker = "W-C01-OS-HIST"
        test_station = "STATION-OS-01"

        self.opensearch.index_worker_exposure(test_worker, {
            "worker_id": test_worker,
            "total_incidents": 5,
            "fatigue_index": 72.0,
            "mean_lumbar_torque_nm": 88.0,
            "peak_spinal_compression_n": 2400.0,
            "risk_trend": "CRITICAL",
        })

        self.opensearch.index_station_risk(test_station, {
            "station_id": test_station,
            "total_incidents": 8,
            "mean_risk_score": 64.5,
            "highest_compression_n": 2900.0,
            "dominant_posture": "BENDING",
        })

        # Tool 1: Worker exposure tool reads from OpenSearch
        worker_info = json.loads(get_worker_exposure(test_worker))
        self.assertEqual(worker_info["worker_id"], test_worker)
        self.assertEqual(worker_info["total_incidents"], 5)
        self.assertEqual(worker_info["fatigue_index"], 72.0)
        self.assertEqual(worker_info["source"], "opensearch_indexed_profile")

        # Tool 2: Station risk tool reads from OpenSearch
        stn_info = json.loads(get_station_risk(test_station))
        self.assertEqual(stn_info["station_id"], test_station)
        self.assertEqual(stn_info["total_incidents"], 8)
        self.assertEqual(stn_info["mean_risk_score"], 64.5)
        self.assertEqual(stn_info["source"], "opensearch_station_index")

    def test_04_local_event_dispatcher_routing_log_review_escalate(self):
        """Verify local event dispatcher routes LOG, REVIEW/NOTIFY, and ESCALATE to OpenSearch and events."""
        # Case A: LOG
        dec_log = {
            "incident_id": "INC-EVT-LOG-01",
            "decision": "LOG",
            "worker_identity": {"worker_id": "W-LOG-01", "station_id": "STATION-01", "camera_id": "camera_01"},
            "authoritative_biomechanics": {"cumulative_risk_score": 30.0, "peak_spinal_compression_n": 1400.0},
            "recommended_action": "Continue routine monitoring",
        }
        res_log = self.dispatcher.dispatch_decision(dec_log)
        self.assertEqual(res_log["event_dispatched"]["event_type"], "AUDIT_LOG_EVENT")
        self.assertIsNone(res_log["intervention_created"])

        # Case B: REVIEW/NOTIFY
        dec_notify = {
            "incident_id": "INC-EVT-REV-01",
            "decision": "REVIEW/NOTIFY",
            "worker_identity": {"worker_id": "W-REV-01", "station_id": "STATION-02", "camera_id": "camera_02"},
            "authoritative_biomechanics": {"cumulative_risk_score": 58.0, "peak_spinal_compression_n": 2200.0},
            "recommended_action": "Supervisor station review and task rotation",
        }
        res_notify = self.dispatcher.dispatch_decision(dec_notify)
        self.assertEqual(res_notify["event_dispatched"]["event_type"], "SUPERVISOR_NOTIFICATION_EVENT")
        self.assertIsNotNone(res_notify["intervention_created"])
        self.assertEqual(res_notify["intervention_created"]["action_type"], "SUPERVISOR_REVIEW")

        # Case C: ESCALATE
        dec_esc = {
            "incident_id": "INC-EVT-ESC-01",
            "decision": "ESCALATE",
            "worker_identity": {"worker_id": "W-ESC-01", "station_id": "STATION-03", "camera_id": "camera_03"},
            "authoritative_biomechanics": {"cumulative_risk_score": 82.0, "peak_spinal_compression_n": 3550.0},
            "recommended_action": "Immediate halt of manual picking and mandatory postural recovery",
        }
        res_esc = self.dispatcher.dispatch_decision(dec_esc)
        self.assertEqual(res_esc["event_dispatched"]["event_type"], "CRITICAL_ESCALATION_EVENT")
        self.assertIsNotNone(res_esc["intervention_created"])
        self.assertEqual(res_esc["intervention_created"]["action_type"], "MANDATORY_ROTATION_AND_REST")

    def test_05_intervention_record_linkage_in_opensearch(self):
        """Verify real intervention records are linked to actual incident, worker, and station."""
        dec = {
            "incident_id": "INC-LINK-777",
            "decision": "ESCALATE",
            "worker_identity": {"worker_id": "W-LINK-77", "station_id": "STATION-NORTH", "camera_id": "camera_02"},
            "authoritative_biomechanics": {"cumulative_risk_score": 75.0, "peak_spinal_compression_n": 3450.0},
            "recommended_action": "Enforce scissor lift table use",
        }
        dispatch_res = self.dispatcher.dispatch_decision(dec)
        intv = dispatch_res["intervention_created"]

        self.assertEqual(intv["incident_id"], "INC-LINK-777")
        self.assertEqual(intv["worker_id"], "W-LINK-77")
        self.assertEqual(intv["station_id"], "STATION-NORTH")
        self.assertEqual(intv["status"], "APPLIED")
        self.assertEqual(intv["effectiveness_status"], "WAITING_FOR_SUFFICIENT_DATA")

        # Confirm stored in OpenSearch
        stored = self.opensearch.get_document(INDEX_INTERVENTIONS, intv["intervention_id"])
        self.assertIsNotNone(stored)
        self.assertEqual(stored["worker_id"], "W-LINK-77")

    def test_06_remonitor_waiting_for_sufficient_data_guard(self):
        """Verify re-monitoring engine strictly returns WAITING_FOR_SUFFICIENT_DATA when post frames < min."""
        # Create an intervention in OpenSearch
        intv_id = "INTV-GUARD-TEST-001"
        self.opensearch.index_intervention({
            "intervention_id": intv_id,
            "incident_id": "INC-GUARD-001",
            "worker_id": "W-GUARD-01",
            "station_id": "STATION-01",
            "status": "APPLIED",
            "pre_intervention_metrics": {"cumulative_risk_score": 72.0},
        })

        # Provide only 5 frames (< required 20)
        sparse_frames = [
            {"risk_score": 30.0, "lumbar_torque_nm": 35.0, "compression_force_n": 1100.0, "posture": "UPRIGHT"}
            for _ in range(5)
        ]

        result = self.remonitor.evaluate_intervention_effectiveness(
            intervention_id=intv_id,
            subsequent_frames=sparse_frames,
            min_required_frames=20,
        )

        self.assertEqual(result["status"], "WAITING_FOR_SUFFICIENT_DATA")
        self.assertIsNone(result["is_effective"])
        self.assertIsNone(result["risk_reduction_pct"])
        self.assertEqual(result["frames_observed"], 5)
        self.assertEqual(result["frames_required"], 20)

    def test_07_remonitor_real_effectiveness_comparison(self):
        """Verify comparative analysis calculates real reduction percentage when sufficient frames are provided."""
        intv_id = "INTV-REDUCE-TEST-001"
        self.opensearch.index_intervention({
            "intervention_id": intv_id,
            "incident_id": "INC-REDUCE-001",
            "worker_id": "W-REDUCE-01",
            "station_id": "STATION-01",
            "status": "APPLIED",
            "pre_intervention_metrics": {
                "cumulative_risk_score": 75.0,
                "peak_lumbar_torque_nm": 95.0,
                "peak_spinal_compression_n": 2400.0,
            },
        })

        # Provide 25 subsequent frames with lower risk (e.g. 35.0)
        improved_frames = [
            {
                "frame_index": i,
                "risk_score": 32.0 if i % 2 == 0 else 38.0,
                "lumbar_torque_nm": 42.0,
                "compression_force_n": 1250.0,
                "posture": "UPRIGHT",
            }
            for i in range(25)
        ]

        result = self.remonitor.evaluate_intervention_effectiveness(
            intervention_id=intv_id,
            subsequent_frames=improved_frames,
            min_required_frames=20,
        )

        self.assertEqual(result["status"], "EFFECTIVE")
        self.assertTrue(result["is_effective"])
        self.assertGreater(result["risk_reduction_pct"], 40.0)  # Dropped from 75 to ~35
        self.assertEqual(result["frames_analyzed"], 25)

        # Confirm OpenSearch record updated
        updated_intv = self.opensearch.get_document(INDEX_INTERVENTIONS, intv_id)
        self.assertEqual(updated_intv["status"], "RESOLVED")
        self.assertEqual(updated_intv["effectiveness_status"], "EFFECTIVE")

    def test_08_end_to_end_phase5_closed_loop_execution(self):
        """Verify end-to-end execution: incident -> Strands agent -> OpenSearch -> event -> intervention -> re-monitor."""
        packet = {
            "schema_version": "kineticguard.strands.v1",
            "incident_id": "INC-P5-E2E-001",
            "worker_identity": {"worker_id": "W-E2E-01", "station_id": "STATION-E2E", "camera_id": "camera_01"},
            "biomechanical_exposure": {
                "peak_lumbar_torque_nm": 92.0,
                "peak_spinal_compression_n": 3450.0,  # Breaches NIOSH AL -> Cedar ESCALATE
                "awkward_duty_cycle_pct": 82.0,
                "repetition_rate_per_min": 6.5,
            },
            "risk_assessment": {
                "cumulative_risk_score": 78.0,
                "risk_level": "DANGEROUS",
            },
        }

        # 1. Process incident with Strands Agent
        agent_out = self.guardian.process_incident(packet)
        self.assertEqual(agent_out["decision"], "ESCALATE")

        # 2. Verify OpenSearch indexing
        doc = self.opensearch.get_document(INDEX_INCIDENTS, "INC-P5-E2E-001")
        self.assertIsNotNone(doc)

        # 3. Verify event dispatch receipt
        receipt = agent_out["dispatch_receipt"]
        self.assertTrue(receipt["opensearch_indexed"])
        self.assertEqual(receipt["event_dispatched"]["event_type"], "CRITICAL_ESCALATION_EVENT")

        # 4. Verify intervention created in OpenSearch
        intv_id = receipt["intervention_created"]["intervention_id"]
        intv_doc = self.opensearch.get_document(INDEX_INTERVENTIONS, intv_id)
        self.assertIsNotNone(intv_doc)
        self.assertEqual(intv_doc["effectiveness_status"], "WAITING_FOR_SUFFICIENT_DATA")


if __name__ == "__main__":
    unittest.main()
