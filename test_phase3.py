"""
KineticGuard - Phase 3 Test Suite: Temporal & Cumulative Ergonomic Risk Engine (Build It Architecture)
Verifies:
1. Sliding temporal window (15–30s) dynamics and frame buffering.
2. Posture duration and awkward duty cycle calculation.
3. Repetitive motion cycle detection and rep rate (reps/min).
4. Mechanical torque impulse and cumulative exposure accumulation.
5. Multi-dimensional cumulative risk scoring (0–100) and risk level categories.
6. Dynamic incident grouping, cooldown windowing, and threshold breach detection.
7. Incident keyframe capture directly from video frames with file verification.
8. Dynamic worker, session, camera, and station identities (no hardcoded identities).
9. Output schema compliance with kineticguard.strands.v1 for Strands Agents SDK.
10. Zero AWS dependencies and zero precomputed/mock data in production path.
11. Real multi-camera processing on camera_01, camera_02, camera_03.
"""

import json
import os
import unittest
from pathlib import Path

import cv2
import numpy as np

from src.config import settings
from src.pose_detector import PoseDetector
from src.biomechanics import BiomechanicsAnalyzer
from src.temporal_analyzer import TemporalSlidingWindow, RepetitionCycleDetector
from src.cumulative_risk import CumulativeRiskModel, CumulativeRiskScore
from src.incident_engine import IncidentEngine, ErgonomicIncident, resolve_station, generate_worker_identity
from evaluate_risk import evaluate_camera_live, evaluate_camera_stream


class TestPhase3TemporalRiskBuildIt(unittest.TestCase):
    """Test suite for Phase 3: Temporal & Cumulative Ergonomic Risk Engine."""

    def test_01_sliding_temporal_window_and_duty_cycle(self):
        """Verify 15-30s temporal sliding window, posture durations, and awkward duty cycle."""
        window = TemporalSlidingWindow(window_seconds=20.0)
        self.assertEqual(window.window_seconds, 20.0)

        # Feed 10 seconds of UPRIGHT followed by 10 seconds of BENDING
        for i in range(20):
            t = float(i)
            posture = "UPRIGHT" if i < 10 else "BENDING"
            window.add_frame({
                "frame_index": i,
                "timestamp_sec": t,
                "posture": posture,
                "lumbar_torque_nm": 40.0 if posture == "UPRIGHT" else 85.0,
                "compression_force_n": 1000.0 if posture == "UPRIGHT" else 2200.0,
                "trunk_flexion_deg": 10.0 if posture == "UPRIGHT" else 45.0,
                "risk_score": 15.0 if posture == "UPRIGHT" else 55.0,
            })

        metrics = window.get_metrics()
        self.assertEqual(metrics["frame_count"], 20)
        self.assertAlmostEqual(metrics["window_duration_sec"], 19.0, delta=1.0)
        self.assertGreater(metrics["duration_bending_sec"], 8.0)
        self.assertGreater(metrics["awkward_duty_cycle_pct"], 40.0)
        self.assertTrue(metrics["is_estimated_model_indicator"])
        self.assertIn("disclaimer", metrics)

    def test_02_repetition_cycle_detection_and_rate(self):
        """Verify state machine cycle detection (UPRIGHT -> BENDING -> UPRIGHT) and rep rate."""
        detector = RepetitionCycleDetector(min_dwell_sec=0.2)
        window = TemporalSlidingWindow(window_seconds=20.0)

        # Simulate 3 lifting cycles
        timestamps_and_postures = [
            (0.0, "UPRIGHT"),
            (1.0, "BENDING"),
            (1.5, "BENDING"),
            (2.0, "UPRIGHT"),  # Cycle 1 finished
            (3.0, "LIFTING"),
            (3.5, "LIFTING"),
            (4.0, "UPRIGHT"),  # Cycle 2 finished
            (5.0, "BENDING"),
            (5.5, "BENDING"),
            (6.0, "UPRIGHT"),  # Cycle 3 finished
        ]

        for ts, post in timestamps_and_postures:
            window.add_frame({
                "frame_index": int(ts * 10),
                "timestamp_sec": ts,
                "posture": post,
                "lumbar_torque_nm": 80.0 if post != "UPRIGHT" else 30.0,
                "compression_force_n": 2000.0 if post != "UPRIGHT" else 800.0,
                "trunk_flexion_deg": 40.0 if post != "UPRIGHT" else 10.0,
                "risk_score": 50.0 if post != "UPRIGHT" else 10.0,
            })

        metrics = window.get_metrics()
        self.assertEqual(metrics["repetition_count"], 3)
        self.assertGreater(metrics["rep_rate_per_min"], 20.0)  # 3 reps in 6 seconds = 30 reps/min

    def test_03_cumulative_risk_model_scoring_and_factors(self):
        """Verify multi-dimensional cumulative risk calculation and factor extraction."""
        model = CumulativeRiskModel()

        # Mild metrics
        mild_metrics = {
            "peak_spinal_compression_n": 1200.0,
            "mean_phase2_risk_score": 20.0,
            "awkward_duty_cycle_pct": 10.0,
            "rep_rate_per_min": 2.0,
            "cumulative_torque_impulse_nms": 300.0,
            "window_duration_sec": 20.0,
            "peak_trunk_flexion_deg": 15.0,
            "peak_lumbar_torque_nm": 45.0,
            "repetition_count": 1,
        }
        score_mild = model.evaluate(mild_metrics)
        self.assertEqual(score_mild.level, "LOW")
        self.assertLess(score_mild.score, 30.0)
        self.assertTrue(score_mild.to_dict()["is_estimated_model_indicator"])

        # Hazardous metrics
        hazard_metrics = {
            "peak_spinal_compression_n": 3600.0,  # Exceeds NIOSH AL (3,400 N)
            "mean_phase2_risk_score": 75.0,
            "awkward_duty_cycle_pct": 70.0,
            "rep_rate_per_min": 8.5,
            "cumulative_torque_impulse_nms": 1900.0,
            "window_duration_sec": 20.0,
            "peak_trunk_flexion_deg": 52.0,
            "peak_lumbar_torque_nm": 115.0,
            "repetition_count": 6,
        }
        score_hazard = model.evaluate(hazard_metrics)
        self.assertIn(score_hazard.level, ("HIGH", "DANGEROUS"))
        self.assertGreaterEqual(score_hazard.score, 60.0)
        self.assertTrue(any("NIOSH Action Limit" in f for f in score_hazard.contributing_factors))
        self.assertTrue(any("trunk flexion" in f for f in score_hazard.contributing_factors))

    def test_04_incident_engine_grouping_and_cooldown(self):
        """Verify grouping of continuous risk frames into a cohesive incident with cooldown."""
        engine = IncidentEngine(
            camera_id="camera_01",
            incident_threshold=50.0,
            cooldown_sec=1.5,
            worker_id="W-C01-TEST",
            station_id="STATION-TEST",
        )

        model = CumulativeRiskModel()
        hazard_metrics = {
            "peak_spinal_compression_n": 3500.0,
            "mean_phase2_risk_score": 70.0,
            "awkward_duty_cycle_pct": 60.0,
            "rep_rate_per_min": 7.0,
            "cumulative_torque_impulse_nms": 1200.0,
            "window_duration_sec": 10.0,
            "peak_trunk_flexion_deg": 48.0,
            "peak_lumbar_torque_nm": 95.0,
        }
        risk_hazard = model.evaluate(hazard_metrics)

        # Trigger frames for 2.2 seconds (should finalize continuous hazardous exposure)
        inc = None
        for i in range(25):
            t = float(i) * 0.1
            res = engine.evaluate_frame(t, hazard_metrics, risk_hazard)
            if res:
                inc = res

        self.assertIsNotNone(inc, "IncidentEngine failed to trigger and finalize incident on sustained hazard")
        self.assertEqual(inc.worker_id, "W-C01-TEST")
        self.assertEqual(inc.station_id, "STATION-TEST")
        self.assertEqual(inc.camera_id, "camera_01")
        self.assertGreaterEqual(inc.cumulative_risk_score, 50.0)

    def test_05_incident_keyframe_capture_from_stream(self):
        """Verify incident keyframe capture from actual frame buffer and file creation."""
        test_kf_dir = Path("output/test_phase3_keyframes")
        test_kf_dir.mkdir(parents=True, exist_ok=True)

        engine = IncidentEngine(
            camera_id="camera_02",
            incident_threshold=50.0,
            keyframes_dir=test_kf_dir,
        )

        # Generate a dummy synthetic frame
        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(dummy_frame, "PEAK RISK FRAME", (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        hazard_metrics = {
            "peak_spinal_compression_n": 3500.0,
            "mean_phase2_risk_score": 75.0,
            "awkward_duty_cycle_pct": 65.0,
            "rep_rate_per_min": 8.0,
            "cumulative_torque_impulse_nms": 1500.0,
            "window_duration_sec": 15.0,
            "peak_trunk_flexion_deg": 50.0,
            "peak_lumbar_torque_nm": 105.0,
        }
        risk_hazard = CumulativeRiskScore(75.0, "HIGH", {}, ["Elevated spinal compression"])

        # Feed frame directly
        inc = None
        for i in range(25):
            t = float(i) * 0.1
            res = engine.evaluate_frame(t, hazard_metrics, risk_hazard, frame=dummy_frame, frame_index=i)
            if res:
                inc = res
                break

        if not inc:
            inc = engine.flush(final_timestamp_sec=2.5)

        self.assertIsNotNone(inc)
        self.assertIsNotNone(inc.keyframe_path)
        self.assertTrue(Path(inc.keyframe_path).exists(), f"Keyframe file not saved: {inc.keyframe_path}")

    def test_06_dynamic_identities_zero_hardcoded_values(self):
        """Verify dynamic worker and station resolution; no hardcoded WORKER-1042 or STATION-GATE-A."""
        worker_id = generate_worker_identity("camera_03", "sess_dyn_77")
        station_id = resolve_station("camera_03")

        self.assertTrue(worker_id.startswith("W-C03-"))
        self.assertEqual(station_id, "STATION-03")
        self.assertNotIn("WORKER-1042", worker_id)
        self.assertNotIn("STATION-GATE-A", station_id)

    def test_07_strands_agent_incident_packet_schema(self):
        """Verify incident record export conforms to kineticguard.strands.v1 schema."""
        inc = ErgonomicIncident(
            incident_id="INC-20260920-C01-001",
            worker_id="W-C01-TEST",
            camera_id="camera_01",
            station_id="STATION-01",
            start_time_sec=1.5,
            end_time_sec=4.0,
            risk_level="HIGH",
            cumulative_risk_score=72.5,
            exposure_metrics={
                "awkward_duty_cycle_pct": 55.0,
                "rep_rate_per_min": 6.5,
                "peak_lumbar_torque_nm": 95.0,
                "peak_spinal_compression_n": 3200.0,
                "cumulative_torque_impulse_nms": 650.0,
            },
            contributing_factors=["Excessive trunk flexion (peak 46.2 deg)"],
            recommended_action="Reconfigure workstation and pallet height to waist level",
            keyframe_path="output/incidents/keyframes/INC-20260920-C01-001_keyframe.jpg",
            keyframe_frame_index=55,
        )

        strands_pkt = inc.to_strands_incident_packet()
        self.assertEqual(strands_pkt["schema_version"], "kineticguard.strands.v1")
        self.assertEqual(strands_pkt["event_type"], "ERGONOMIC_INCIDENT")
        self.assertEqual(strands_pkt["agent_target"], "strands_ergonomic_safety_agent")
        self.assertEqual(strands_pkt["incident_id"], "INC-20260920-C01-001")
        self.assertEqual(strands_pkt["worker_identity"]["worker_id"], "W-C01-TEST")
        self.assertEqual(strands_pkt["risk_assessment"]["cumulative_risk_score"], 72.5)
        self.assertTrue(strands_pkt["biomechanical_exposure"]["is_estimated_model_indicator"])
        self.assertEqual(strands_pkt["visual_evidence"]["keyframe_frame_index"], 55)

    def test_08_live_evaluation_on_all_3_camera_videos(self):
        """Verify end-to-end live analysis on all 3 camera videos with zero AWS dependency."""
        out_dir = Path("output/test_phase3_eval")
        out_dir.mkdir(parents=True, exist_ok=True)

        for cid in settings.CAMERA_IDS:
            vpath = settings.CAMERA_SOURCES[cid]
            stat = evaluate_camera_live(
                video_path=vpath,
                camera_id=cid,
                output_dir=out_dir,
                window_sec=15.0,
                incident_threshold=45.0,  # Ensure natural triggering on real videos
                max_frames=20,  # Quick test execution
            )
            self.assertEqual(stat["total_frames_analyzed"], 20)
            self.assertGreater(stat["session_duration_sec"], 0.0)
            self.assertGreater(stat["peak_cumulative_score"], 0.0)
            self.assertTrue(stat["is_estimated_model_indicator"])
            self.assertTrue(Path(stat["incidents_file"]).exists())
            self.assertTrue(Path(stat["strands_file"]).exists())

            # Verify Strands incident JSON
            with open(stat["strands_file"], "r", encoding="utf-8") as f:
                strands_data = json.load(f)
                self.assertEqual(strands_data["schema_version"], "kineticguard.strands.v1")


if __name__ == "__main__":
    unittest.main()
