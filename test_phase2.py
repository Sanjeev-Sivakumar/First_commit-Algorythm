"""
KineticGuard - Phase 2 Test Suite: Pose Estimation & Biomechanics (Build It Architecture)
Verifies:
1. Real MediaPipe keypoint detection from Phase 1 frame packets.
2. 3D/body keypoints tracked dynamically per frame.
3. Joint-angle calculation: trunk/lumbar, shoulder, elbow, hip, knee.
4. Posture classification: UPRIGHT, BENDING, SQUATTING, LIFTING.
5. Biomechanical indicators: lumbar torque/load and compression estimate.
6. Torque/compression clearly labeled as estimated model indicators (non-medical).
7. Dynamic worker/session/camera identity (no hardcoded workers).
8. Structured telemetry output suitable for Strands Agents SDK.
9. Real analysis on all 3 camera videos with zero AWS dependency.
"""

import os
import time
import unittest
from pathlib import Path
import cv2
import numpy as np

from src.config import settings
from src.video_source import VideoSource
from src.pose_detector import PoseDetector, calculate_angle_2d
from src.biomechanics import BiomechanicsAnalyzer, BiomechanicsMetrics
from src.telemetry_exporter import TelemetryExporter
from analyze_video import analyze_single_stream


class TestPhase2BiomechanicsBuildIt(unittest.TestCase):
    """Test suite for Phase 2: Pose Estimation & Biomechanical Analysis."""

    @classmethod
    def setUpClass(cls):
        cls.detector = PoseDetector()
        cls.analyzer = BiomechanicsAnalyzer(body_weight_kg=75.0, load_weight_kg=10.0)

    def test_01_joint_angle_calculation_geometry(self):
        """Verify mathematical correctness of 2D joint angle calculations."""
        # Right angle (90 deg)
        a = (0.0, 1.0)
        b = (0.0, 0.0)
        c = (1.0, 0.0)
        angle_90 = calculate_angle_2d(a, b, c)
        self.assertAlmostEqual(angle_90, 90.0, places=1)

        # Straight angle (180 deg)
        a_line = (-1.0, 0.0)
        c_line = (1.0, 0.0)
        angle_180 = calculate_angle_2d(a_line, b, c_line)
        self.assertAlmostEqual(angle_180, 180.0, places=1)

    def test_02_real_mediapipe_detection_from_video_frame(self):
        """Verify real MediaPipe 3D pose detection on actual camera_01 video frame."""
        cap = cv2.VideoCapture(settings.CAMERA_SOURCES["camera_01"])
        ret, frame = cap.read()
        cap.release()
        self.assertTrue(ret, "Failed to read frame from camera_01")

        pose_res = self.detector.detect_pose(frame)
        self.assertTrue(pose_res.detected, "MediaPipe failed to detect pose on sample frame")
        self.assertIn("trunk_flexion_deg", pose_res.angles)
        self.assertIn("left_shoulder_deg", pose_res.angles)
        self.assertIn("left_elbow_deg", pose_res.angles)
        self.assertIn("left_hip_deg", pose_res.angles)
        self.assertIn("left_knee_deg", pose_res.angles)

        # Confirm 3D landmarks
        self.assertGreater(len(pose_res.landmarks_3d), 10)
        sample_lm = next(iter(pose_res.landmarks_3d.values()))
        self.assertIn("x", sample_lm)
        self.assertIn("y", sample_lm)
        self.assertIn("z", sample_lm)

    def test_03_posture_classification_categories(self):
        """Verify posture classification: UPRIGHT, BENDING, SQUATTING, LIFTING."""
        # Case A: Neutral upright
        angles_upright = {
            "trunk_flexion_deg": 10.0,
            "left_knee_deg": 175.0,
            "right_knee_deg": 175.0,
            "left_hip_deg": 170.0,
            "right_hip_deg": 170.0,
            "left_elbow_deg": 160.0,
            "right_elbow_deg": 160.0,
        }
        res_upright = self.analyzer.analyze(angles_upright)
        self.assertEqual(res_upright.posture, "UPRIGHT")

        # Case B: Significant bending
        angles_bending = dict(angles_upright)
        angles_bending["trunk_flexion_deg"] = 45.0
        res_bending = self.analyzer.analyze(angles_bending)
        self.assertEqual(res_bending.posture, "BENDING")

        # Case C: Deep squatting
        angles_squat = dict(angles_upright)
        angles_squat["left_knee_deg"] = 95.0
        angles_squat["right_knee_deg"] = 95.0
        res_squat = self.analyzer.analyze(angles_squat)
        self.assertIn(res_squat.posture, ("SQUATTING", "LIFTING"))

    def test_04_biomechanical_indicators_and_non_medical_labeling(self):
        """Verify torque/compression calculation and non-medical estimated indicator labeling."""
        angles = {"trunk_flexion_deg": 35.0, "left_knee_deg": 170.0, "right_knee_deg": 170.0}
        bio = self.analyzer.analyze(angles)

        self.assertGreater(bio.lumbar_torque_nm, 0.0)
        self.assertGreater(bio.compression_force_n, 500.0)

        # Confirm non-medical labeling
        self.assertTrue(bio.is_estimated_model_indicator)
        self.assertIn("Estimated biomechanical model indicator", bio.disclaimer)

        bio_dict = bio.to_dict()
        self.assertTrue(bio_dict["is_estimated_model_indicator"])
        self.assertIn("disclaimer", bio_dict)

    def test_05_dynamic_identities_zero_hardcoded_workers(self):
        """Verify dynamic worker and session identity; ensure no hardcoded WORKER-1042."""
        session_id = "sess_test_dynamic_99"
        exporter = TelemetryExporter(camera_id="camera_02", session_id=session_id)
        self.assertEqual(exporter.session_id, session_id)
        self.assertTrue(exporter.worker_id.startswith("W-C02-"))
        self.assertNotIn("WORKER-1042", exporter.worker_id)

        rec = exporter.record_frame(
            frame_index=1,
            timestamp_sec=0.04,
            pose_detected=True,
            angles={"left_shoulder_deg": 30.0},
            biomech_dict={"posture": "UPRIGHT", "lumbar_torque_nm": 45.0, "compression_force_n": 1200.0},
        )
        self.assertEqual(rec["worker_id"], exporter.worker_id)
        self.assertEqual(rec["session_id"], session_id)
        self.assertTrue(rec["is_estimated_model_indicator"])

    def test_06_strands_agent_packet_compatibility(self):
        """Verify telemetry formatting conforms to Strands Agents SDK event structure."""
        exporter = TelemetryExporter(camera_id="camera_01", session_id="sess_strands_01")
        rec = exporter.record_frame(
            frame_index=10,
            timestamp_sec=0.40,
            pose_detected=True,
            angles={"trunk_flexion_deg": 28.5, "left_knee_deg": 170.0},
            biomech_dict={
                "posture": "BENDING",
                "lumbar_torque_nm": 78.4,
                "compression_force_n": 1950.0,
                "risk_level": "MODERATE",
                "risk_score": 38.0,
            },
        )

        strands_packet = exporter.to_strands_agent_packet(rec)
        self.assertEqual(strands_packet["schema_version"], "kineticguard.strands.v1")
        self.assertEqual(strands_packet["agent_target"], "strands_ergonomic_agent")
        self.assertEqual(strands_packet["posture"], "BENDING")
        self.assertTrue(strands_packet["biomechanical_indicators"]["is_estimated_model_indicator"])
        self.assertIn("disclaimer", strands_packet["biomechanical_indicators"])
        self.assertIn("joint_angles", strands_packet)
        self.assertIn("risk_assessment", strands_packet)

    def test_07_phase1_frame_packet_to_phase2_processing(self):
        """Verify direct consumption of Phase 1 frame packets by PoseDetector."""
        vs = VideoSource(
            video_path=settings.CAMERA_SOURCES["camera_01"],
            camera_id="camera_01",
            target_fps=25,
            loop=False,
        )
        gen = vs.get_frame_packet_generator("sess_p1_to_p2")
        packet = next(gen)
        vs.close()

        pose_res, returned_packet = self.detector.process_frame_packet(packet)
        self.assertTrue(pose_res.detected)
        self.assertEqual(returned_packet["session_id"], "sess_p1_to_p2")

    def test_08_real_analysis_on_all_3_camera_videos(self):
        """Run real analysis on camera_01, camera_02, and camera_03 with zero AWS credentials."""
        out_test_dir = Path("output/test_phase2_telemetry")
        out_test_dir.mkdir(parents=True, exist_ok=True)

        for cam_id in settings.CAMERA_IDS:
            vpath = settings.CAMERA_SOURCES[cam_id]
            summary = analyze_single_stream(
                video_path=vpath,
                camera_id=cam_id,
                output_dir=out_test_dir,
                max_frames=15,  # 15 frames for quick multi-camera test
                save_video=False,
            )
            self.assertEqual(summary["total_frames"], 15)
            self.assertGreater(summary["detected_frames"], 0)
            self.assertGreater(summary["mean_lumbar_torque_nm"], 0.0)
            self.assertGreater(summary["mean_compression_force_n"], 0.0)
            self.assertEqual(summary["indicator_type"], "ESTIMATED_MODEL_INDICATOR")
            self.assertTrue(Path(summary["output_json"]).exists())
            self.assertTrue(Path(summary["output_csv"]).exists())


if __name__ == "__main__":
    unittest.main()
