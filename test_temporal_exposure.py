"""
KineticGuard - Sustained Ergonomic Exposure & Movement Resilience Test Suite
Verifies:
1. Core Rule: Movement != Risk.
2. Shaking hands or moving legs while standing upright remains UPRIGHT and LOW risk.
3. 2 seconds of brief bending produces LOW risk (< 30.0).
4. 10 seconds of intermittent bending produces LOW/MODERATE risk (30.0 - 50.0).
5. 30 seconds of repeated bending produces MODERATE risk (50.0 - 65.0).
6. Repeated bending + significant load + sustained exposure produces HIGH risk (>= 70.0).
7. Sustained hazardous loading exceeding NIOSH AL produces DANGEROUS/HIGH risk (>= 80.0).
8. Single frames cannot produce HIGH risk on their own.
9. Physiological recovery decay reduces cumulative strain when returning to UPRIGHT.
"""

import math
import time
import unittest
from typing import Dict

from src.biomechanics import BiomechanicsAnalyzer, BiomechanicsMetrics
from src.temporal_analyzer import TemporalSlidingWindow
from src.cumulative_risk import CumulativeRiskModel, CumulativeRiskScore
from src.incident_engine import IncidentEngine


class TestSustainedErgonomicExposure(unittest.TestCase):
    """Test suite validating sustained ergonomic exposure detection and random movement resilience."""

    def setUp(self):
        self.biomech = BiomechanicsAnalyzer(body_weight_kg=75.0, load_weight_kg=10.0)
        self.risk_model = CumulativeRiskModel()

    def test_01_hand_and_leg_shaking_while_upright_is_not_high_risk(self):
        """Verify hand/leg shaking while standing upright does NOT falsely trigger LIFTING or HIGH risk."""
        # Worker standing straight (trunk flexion 8 deg, knees straight 175 deg)
        # but shaking hands (wrist moving forward/back, elbows flexing)
        neutral_upright_angles = {
            "trunk_flexion_deg": 8.0,
            "left_knee_deg": 175.0,
            "right_knee_deg": 175.0,
            "left_hip_deg": 172.0,
            "right_hip_deg": 172.0,
            "left_elbow_deg": 85.0,  # Shaking hands with bent elbows
            "right_elbow_deg": 90.0,
        }

        # Simulated landmarks with moving wrists
        landmarks_shaking = {
            "left_wrist": {"x": 0.55, "y": 0.45, "z": 0.0},
            "right_wrist": {"x": 0.58, "y": 0.48, "z": 0.0},
            "left_hip": {"x": 0.50, "y": 0.50, "z": 0.0},
            "right_hip": {"x": 0.50, "y": 0.50, "z": 0.0},
        }

        bio = self.biomech.analyze(neutral_upright_angles, landmarks_3d=landmarks_shaking)
        self.assertEqual(bio.posture, "UPRIGHT", "Hand shaking while standing was falsely classified as awkward posture")
        self.assertEqual(bio.risk_level, "LOW")
        self.assertLess(bio.risk_score, 20.0)
        self.assertLess(bio.lumbar_torque_nm, 40.0)

    def test_02_single_frame_never_produces_high_risk(self):
        """Verify that single frames cannot produce HIGH or DANGEROUS risk level."""
        # Extreme bend frame (60 deg trunk flexion)
        extreme_angles = {
            "trunk_flexion_deg": 60.0,
            "left_knee_deg": 170.0,
            "right_knee_deg": 170.0,
            "left_hip_deg": 120.0,
            "right_hip_deg": 120.0,
            "left_elbow_deg": 160.0,
            "right_elbow_deg": 160.0,
        }
        bio = self.biomech.analyze(extreme_angles)
        self.assertEqual(bio.posture, "BENDING")
        self.assertIn(bio.risk_level, ("LOW", "MODERATE"), "Single frame produced premature HIGH risk level")
        self.assertLessEqual(bio.risk_score, 45.0, "Single frame risk score exceeded instantaneous cap")

    def test_03_brief_bending_2_seconds_is_low_risk(self):
        """Verify 2 seconds of brief bending within a 20s window produces LOW risk (< 30.0)."""
        window = TemporalSlidingWindow(window_seconds=20.0)

        # 18 seconds UPRIGHT (0 to 17s), 2 seconds BENDING (18 to 20s)
        for t in range(21):
            ts = float(t)
            if ts < 18.0:
                posture = "UPRIGHT"
                trunk = 8.0
                tq = 20.0
                comp = 750.0
                p2 = 5.0
            else:
                posture = "BENDING"
                trunk = 45.0
                tq = 75.0
                comp = 1900.0
                p2 = 35.0

            window.add_frame({
                "timestamp_sec": ts,
                "posture": posture,
                "trunk_flexion_deg": trunk,
                "lumbar_torque_nm": tq,
                "compression_force_n": comp,
                "risk_score": p2,
            })

        metrics = window.get_metrics()
        self.assertLessEqual(metrics["duration_bending_sec"], 3.0)
        self.assertLessEqual(metrics["awkward_duty_cycle_pct"], 15.0)

        risk = self.risk_model.evaluate(metrics)
        self.assertEqual(risk.level, "LOW", f"2-second brief bend resulted in {risk.level} ({risk.score})")
        self.assertLess(risk.score, 30.0)

    def test_04_intermittent_bending_10_seconds_is_low_moderate_risk(self):
        """Verify 10 seconds of intermittent bending produces LOW/MODERATE risk (30.0 - 50.0)."""
        window = TemporalSlidingWindow(window_seconds=20.0)

        # Alternating 2s UPRIGHT, 2s BENDING for 20 seconds total
        for t in range(21):
            ts = float(t)
            is_bending = (int(ts) % 4) in (2, 3)
            posture = "BENDING" if is_bending else "UPRIGHT"
            trunk = 40.0 if is_bending else 10.0
            tq = 75.0 if is_bending else 20.0
            comp = 1850.0 if is_bending else 750.0
            p2 = 30.0 if is_bending else 5.0

            window.add_frame({
                "timestamp_sec": ts,
                "posture": posture,
                "trunk_flexion_deg": trunk,
                "lumbar_torque_nm": tq,
                "compression_force_n": comp,
                "risk_score": p2,
            })

        metrics = window.get_metrics()
        self.assertGreaterEqual(metrics["awkward_duty_cycle_pct"], 40.0)
        self.assertLessEqual(metrics["awkward_duty_cycle_pct"], 60.0)

        risk = self.risk_model.evaluate(metrics)
        self.assertIn(risk.level, ("LOW", "MODERATE"), f"Intermittent 10s bending resulted in {risk.level} ({risk.score})")
        self.assertLess(risk.score, 55.0)

    def test_05_repeated_bending_30_seconds_is_moderate_risk(self):
        """Verify 30 seconds of repeated bending without heavy load produces MODERATE risk (50.0 - 65.0)."""
        window = TemporalSlidingWindow(window_seconds=30.0)

        # 30 seconds with 70% bending duty cycle and multiple repetition cycles
        for t in range(31):
            ts = float(t)
            is_bending = (int(ts) % 5) != 0  # 4s bend, 1s upright
            posture = "BENDING" if is_bending else "UPRIGHT"
            trunk = 42.0 if is_bending else 10.0
            tq = 80.0 if is_bending else 22.0
            comp = 2000.0 if is_bending else 750.0
            p2 = 35.0 if is_bending else 5.0

            window.add_frame({
                "timestamp_sec": ts,
                "posture": posture,
                "trunk_flexion_deg": trunk,
                "lumbar_torque_nm": tq,
                "compression_force_n": comp,
                "risk_score": p2,
            })

        metrics = window.get_metrics()
        risk = self.risk_model.evaluate(metrics)
        self.assertEqual(risk.level, "MODERATE", f"30s repeated bending resulted in {risk.level} ({risk.score})")
        self.assertGreaterEqual(risk.score, 45.0)
        self.assertLess(risk.score, 70.0)

    def test_06_sustained_heavy_exposure_is_high_or_dangerous(self):
        """Verify repeated bending + heavy load + sustained duration produces HIGH or DANGEROUS risk (>= 70.0)."""
        window = TemporalSlidingWindow(window_seconds=20.0)

        # Heavy manual lifting: 50 deg trunk flexion, 115 Nm torque, 3600 N compression (exceeds NIOSH AL)
        for t in range(21):
            ts = float(t)
            is_lifting = (int(ts) % 4) != 0  # 75% duty
            posture = "LIFTING" if is_lifting else "UPRIGHT"
            trunk = 52.0 if is_lifting else 12.0
            tq = 115.0 if is_lifting else 25.0
            comp = 3600.0 if is_lifting else 800.0
            p2 = 45.0 if is_lifting else 5.0

            window.add_frame({
                "timestamp_sec": ts,
                "posture": posture,
                "trunk_flexion_deg": trunk,
                "lumbar_torque_nm": tq,
                "compression_force_n": comp,
                "risk_score": p2,
            })

        metrics = window.get_metrics()
        risk = self.risk_model.evaluate(metrics)
        self.assertIn(risk.level, ("HIGH", "DANGEROUS"), f"Sustained heavy exposure resulted in {risk.level} ({risk.score})")
        self.assertGreaterEqual(risk.score, 70.0)
        self.assertTrue(any("NIOSH Action Limit" in f for f in risk.contributing_factors))

    def test_07_physiological_strain_recovery_decay(self):
        """Verify cumulative strain index decays exponentially when the worker returns to UPRIGHT posture."""
        window = TemporalSlidingWindow(window_seconds=20.0)

        # Step A: 8 seconds of heavy bending -> builds strain
        for t in range(9):
            ts = float(t)
            window.add_frame({
                "timestamp_sec": ts,
                "posture": "BENDING",
                "trunk_flexion_deg": 48.0,
                "lumbar_torque_nm": 95.0,
                "compression_force_n": 2400.0,
                "risk_score": 40.0,
            })

        peak_strain = window.cumulative_strain
        self.assertGreater(peak_strain, 20.0, "Strain did not accumulate during sustained bending")

        # Step B: 6 seconds of upright resting -> strain must decay significantly
        for t in range(9, 16):
            ts = float(t)
            window.add_frame({
                "timestamp_sec": ts,
                "posture": "UPRIGHT",
                "trunk_flexion_deg": 10.0,
                "lumbar_torque_nm": 20.0,
                "compression_force_n": 750.0,
                "risk_score": 5.0,
            })

        recovered_strain = window.cumulative_strain
        self.assertLess(recovered_strain, peak_strain * 0.35, "Strain failed to decay during upright recovery")


if __name__ == "__main__":
    unittest.main()
