"""
KineticGuard - Biomechanics Analysis Module
Classifies ergonomic postures (Upright, Bending, Squatting, Lifting) and calculates
L5/S1 lumbar moment torque and spinal compressive loading based on human body geometry.
"""

import math
from typing import Dict, Optional, Tuple
import numpy as np


class BiomechanicsMetrics:
    """Holds computed posture classification and biomechanical loading telemetry."""

    def __init__(
        self,
        posture: str = "UNKNOWN",
        trunk_flexion_deg: float = 0.0,
        lumbar_torque_nm: float = 0.0,
        compression_force_n: float = 0.0,
        torso_lever_arm_m: float = 0.0,
        load_lever_arm_m: float = 0.0,
        risk_level: str = "LOW",
        risk_score: float = 0.0,
        details: Optional[Dict[str, float]] = None,
    ):
        self.posture = posture
        self.trunk_flexion_deg = trunk_flexion_deg
        self.lumbar_torque_nm = lumbar_torque_nm
        self.compression_force_n = compression_force_n
        self.torso_lever_arm_m = torso_lever_arm_m
        self.load_lever_arm_m = load_lever_arm_m
        self.risk_level = risk_level
        self.risk_score = risk_score
        self.details = details or {}
        self.is_estimated_model_indicator = True
        self.disclaimer = (
            "Estimated biomechanical model indicator based on geometric posture "
            "and gravity loading; not a direct in-vivo medical measurement."
        )

    def to_dict(self) -> dict:
        return {
            "posture": self.posture,
            "trunk_flexion_deg": round(self.trunk_flexion_deg, 1),
            "lumbar_torque_nm": round(self.lumbar_torque_nm, 1),
            "compression_force_n": round(self.compression_force_n, 1),
            "torso_lever_arm_m": round(self.torso_lever_arm_m, 2),
            "load_lever_arm_m": round(self.load_lever_arm_m, 2),
            "risk_level": self.risk_level,
            "risk_score": round(self.risk_score, 1),
            "is_estimated_model_indicator": True,
            "disclaimer": self.disclaimer,
        }


class BiomechanicsAnalyzer:
    """
    Analyzes body landmark geometry and calculates ergonomic posture and
    lumbar torque (L5/S1) under gravitational and manual material handling loads.
    """

    def __init__(
        self,
        body_weight_kg: float = 75.0,
        load_weight_kg: float = 10.0,
        torso_length_m: float = 0.50,
    ):
        self.body_weight_kg = body_weight_kg
        self.load_weight_kg = load_weight_kg
        self.torso_length_m = torso_length_m

        # Biomechanical mass fractions (Dempster anthropometric model)
        self.g = 9.81
        self.trunk_mass_kg = 0.45 * self.body_weight_kg
        self.trunk_weight_n = self.trunk_mass_kg * self.g
        self.load_weight_n = self.load_weight_kg * self.g

        # Muscle lever arm of erector spinae back muscles at L5/S1 (~5 cm)
        self.erector_spinae_arm_m = 0.05

    def analyze(
        self,
        angles: Dict[str, float],
        landmarks_3d: Optional[Dict[str, Dict[str, float]]] = None,
    ) -> BiomechanicsMetrics:
        """
        Computes posture classification and biomechanical torque for a given frame.
        """
        trunk_deg = angles.get("trunk_flexion_deg", 0.0)
        left_knee = angles.get("left_knee_deg", 175.0)
        right_knee = angles.get("right_knee_deg", 175.0)
        avg_knee = (left_knee + right_knee) / 2.0

        left_hip = angles.get("left_hip_deg", 170.0)
        right_hip = angles.get("right_hip_deg", 170.0)
        avg_hip = (left_hip + right_hip) / 2.0

        left_elbow = angles.get("left_elbow_deg", 160.0)
        right_elbow = angles.get("right_elbow_deg", 160.0)
        avg_elbow = (left_elbow + right_elbow) / 2.0

        # 1. Posture Classification
        posture = self._classify_posture(
            trunk_deg, avg_knee, avg_hip, avg_elbow, landmarks_3d
        )

        # 2. Lever Arm Estimation
        torso_lever_arm, load_lever_arm = self._calculate_lever_arms(
            trunk_deg, landmarks_3d
        )

        # 3. Lumbar Torque Calculation (L5/S1 Moment in Nm)
        # M_trunk = F_trunk * d_trunk
        # M_load = F_load * d_load (active when lifting or holding)
        is_load_active = posture in ("LIFTING", "BENDING", "SQUATTING")
        active_load_n = self.load_weight_n if is_load_active else (0.2 * self.load_weight_n)

        torque_trunk = self.trunk_weight_n * torso_lever_arm
        torque_load = active_load_n * load_lever_arm
        total_lumbar_torque = max(0.0, torque_trunk + torque_load)

        # 4. Spinal Compressive Force Estimation (N)
        # F_comp = (M_lumbar / d_erector) + (F_trunk + F_load) * cos(theta)
        trunk_rad = math.radians(trunk_deg)
        cos_theta = max(0.0, math.cos(trunk_rad))
        muscle_force = total_lumbar_torque / self.erector_spinae_arm_m
        body_gravity_comp = (self.trunk_weight_n + active_load_n) * cos_theta
        compression_force = muscle_force + body_gravity_comp

        # 5. Instantaneous Frame Risk Categorization
        # Single frames represent instantaneous posture and loading intensity.
        # Core Rule: Do NOT classify HIGH risk from a single frame.
        # Single frames are categorized as LOW or MODERATE potential.
        # HIGH and DANGEROUS classifications are strictly authoritative in the Temporal Cumulative Engine.
        if total_lumbar_torque < 85.0 and compression_force < 2200.0:
            risk_level = "LOW"
            risk_score = min(20.0, (total_lumbar_torque / 85.0) * 20.0)
        elif total_lumbar_torque < 180.0 and compression_force < 3400.0:
            risk_level = "MODERATE"
            risk_score = 20.0 + ((total_lumbar_torque - 85.0) / 95.0) * 15.0
        else:
            risk_level = "MODERATE"
            risk_score = min(45.0, 35.0 + ((total_lumbar_torque - 180.0) / 100.0) * 10.0)

        return BiomechanicsMetrics(
            posture=posture,
            trunk_flexion_deg=trunk_deg,
            lumbar_torque_nm=total_lumbar_torque,
            compression_force_n=compression_force,
            torso_lever_arm_m=torso_lever_arm,
            load_lever_arm_m=load_lever_arm,
            risk_level=risk_level,
            risk_score=risk_score,
            details={
                "torque_trunk_nm": round(torque_trunk, 1),
                "torque_load_nm": round(torque_load, 1),
                "avg_knee_deg": round(avg_knee, 1),
                "avg_hip_deg": round(avg_hip, 1),
                "avg_elbow_deg": round(avg_elbow, 1),
            },
        )

    def _classify_posture(
        self,
        trunk_deg: float,
        avg_knee: float,
        avg_hip: float,
        avg_elbow: float,
        landmarks: Optional[Dict[str, Dict[str, float]]],
    ) -> str:
        """
        Differentiates between UPRIGHT, BENDING, SQUATTING, and LIFTING postures.
        Protects against false positives from hand/leg shaking or random movement while standing upright.
        """
        # Check hand position relative to hips
        hands_forward = False
        hands_low = False

        if landmarks:
            lw = landmarks.get("left_wrist")
            rw = landmarks.get("right_wrist")
            lh = landmarks.get("left_hip")
            rh = landmarks.get("right_hip")

            if (lw or rw) and (lh or rh):
                wrist_y = min([w["y"] for w in (lw, rw) if w])
                hip_y = min([h["y"] for h in (lh, rh) if h])
                wrist_x = np.mean([w["x"] for w in (lw, rw) if w])
                hip_x = np.mean([h["x"] for h in (lh, rh) if h])

                # Hands are low near knees or lower (below hip level by > 15% body frame)
                if wrist_y > hip_y + 0.15:
                    hands_low = True
                # Hands extended forward horizontally away from torso
                if abs(wrist_x - hip_x) > 0.18:
                    hands_forward = True

        # Rule 1: Standing Upright Protection (Movement != Risk)
        # If trunk is upright (< 25 deg) and knees are extended (> 135 deg),
        # hand shaking, gesturing, or leg twitches remain strictly UPRIGHT.
        if trunk_deg < 25.0 and avg_knee > 135.0:
            if trunk_deg >= 18.0 and avg_elbow < 90.0 and hands_forward:
                return "LIFTING"
            return "UPRIGHT"

        # Rule 2: SQUATTING: Deep knee bend (knee angle <= 115 deg)
        if avg_knee <= 115.0:
            if hands_low or avg_elbow < 120.0:
                return "LIFTING"
            return "SQUATTING"

        # Rule 3: BENDING: Significant trunk flexion (>= 30 deg) with relatively straight legs
        if trunk_deg >= 30.0 and avg_knee > 120.0:
            if hands_low or hands_forward or avg_elbow < 110.0:
                return "LIFTING"
            return "BENDING"

        # Rule 4: Moderate stoop (25.0 - 30.0 deg)
        if trunk_deg >= 25.0:
            if hands_low or (hands_forward and avg_elbow < 110.0):
                return "LIFTING"
            return "BENDING"

        return "UPRIGHT"

    def _calculate_lever_arms(
        self,
        trunk_deg: float,
        landmarks: Optional[Dict[str, Dict[str, float]]],
    ) -> Tuple[float, float]:
        """
        Calculates physical horizontal moment arms (meters) for trunk and hand load.
        """
        trunk_rad = math.radians(trunk_deg)

        # Torso center-of-mass is located at ~55% of torso length from hip
        trunk_com_length = 0.55 * self.torso_length_m
        torso_lever_arm = max(0.04, trunk_com_length * abs(math.sin(trunk_rad)))

        # Load lever arm (horizontal distance from hips to hands)
        load_lever_arm = 0.25  # default neutral arm distance (~25 cm)

        if landmarks:
            lw = landmarks.get("left_wrist")
            rw = landmarks.get("right_wrist")
            lh = landmarks.get("left_hip")
            rh = landmarks.get("right_hip")

            if (lw or rw) and (lh or rh):
                wrist_x = np.mean([w["x"] for w in (lw, rw) if w])
                hip_x = np.mean([h["x"] for h in (lh, rh) if h])

                # Normalized coordinates (0 to 1 across frame width ~2.0m FOV)
                estimated_dx_m = abs(wrist_x - hip_x) * 2.0
                load_lever_arm = max(0.15, min(0.75, estimated_dx_m))
        else:
            # Estimate from trunk angle if landmarks absent
            load_lever_arm = 0.20 + (self.torso_length_m * abs(math.sin(trunk_rad)))

        return torso_lever_arm, load_lever_arm
