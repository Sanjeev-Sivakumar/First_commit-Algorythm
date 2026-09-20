"""
KineticGuard - Ergonomic Dataset Generator (Phase 4)
Generates a curated 6-class local ergonomic dataset for Qwen2-VL LoRA fine-tuning:
1. UPRIGHT: Neutral standing posture, low biomechanical strain.
2. BENDING: Forward stoop trunk flexion, increased lumbar moment arm.
3. LIFTING: Manual handling of loads, lumbar flexion and knee engagement.
4. AWKWARD_POSTURE: Severe torso flexion or extended reaching.
5. REPETITIVE_MOTION: High frequency repetitive handling with elevated duty cycle.
6. HIGH_RISK_LIFTING: Heavy load lifted with excessive trunk flexion breaching NIOSH action limits.
"""

import json
import math
import os
from pathlib import Path
from typing import Dict, List, Any
import cv2
import numpy as np


class ErgonomicDatasetGenerator:
    """Generates synthetic visual keyframes and multimodal ChatML training pairs."""

    def __init__(self, output_dir: str = "data/ergonomic_dataset"):
        self.output_dir = Path(output_dir)
        self.images_dir = self.output_dir / "images"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.images_dir.mkdir(parents=True, exist_ok=True)

        self.width = 640
        self.height = 480

    def generate_all(self, num_samples_per_class: int = 2) -> Dict[str, Any]:
        """Generates all 6 classes of ergonomic training samples."""
        categories = [
            "UPRIGHT",
            "BENDING",
            "LIFTING",
            "AWKWARD_POSTURE",
            "REPETITIVE_MOTION",
            "HIGH_RISK_LIFTING",
        ]

        dataset = []
        sample_idx = 1

        for cat in categories:
            for variation in range(num_samples_per_class):
                sample_id = f"ERGO-{sample_idx:03d}"
                sample_data = self._generate_sample(sample_id, cat, variation)
                dataset.append(sample_data)
                sample_idx += 1

        # Write dataset JSON
        dataset_path = self.output_dir / "annotations.json"
        with open(dataset_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "version": "1.0",
                    "system": "KineticGuard Ergonomic Dataset",
                    "total_samples": len(dataset),
                    "categories": categories,
                    "samples": dataset,
                },
                f,
                indent=2,
            )

        # Also write ChatML format for direct HuggingFace SFTTrainer/Qwen fine-tuning
        chatml_path = self.output_dir / "chatml_dataset.json"
        chatml_records = [s["conversational_data"] for s in dataset]
        with open(chatml_path, "w", encoding="utf-8") as f:
            json.dump(chatml_records, f, indent=2)

        return {
            "total_samples": len(dataset),
            "annotations_file": str(dataset_path),
            "chatml_file": str(chatml_path),
            "images_dir": str(self.images_dir),
        }

    def _generate_sample(
        self, sample_id: str, category: str, variation: int
    ) -> Dict[str, Any]:
        """Generates a single multimodal training pair."""
        # Set kinematics and telemetry based on category & variation
        cfg = self._get_category_config(category, variation)

        # Render visual frame
        image_filename = f"{sample_id}_{category.lower()}.jpg"
        image_path = self.images_dir / image_filename
        self._render_ergonomic_scene(
            str(image_path),
            cfg["trunk_angle_deg"],
            cfg["knee_angle_deg"],
            cfg["elbow_angle_deg"],
            cfg["has_load"],
            cfg["load_distance_cm"],
            cfg["hazard_title"],
            sample_id,
        )

        # Authoritative telemetry dictionary
        telemetry = {
            "posture_class": cfg["posture_class"],
            "trunk_flexion_deg": cfg["trunk_angle_deg"],
            "knee_angle_deg": cfg["knee_angle_deg"],
            "elbow_angle_deg": cfg["elbow_angle_deg"],
            "load_weight_kg": cfg["load_weight_kg"],
            "load_distance_cm": cfg["load_distance_cm"],
            "lumbar_torque_nm": cfg["lumbar_torque_nm"],
            "spinal_compression_n": cfg["spinal_compression_n"],
            "awkward_duty_cycle_pct": cfg["awkward_duty_cycle_pct"],
            "rep_rate_per_min": cfg["rep_rate_per_min"],
            "cumulative_risk_score": cfg["cumulative_risk_score"],
            "risk_level": cfg["risk_level"],
        }

        # Ground truth structured assistant output
        assistant_output = {
            "posture": cfg["posture_class"],
            "risk_factors": cfg["risk_factors"],
            "ergonomic_observation": cfg["ergonomic_observation"],
            "root_cause": cfg["root_cause"],
            "recommended_action": cfg["recommended_action"],
            "confidence": cfg["confidence"],
        }

        # Format ChatML conversation for Qwen2-VL instruction tuning
        conversational_data = {
            "id": sample_id,
            "category": category,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are KineticGuard Vision-Language AI, an expert industrial ergonomist specializing "
                        "in biomechanics and OSHA/NIOSH standards. Analyze the provided worker keyframe and "
                        "authoritative Phase 3 telemetry, and output strict structured JSON matching the schema: "
                        "posture, risk_factors, ergonomic_observation, root_cause, recommended_action, confidence."
                    ),
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": str(image_path)},
                        {
                            "type": "text",
                            "text": (
                                f"Analyze worker posture and ergonomic risk for keyframe {sample_id}.\n"
                                f"Authoritative Telemetry:\n"
                                f"- Posture Detected: {cfg['posture_class']}\n"
                                f"- Trunk Flexion: {cfg['trunk_angle_deg']:.1f} deg\n"
                                f"- Knee Angle: {cfg['knee_angle_deg']:.1f} deg\n"
                                f"- Load Moment Arm: {cfg['load_distance_cm']:.1f} cm\n"
                                f"- L5/S1 Lumbar Torque: {cfg['lumbar_torque_nm']:.1f} Nm\n"
                                f"- Spinal Compression: {cfg['spinal_compression_n']:.1f} N\n"
                                f"- Awkward Duty Cycle: {cfg['awkward_duty_cycle_pct']:.1f}%\n"
                                f"- Repetition Rate: {cfg['rep_rate_per_min']:.1f} reps/min\n"
                                f"- Cumulative Risk Score: {cfg['cumulative_risk_score']:.1f} ({cfg['risk_level']})\n\n"
                                f"Provide the structured ergonomic diagnosis JSON."
                            ),
                        },
                    ],
                },
                {
                    "role": "assistant",
                    "content": json.dumps(assistant_output, indent=2),
                },
            ],
        }

        return {
            "sample_id": sample_id,
            "category": category,
            "image_path": str(image_path),
            "telemetry_context": telemetry,
            "ground_truth_output": assistant_output,
            "conversational_data": conversational_data,
        }

    def _get_category_config(self, category: str, var: int) -> Dict[str, Any]:
        """Provides realistic biomechanical metrics and expert reasoning for each scenario."""
        if category == "UPRIGHT":
            trunk = 4.5 + var * 2.0
            knee = 175.0 - var * 2.0
            torque = 22.0 + var * 5.0
            comp = 920.0 + var * 40.0
            return {
                "posture_class": "UPRIGHT",
                "trunk_angle_deg": trunk,
                "knee_angle_deg": knee,
                "elbow_angle_deg": 160.0,
                "has_load": False,
                "load_weight_kg": 0.0,
                "load_distance_cm": 15.0,
                "lumbar_torque_nm": torque,
                "spinal_compression_n": comp,
                "awkward_duty_cycle_pct": 8.5 + var * 3.0,
                "rep_rate_per_min": 0.0,
                "cumulative_risk_score": 12.0 + var * 3.0,
                "risk_level": "LOW",
                "hazard_title": "NEUTRAL STANDING",
                "risk_factors": [
                    "Slight prolonged static standing posture",
                ],
                "ergonomic_observation": (
                    f"The operator maintains an upright neutral spinal alignment with negligible torso flexion "
                    f"({trunk:.1f} deg). L5/S1 spinal compressive loading remains well within safe physiological limits "
                    f"({comp:.0f} N vs 3400 N NIOSH threshold)."
                ),
                "root_cause": (
                    "Workstation height is well-aligned with elbow height for inspection tasks."
                ),
                "recommended_action": (
                    "Provide anti-fatigue floor matting to alleviate static leg and plantar pressure."
                ),
                "confidence": 0.98,
            }

        elif category == "BENDING":
            trunk = 38.0 + var * 6.0
            knee = 168.0 - var * 4.0
            torque = 88.0 + var * 14.0
            comp = 2150.0 + var * 200.0
            return {
                "posture_class": "BENDING",
                "trunk_angle_deg": trunk,
                "knee_angle_deg": knee,
                "elbow_angle_deg": 145.0,
                "has_load": False,
                "load_weight_kg": 0.0,
                "load_distance_cm": 35.0,
                "lumbar_torque_nm": torque,
                "spinal_compression_n": comp,
                "awkward_duty_cycle_pct": 72.0 + var * 10.0,
                "rep_rate_per_min": 6.0 + var * 2.0,
                "cumulative_risk_score": 62.0 + var * 8.0,
                "risk_level": "HIGH",
                "hazard_title": "FORWARD STOOP BEND",
                "risk_factors": [
                    f"Pronounced torso forward flexion ({trunk:.1f} deg)",
                    "Substantial trunk center-of-mass moment arm exceeding 30 cm",
                    "Elevated L5/S1 spinal compressive force",
                ],
                "ergonomic_observation": (
                    f"The worker is stooping forward ({trunk:.1f} deg flexion) without bending the knees "
                    f"({knee:.1f} deg), resulting in significant posterior chain strain and an elevated lumbar moment "
                    f"arm of {torque:.1f} Nm."
                ),
                "root_cause": (
                    "Parts staging bin is positioned below knuckle height (approx. 45 cm off floor), "
                    "forcing continuous forward spinal inclination."
                ),
                "recommended_action": (
                    "Elevate staging bin to waist level (75-90 cm) using an adjustable mechanical stand."
                ),
                "confidence": 0.95,
            }

        elif category == "LIFTING":
            trunk = 32.0 + var * 4.0
            knee = 130.0 - var * 8.0
            torque = 145.0 + var * 20.0
            comp = 2950.0 + var * 250.0
            return {
                "posture_class": "LIFTING",
                "trunk_angle_deg": trunk,
                "knee_angle_deg": knee,
                "elbow_angle_deg": 110.0,
                "has_load": True,
                "load_weight_kg": 10.0 + var * 3.0,
                "load_distance_cm": 42.0 + var * 4.0,
                "lumbar_torque_nm": torque,
                "spinal_compression_n": comp,
                "awkward_duty_cycle_pct": 68.0 + var * 6.0,
                "rep_rate_per_min": 8.5 + var * 2.0,
                "cumulative_risk_score": 68.0 + var * 6.0,
                "risk_level": "HIGH",
                "hazard_title": "MANUAL CRATE LIFT",
                "risk_factors": [
                    f"Extended load distance from spine ({42.0 + var * 4.0:.1f} cm)",
                    f"Simultaneous trunk flexion ({trunk:.1f} deg) and moderate knee bend",
                    f"Spinal compressive force approaching NIOSH action limit ({comp:.0f} N)",
                ],
                "ergonomic_observation": (
                    f"The operator is lifting a parcel with hands extended away from the torso ({42.0 + var * 4.0:.1f} cm), "
                    f"amplifying the load moment arm and producing {torque:.1f} Nm of lumbar moment."
                ),
                "root_cause": (
                    "Deep conveyor clearance barrier prevents worker from keeping the load close to the body."
                ),
                "recommended_action": (
                    "Modify conveyor lip edge and train worker to hug objects closer to body center before lifting."
                ),
                "confidence": 0.94,
            }

        elif category == "AWKWARD_POSTURE":
            trunk = 48.0 + var * 5.0
            knee = 155.0 - var * 5.0
            torque = 175.0 + var * 25.0
            comp = 3250.0 + var * 200.0
            return {
                "posture_class": "BENDING",
                "trunk_angle_deg": trunk,
                "knee_angle_deg": knee,
                "elbow_angle_deg": 160.0,
                "has_load": False,
                "load_weight_kg": 0.0,
                "load_distance_cm": 50.0 + var * 5.0,
                "lumbar_torque_nm": torque,
                "spinal_compression_n": comp,
                "awkward_duty_cycle_pct": 86.0 + var * 5.0,
                "rep_rate_per_min": 5.0,
                "cumulative_risk_score": 78.0 + var * 5.0,
                "risk_level": "HIGH",
                "hazard_title": "EXTENDED REACH AWKWARD POSTURE",
                "risk_factors": [
                    f"Severe torso forward flexion ({trunk:.1f} deg) past acceptable ergonomics limit",
                    "Extreme horizontal reach distance (> 50 cm)",
                    "High biomechanical trunk torque impulse",
                ],
                "ergonomic_observation": (
                    f"Worker is over-reaching horizontally across a wide table ({trunk:.1f} deg flexion), "
                    f"placing severe cantilever bending stress on the lumbar erector spinae muscles."
                ),
                "root_cause": (
                    "Worktable depth is 110 cm without a rotating turntable or rear access point."
                ),
                "recommended_action": (
                    "Install a manual rotating turntable or reduce table reach envelope to under 45 cm."
                ),
                "confidence": 0.96,
            }

        elif category == "REPETITIVE_MOTION":
            trunk = 35.0 + var * 3.0
            knee = 160.0 - var * 5.0
            torque = 85.0 + var * 10.0
            comp = 2100.0 + var * 150.0
            return {
                "posture_class": "BENDING",
                "trunk_angle_deg": trunk,
                "knee_angle_deg": knee,
                "elbow_angle_deg": 130.0,
                "has_load": True,
                "load_weight_kg": 5.0,
                "load_distance_cm": 38.0,
                "lumbar_torque_nm": torque,
                "spinal_compression_n": comp,
                "awkward_duty_cycle_pct": 94.0 + var * 2.0,
                "rep_rate_per_min": 18.0 + var * 4.0,
                "cumulative_risk_score": 82.0 + var * 4.0,
                "risk_level": "HIGH",
                "hazard_title": "HIGH-FREQUENCY REPETITIVE CYCLING",
                "risk_factors": [
                    f"Excessive repetition rate ({18.0 + var * 4.0:.1f} cycles/min > 12/min threshold)",
                    "Extremely high awkward posture duty cycle (> 90%)",
                    "Cumulative soft-tissue creep and micro-trauma accumulation",
                ],
                "ergonomic_observation": (
                    f"High-cadence repetitive picking at {18.0 + var * 4.0:.1f} cycles/min with 94% awkward duty cycle "
                    f"leaves zero recovery interval for muscular and intervertebral tissue regeneration."
                ),
                "root_cause": (
                    "Pacing enforced by fixed-speed automated sorting chute with no buffer queue."
                ),
                "recommended_action": (
                    "Enforce 15-minute job rotation intervals and introduce an accumulation buffer conveyor."
                ),
                "confidence": 0.95,
            }

        else:  # HIGH_RISK_LIFTING
            trunk = 46.0 + var * 4.0
            knee = 140.0 - var * 10.0
            torque = 210.0 + var * 25.0
            comp = 3950.0 + var * 350.0
            return {
                "posture_class": "LIFTING",
                "trunk_angle_deg": trunk,
                "knee_angle_deg": knee,
                "elbow_angle_deg": 115.0,
                "has_load": True,
                "load_weight_kg": 18.0 + var * 4.0,
                "load_distance_cm": 52.0 + var * 4.0,
                "lumbar_torque_nm": torque,
                "spinal_compression_n": comp,
                "awkward_duty_cycle_pct": 89.0 + var * 4.0,
                "rep_rate_per_min": 7.0,
                "cumulative_risk_score": 92.0 + var * 3.0,
                "risk_level": "DANGEROUS",
                "hazard_title": "CRITICAL HIGH-RISK LIFT",
                "risk_factors": [
                    f"Spinal compressive force ({comp:.0f} N) strictly violates the NIOSH Action Limit (3400 N)",
                    f"Heavy manual load ({18.0 + var * 4.0:.1f} kg) lifted with extended lever arm ({52.0 + var * 4.0:.1f} cm)",
                    f"Dangerous lumbar moment ({torque:.1f} Nm) exceeding tissue tolerance",
                ],
                "ergonomic_observation": (
                    f"CRITICAL HAZARD: Operative executes an unassisted lift of an 18 kg payload with an extended "
                    f"horizontal reach and 46 deg torso flexion. Spinal compression reaches {comp:.0f} N, creating "
                    f"acute risk of lumbar disc herniation."
                ),
                "root_cause": (
                    "Heavy bulk parts crates handled manually off low floor pallets without vacuum lifters."
                ),
                "recommended_action": (
                    "MANDATORY IMMEDIATE INTERVENTION: Implement dual-person lift protocol immediately and "
                    "install an overhead pneumatic zero-gravity crane lifter."
                ),
                "confidence": 0.99,
            }

    def _render_ergonomic_scene(
        self,
        output_path: str,
        trunk_angle_deg: float,
        knee_angle_deg: float,
        elbow_angle_deg: float,
        has_load: bool,
        load_distance_cm: float,
        hazard_title: str,
        sample_id: str,
    ):
        """Renders an articulated human figure in an industrial workstation setting."""
        # Create base warehouse image (concrete floor, safety lines, workstation)
        img = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        # Background gradient (industrial warehouse wall / floor)
        img[:280, :] = [45, 40, 42]  # Dark grey wall
        img[280:, :] = [75, 70, 72]  # Industrial concrete floor

        # Safety line on floor
        cv2.line(img, (0, 310), (self.width, 310), (30, 180, 240), 4)
        # Pallet / workstation rack
        cv2.rectangle(img, (400, 260), (580, 400), (90, 75, 55), -1)
        cv2.rectangle(img, (400, 260), (580, 400), (120, 105, 80), 2)
        cv2.putText(
            img,
            "STATION RACK",
            (420, 335),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (200, 200, 200),
            1,
        )

        # Worker geometry
        # Ankle / foot position
        ankle_x, ankle_y = 260, 410
        foot_x, foot_y = 290, 415
        cv2.line(img, (ankle_x, ankle_y), (foot_x, foot_y), (30, 30, 30), 8)

        # Knee position calculated from knee angle
        shin_len = 80
        # When knee angle is 180 (straight), knee is directly above ankle
        knee_flex = math.radians(180 - knee_angle_deg)
        knee_x = int(ankle_x + shin_len * math.sin(knee_flex))
        knee_y = int(ankle_y - shin_len * math.cos(knee_flex))

        # Hip position
        thigh_len = 85
        hip_x = int(knee_x - thigh_len * math.sin(knee_flex * 0.8))
        hip_y = int(knee_y - thigh_len * math.cos(knee_flex * 0.8))

        # Torso / Shoulder position
        torso_len = 100
        trunk_rad = math.radians(trunk_angle_deg)
        shoulder_x = int(hip_x + torso_len * math.sin(trunk_rad))
        shoulder_y = int(hip_y - torso_len * math.cos(trunk_rad))

        # Neck & Head
        neck_len = 25
        head_x = int(shoulder_x + neck_len * math.sin(trunk_rad))
        head_y = int(shoulder_y - neck_len * math.cos(trunk_rad))

        # Arms and Hands
        upper_arm_len = 55
        forearm_len = 50
        elbow_x = int(shoulder_x + upper_arm_len * 0.6)
        elbow_y = int(shoulder_y + upper_arm_len * 0.8)

        hand_x = int(shoulder_x + (load_distance_cm * 1.8))
        hand_y = int(elbow_y + forearm_len * 0.7)

        # Draw Figure Limbs (Hi-vis vest colors & worker apparel)
        # Legs (navy denim)
        cv2.line(img, (ankle_x, ankle_y), (knee_x, knee_y), (140, 80, 40), 12)
        cv2.line(img, (knee_x, knee_y), (hip_x, hip_y), (140, 80, 40), 14)
        # Torso (Hi-vis yellow/orange vest)
        torso_color = (0, 165, 255) if trunk_angle_deg > 30 else (0, 215, 255)
        cv2.line(img, (hip_x, hip_y), (shoulder_x, shoulder_y), torso_color, 18)
        # Reflective stripe on torso
        mid_torso_x = (hip_x + shoulder_x) // 2
        mid_torso_y = (hip_y + shoulder_y) // 2
        cv2.circle(img, (mid_torso_x, mid_torso_y), 6, (255, 255, 255), -1)

        # Head (Hardhat & face)
        cv2.circle(img, (head_x, head_y - 12), 16, (180, 160, 140), -1)  # Face
        cv2.ellipse(
            img, (head_x, head_y - 18), (18, 12), 0, 180, 360, (0, 220, 255), -1
        )  # Hardhat

        # Arms
        cv2.line(img, (shoulder_x, shoulder_y), (elbow_x, elbow_y), torso_color, 8)
        cv2.line(img, (elbow_x, elbow_y), (hand_x, hand_y), (180, 160, 140), 7)

        # Draw Load Crate if present
        if has_load:
            crate_w, crate_h = 60, 50
            cv2.rectangle(
                img,
                (hand_x - 10, hand_y - 20),
                (hand_x - 10 + crate_w, hand_y - 20 + crate_h),
                (40, 90, 160),
                -1,
            )
            cv2.rectangle(
                img,
                (hand_x - 10, hand_y - 20),
                (hand_x - 10 + crate_w, hand_y - 20 + crate_h),
                (20, 50, 110),
                2,
            )
            cv2.putText(
                img,
                "PARCEL",
                (hand_x - 5, hand_y + 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.35,
                (255, 255, 255),
                1,
            )

        # Overlays: HUD header & sample ID badge
        cv2.rectangle(img, (0, 0), (self.width, 42), (20, 20, 20), -1)
        cv2.putText(
            img,
            f"KINETICGUARD VLM DATASET | {sample_id} | {hazard_title}",
            (15, 26),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 255),
            2,
        )

        # Angle annotation overlay
        cv2.putText(
            img,
            f"Trunk: {trunk_angle_deg:.1f} deg",
            (shoulder_x - 80, shoulder_y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 0),
            1,
        )
        cv2.putText(
            img,
            f"L5/S1 Moment Arm: {load_distance_cm:.0f} cm",
            (hip_x - 90, hip_y + 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 200, 255),
            1,
        )

        cv2.imwrite(output_path, img)


def get_dataset_stats(dataset_path: str = "data/ergonomic_dataset/annotations.json"):
    """Returns basic stats on the generated dataset."""
    path = Path(dataset_path)
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return {
        "total_samples": data.get("total_samples", 0),
        "categories": data.get("categories", []),
    }
