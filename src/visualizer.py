"""
KineticGuard - Biomechanical Visualizer Module
Draws color-coded skeleton connections, joint angle callouts, and an ergonomic
HUD overlay (posture badge, lumbar torque gauge, compression forces) directly onto video frames.
"""

from typing import Dict, Optional, Tuple
import cv2
import numpy as np

from .biomechanics import BiomechanicsMetrics
from .pose_detector import PoseDetectionResult

# Skeletal joint connections for 2D visualization
SKELETON_PAIRS = [
    # Torso
    ("left_shoulder", "right_shoulder"),
    ("left_shoulder", "left_hip"),
    ("right_shoulder", "right_hip"),
    ("left_hip", "right_hip"),
    # Left Arm
    ("left_shoulder", "left_elbow"),
    ("left_elbow", "left_wrist"),
    # Right Arm
    ("right_shoulder", "right_elbow"),
    ("right_elbow", "right_wrist"),
    # Left Leg
    ("left_hip", "left_knee"),
    ("left_knee", "left_ankle"),
    # Right Leg
    ("right_hip", "right_knee"),
    ("right_knee", "right_ankle"),
]

# Color map (BGR format) for risk levels
RISK_COLORS = {
    "LOW": (60, 210, 60),         # Green
    "MODERATE": (0, 220, 255),     # Yellow/Amber
    "HIGH": (0, 140, 255),         # Orange
    "DANGEROUS": (40, 40, 245),     # Red
}


class BiomechanicalVisualizer:
    """Renders skeletal pose landmarks and biomechanical telemetry HUD overlays."""

    def __init__(self, camera_id: str):
        self.camera_id = camera_id

    def draw_frame(
        self,
        frame_bgr: np.ndarray,
        pose_res: PoseDetectionResult,
        biomech: BiomechanicsMetrics,
        frame_idx: int = 0,
        fps: float = 25.0,
    ) -> np.ndarray:
        """Draws skeleton, joint angles, and biomechanical telemetry HUD on frame."""
        annotated = frame_bgr.copy()
        h, w = annotated.shape[:2]

        risk_color = RISK_COLORS.get(biomech.risk_level, (180, 180, 180))

        if pose_res.detected:
            # 1. Draw Skeleton Bones
            pts = pose_res.landmarks_px
            for joint_a, joint_b in SKELETON_PAIRS:
                if joint_a in pts and joint_b in pts:
                    p1 = pts[joint_a]
                    p2 = pts[joint_b]
                    cv2.line(annotated, p1, p2, risk_color, 3, cv2.LINE_AA)

            # 2. Draw Skeleton Joints
            for joint_name, coord in pts.items():
                cv2.circle(annotated, coord, 5, (255, 255, 255), -1, cv2.LINE_AA)
                cv2.circle(annotated, coord, 6, risk_color, 2, cv2.LINE_AA)

            # 3. Draw Joint Angle Callouts
            self._draw_angle_labels(annotated, pts, pose_res.angles)

        # 4. Draw Top Biomechanical HUD
        self._draw_telemetry_hud(annotated, biomech, pose_res.detected, frame_idx, fps, risk_color)

        return annotated

    def _draw_angle_labels(
        self,
        frame: np.ndarray,
        pts: Dict[str, Tuple[int, int]],
        angles: Dict[str, float],
    ) -> None:
        """Overlays text callouts next to major joints."""
        # Knee angle callout
        if "left_knee" in pts:
            kp = pts["left_knee"]
            ang = angles.get("left_knee_deg", 0.0)
            cv2.putText(
                frame,
                f"Knee: {ang:.0f} deg",
                (kp[0] + 8, kp[1] - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.40,
                (255, 255, 200),
                1,
                cv2.LINE_AA,
            )

        # Elbow angle callout
        if "left_elbow" in pts:
            ep = pts["left_elbow"]
            ang = angles.get("left_elbow_deg", 0.0)
            cv2.putText(
                frame,
                f"Elbow: {ang:.0f} deg",
                (ep[0] + 8, ep[1] - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.40,
                (255, 255, 200),
                1,
                cv2.LINE_AA,
            )

        # Trunk flexion callout near hip
        if "left_hip" in pts:
            hp = pts["left_hip"]
            ang = angles.get("trunk_flexion_deg", 0.0)
            cv2.putText(
                frame,
                f"Trunk: {ang:.0f} deg",
                (hp[0] + 10, hp[1] + 12),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
                (100, 255, 255),
                1,
                cv2.LINE_AA,
            )

    def _draw_telemetry_hud(
        self,
        frame: np.ndarray,
        biomech: BiomechanicsMetrics,
        detected: bool,
        frame_idx: int,
        fps: float,
        risk_color: Tuple[int, int, int],
    ) -> None:
        """Renders the top telemetry dashboard with posture badge and torque bar."""
        h, w = frame.shape[:2]

        # Top HUD bar background (semi-transparent dark overlay)
        hud_h = 70
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, hud_h), (18, 18, 22), -1)
        cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)

        # Divider line
        cv2.line(frame, (0, hud_h), (w, hud_h), risk_color, 2)

        # Left: Camera and Posture Badge
        cam_text = f"KG [{self.camera_id.upper()}]  FRM:{frame_idx:05d}"
        cv2.putText(
            frame,
            cam_text,
            (14, 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (180, 180, 180),
            1,
            cv2.LINE_AA,
        )

        posture_badge = f"POSTURE: {biomech.posture}" if detected else "POSE: SEARCHING..."
        badge_bg = risk_color if detected else (80, 80, 80)
        cv2.rectangle(frame, (12, 32), (180, 58), badge_bg, -1)
        text_color = (0, 0, 0) if biomech.risk_level in ("LOW", "MODERATE") else (255, 255, 255)
        cv2.putText(
            frame,
            posture_badge,
            (20, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            text_color,
            1,
            cv2.LINE_AA,
        )

        # Middle: Lumbar Torque & Compression
        torque_text = f"L5/S1 Torque: {biomech.lumbar_torque_nm:.1f} Nm"
        comp_text = f"Spinal Comp: {biomech.compression_force_n:.0f} N"
        cv2.putText(
            frame,
            torque_text,
            (200, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.46,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            comp_text,
            (200, 48),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (200, 220, 255),
            1,
            cv2.LINE_AA,
        )

        # Right: Risk Level Gauge & Score
        risk_label = f"RISK: {biomech.risk_level} ({biomech.risk_score:.0f}/100)"
        cv2.putText(
            frame,
            risk_label,
            (w - 230, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.46,
            risk_color,
            1,
            cv2.LINE_AA,
        )

        # Risk Gauge Bar
        bar_x, bar_y, bar_w, bar_h = w - 230, 36, 210, 16
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (40, 40, 40), -1)
        fill_w = int((min(100.0, biomech.risk_score) / 100.0) * bar_w)
        if fill_w > 0:
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h), risk_color, -1)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (120, 120, 120), 1)

        # Bottom Lever Arm Info
        lever_str = f"Lever Arms -> Torso: {biomech.torso_lever_arm_m:.2f}m | Load: {biomech.load_lever_arm_m:.2f}m | Trunk Flex: {biomech.trunk_flexion_deg:.1f} deg"
        cv2.putText(
            frame,
            lever_str,
            (14, h - 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.36,
            (210, 210, 210),
            1,
            cv2.LINE_AA,
        )
