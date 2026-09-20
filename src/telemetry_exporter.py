"""
KineticGuard - Telemetry Exporter
Collects, structures, and exports per-frame ergonomic and biomechanical metrics
to JSON, CSV, and Strands Agents SDK formats for downstream temporal risk scoring.
"""

import csv
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional


class TelemetryExporter:
    """Manages telemetry recording and export in JSON, CSV, and Strands Agent formats."""

    def __init__(
        self,
        camera_id: str,
        session_id: Optional[str] = None,
        worker_id: Optional[str] = None,
    ):
        self.camera_id = camera_id
        self.session_id = session_id or f"sess_{int(time.time())}_{camera_id}"
        clean_cam = camera_id.replace("camera_", "C").upper()
        self.worker_id = worker_id or f"W-{clean_cam}-{self.session_id[-4:]}"
        self.records: List[Dict[str, Any]] = []

    def record_frame(
        self,
        frame_index: int,
        timestamp_sec: float,
        pose_detected: bool,
        angles: Dict[str, float],
        biomech_dict: Dict[str, Any],
        keypoints: Optional[Dict[str, Dict[str, float]]] = None,
    ) -> Dict[str, Any]:
        """Formats and stores a single frame telemetry record."""
        record = {
            "frame_index": frame_index,
            "timestamp_sec": round(timestamp_sec, 3),
            "camera_id": self.camera_id,
            "session_id": self.session_id,
            "worker_id": self.worker_id,
            "pose_detected": pose_detected,
            "posture": biomech_dict.get("posture", "UNKNOWN"),
            "trunk_flexion_deg": biomech_dict.get("trunk_flexion_deg", 0.0),
            "lumbar_torque_nm": biomech_dict.get("lumbar_torque_nm", 0.0),
            "compression_force_n": biomech_dict.get("compression_force_n", 0.0),
            "torso_lever_arm_m": biomech_dict.get("torso_lever_arm_m", 0.0),
            "load_lever_arm_m": biomech_dict.get("load_lever_arm_m", 0.0),
            "risk_level": biomech_dict.get("risk_level", "LOW"),
            "risk_score": biomech_dict.get("risk_score", 0.0),
            # Estimated indicator labeling
            "is_estimated_model_indicator": True,
            "disclaimer": "Estimated biomechanical model indicator; not a direct medical measurement.",
            # Joint angles
            "left_shoulder_deg": round(angles.get("left_shoulder_deg", 0.0), 1),
            "right_shoulder_deg": round(angles.get("right_shoulder_deg", 0.0), 1),
            "left_elbow_deg": round(angles.get("left_elbow_deg", 0.0), 1),
            "right_elbow_deg": round(angles.get("right_elbow_deg", 0.0), 1),
            "left_hip_deg": round(angles.get("left_hip_deg", 0.0), 1),
            "right_hip_deg": round(angles.get("right_hip_deg", 0.0), 1),
            "left_knee_deg": round(angles.get("left_knee_deg", 0.0), 1),
            "right_knee_deg": round(angles.get("right_knee_deg", 0.0), 1),
        }

        if keypoints:
            record["keypoints"] = {
                k: {"x": round(v["x"], 3), "y": round(v["y"], 3), "z": round(v["z"], 3)}
                for k, v in keypoints.items()
                if k in ("left_shoulder", "right_shoulder", "left_wrist", "right_wrist", "left_hip", "right_hip", "left_knee", "right_knee")
            }

        self.records.append(record)
        return record

    def to_strands_agent_packet(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """Formats a telemetry record into an event payload ready for Strands Agents SDK."""
        return {
            "schema_version": "kineticguard.strands.v1",
            "agent_target": "strands_ergonomic_agent",
            "session_id": self.session_id,
            "worker_id": self.worker_id,
            "camera_id": self.camera_id,
            "frame_index": record["frame_index"],
            "timestamp_sec": record["timestamp_sec"],
            "pose_detected": record["pose_detected"],
            "posture": record["posture"],
            "biomechanical_indicators": {
                "trunk_flexion_deg": record["trunk_flexion_deg"],
                "lumbar_torque_nm": record["lumbar_torque_nm"],
                "compression_force_n": record["compression_force_n"],
                "is_estimated_model_indicator": True,
                "disclaimer": "Estimated biomechanical model indicator; not a direct medical measurement.",
            },
            "joint_angles": {
                "left_shoulder_deg": record.get("left_shoulder_deg", 0.0),
                "right_shoulder_deg": record.get("right_shoulder_deg", 0.0),
                "left_elbow_deg": record.get("left_elbow_deg", 0.0),
                "right_elbow_deg": record.get("right_elbow_deg", 0.0),
                "left_hip_deg": record.get("left_hip_deg", 0.0),
                "right_hip_deg": record.get("right_hip_deg", 0.0),
                "left_knee_deg": record.get("left_knee_deg", 0.0),
                "right_knee_deg": record.get("right_knee_deg", 0.0),
            },
            "risk_assessment": {
                "risk_level": record.get("risk_level", "LOW"),
                "risk_score": record.get("risk_score", 0.0),
            },
            "keypoints": record.get("keypoints", {}),
        }

    def get_summary(self) -> Dict[str, Any]:
        """Computes aggregate session statistics."""
        if not self.records:
            return {"total_frames": 0}

        total_frames = len(self.records)
        detected_frames = sum(1 for r in self.records if r["pose_detected"])
        torques = [r["lumbar_torque_nm"] for r in self.records if r["pose_detected"]]
        compressions = [r["compression_force_n"] for r in self.records if r["pose_detected"]]

        posture_counts: Dict[str, int] = {}
        risk_counts: Dict[str, int] = {}

        for r in self.records:
            p = r["posture"]
            posture_counts[p] = posture_counts.get(p, 0) + 1
            rk = r["risk_level"]
            risk_counts[rk] = risk_counts.get(rk, 0) + 1

        return {
            "camera_id": self.camera_id,
            "session_id": self.session_id,
            "worker_id": self.worker_id,
            "total_frames": total_frames,
            "detected_frames": detected_frames,
            "detection_rate_pct": round((detected_frames / max(1, total_frames)) * 100, 1),
            "mean_lumbar_torque_nm": round(float(sum(torques) / max(1, len(torques))), 1) if torques else 0.0,
            "peak_lumbar_torque_nm": round(float(max(torques)), 1) if torques else 0.0,
            "mean_compression_force_n": round(float(sum(compressions) / max(1, len(compressions))), 1) if compressions else 0.0,
            "peak_compression_force_n": round(float(max(compressions)), 1) if compressions else 0.0,
            "indicator_type": "ESTIMATED_MODEL_INDICATOR",
            "disclaimer": "All torque and compression values are estimated biomechanical indicators, not direct medical measurements.",
            "posture_breakdown": posture_counts,
            "risk_level_breakdown": risk_counts,
        }

    def save_json(self, output_path: str) -> str:
        """Saves telemetry and summary to a structured JSON file."""
        p = Path(output_path)
        p.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            "metadata": {
                "system": "KineticGuard Biomechanics Engine",
                "version": "Phase 2 (Build It Architecture)",
                "camera_id": self.camera_id,
                "session_id": self.session_id,
                "worker_id": self.worker_id,
                "indicator_type": "ESTIMATED_MODEL_INDICATOR",
                "disclaimer": "All torque and compression values are estimated biomechanical indicators, not direct medical measurements.",
            },
            "summary": self.get_summary(),
            "frames": self.records,
        }

        with open(p, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

        return str(p)

    def save_csv(self, output_path: str) -> str:
        """Saves tabular frame telemetry to a CSV file."""
        p = Path(output_path)
        p.parent.mkdir(parents=True, exist_ok=True)

        if not self.records:
            return str(p)

        fieldnames = [k for k in self.records[0].keys() if k != "keypoints"]

        with open(p, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            for rec in self.records:
                writer.writerow(rec)

        return str(p)
