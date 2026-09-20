"""
KineticGuard - Ergonomic Incident Engine
Detects, groups, and triggers structured ERGONOMIC_INCIDENT records
with contributing factors, root-cause recommendations, and session statistics.
"""

import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import cv2
except ImportError:
    cv2 = None

from .cumulative_risk import CumulativeRiskScore
from .privacy_guard import privacy_guard


def resolve_station(camera_id: str) -> str:
    """Dynamically resolves station identity from environment or runtime camera ID."""
    env_station = os.getenv(f"{camera_id.upper()}_STATION")
    if env_station:
        return env_station
    clean_id = camera_id.replace("camera_", "").upper()
    return f"STATION-{clean_id}"


def generate_worker_identity(camera_id: str, session_id: Optional[str] = None) -> str:
    """Generates a dynamic runtime worker tracking ID bound to camera and session."""
    env_worker = os.getenv(f"{camera_id.upper()}_WORKER")
    if env_worker:
        return env_worker
    clean_cam = camera_id.replace("camera_", "C").upper()
    sess_tag = (session_id or str(uuid.uuid4())[:6]).upper().replace("-", "")[:4]
    return f"W-{clean_cam}-{sess_tag}"


STATION_MAP: Dict[str, Any] = {}


class ErgonomicIncident:
    """Represents a discrete ergonomic hazard event triggered during surveillance."""

    def __init__(
        self,
        incident_id: str,
        worker_id: str,
        camera_id: str,
        station_id: str,
        start_time_sec: float,
        end_time_sec: float,
        risk_level: str,
        cumulative_risk_score: float,
        exposure_metrics: Dict[str, Any],
        contributing_factors: List[str],
        recommended_action: str,
        event_id: Optional[str] = None,
        session_id: Optional[str] = None,
        source: Optional[str] = None,
        keyframe_path: Optional[str] = None,
        keyframe_frame_index: Optional[int] = None,
    ):
        self.event_id = event_id or incident_id
        self.incident_id = incident_id
        self.worker_id = worker_id
        self.camera_id = camera_id
        self.station_id = station_id
        self.session_id = session_id or f"sess_{camera_id}"
        self.source = source or f"edge_runtime_pose_{camera_id}"
        self.start_time_sec = round(start_time_sec, 2)
        self.end_time_sec = round(end_time_sec, 2)
        self.duration_sec = round(max(0.01, end_time_sec - start_time_sec), 2)
        self.timestamp = datetime.now(timezone.utc).isoformat()
        self.risk_level = risk_level
        self.cumulative_risk_score = round(cumulative_risk_score, 1)
        self.exposure_metrics = exposure_metrics
        self.contributing_factors = contributing_factors
        self.recommended_action = recommended_action
        self.keyframe_path = keyframe_path
        self.keyframe_frame_index = keyframe_frame_index

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_id": self.event_id,
            "incident_id": self.incident_id,
            "worker_id": self.worker_id,
            "camera_id": self.camera_id,
            "station_id": self.station_id,
            "session_id": self.session_id,
            "source": self.source,
            "timestamp": self.timestamp,
            "start_time_sec": self.start_time_sec,
            "end_time_sec": self.end_time_sec,
            "duration_sec": self.duration_sec,
            "risk_level": self.risk_level,
            "cumulative_risk_score": self.cumulative_risk_score,
            "exposure_metrics": self.exposure_metrics,
            "contributing_factors": self.contributing_factors,
            "recommended_action": self.recommended_action,
            "keyframe_path": self.keyframe_path,
            "keyframe_frame_index": self.keyframe_frame_index,
            "is_estimated_model_indicator": True,
            "disclaimer": "Biomechanical indicators (torque/compression) are modeled estimates and not direct clinical measurements.",
        }

    def to_strands_incident_packet(self) -> Dict[str, Any]:
        """Formats incident record according to kineticguard.strands.v1 schema for Strands Agents SDK."""
        return {
            "schema_version": "kineticguard.strands.v1",
            "event_type": "ERGONOMIC_INCIDENT",
            "agent_target": "strands_ergonomic_safety_agent",
            "incident_id": self.incident_id,
            "event_id": self.event_id,
            "session_id": self.session_id,
            "source": self.source,
            "timestamp_iso": self.timestamp,
            "worker_identity": {
                "worker_id": self.worker_id,
                "station_id": self.station_id,
                "camera_id": self.camera_id,
            },
            "temporal_window": {
                "start_time_sec": self.start_time_sec,
                "end_time_sec": self.end_time_sec,
                "duration_sec": self.duration_sec,
            },
            "risk_assessment": {
                "cumulative_risk_score": self.cumulative_risk_score,
                "risk_level": self.risk_level,
                "contributing_factors": self.contributing_factors,
                "recommended_action": self.recommended_action,
            },
            "biomechanical_exposure": {
                "is_estimated_model_indicator": True,
                "disclaimer": "Biomechanical indicators (torque/compression) are modeled estimates and not direct clinical measurements.",
                "awkward_duty_cycle_pct": self.exposure_metrics.get("awkward_duty_cycle_pct", 0.0),
                "repetition_rate_per_min": self.exposure_metrics.get("rep_rate_per_min", 0.0),
                "peak_lumbar_torque_nm": self.exposure_metrics.get("peak_lumbar_torque_nm", 0.0),
                "peak_spinal_compression_n": self.exposure_metrics.get("peak_spinal_compression_n", 0.0),
                "cumulative_torque_impulse_nms": self.exposure_metrics.get("cumulative_torque_impulse_nms", 0.0),
            },
            "visual_evidence": {
                "keyframe_path": self.keyframe_path,
                "keyframe_frame_index": self.keyframe_frame_index,
            },
        }

    to_strands_packet = to_strands_incident_packet


class IncidentEngine:
    """
    Evaluates temporal exposure and risk metrics to detect, de-duplicate,
    and trigger discrete ERGONOMIC_INCIDENT records.
    """

    def __init__(
        self,
        camera_id: str,
        incident_threshold: float = 60.0,
        cooldown_sec: float = 2.0,
        worker_id: Optional[str] = None,
        station_id: Optional[str] = None,
        session_id: Optional[str] = None,
        source: Optional[str] = None,
        keyframes_dir: Optional[Path] = None,
    ):
        self.camera_id = camera_id
        self.incident_threshold = incident_threshold
        self.cooldown_sec = cooldown_sec
        self.session_id = session_id or f"sess_{camera_id}"
        self.source = source or f"edge_camera_stream_{camera_id}"
        self.worker_id = worker_id or generate_worker_identity(camera_id, self.session_id)
        self.station_id = station_id or resolve_station(camera_id)
        self.keyframes_dir = Path(keyframes_dir) if keyframes_dir else Path("output/incidents/keyframes")

        self.incidents: List[ErgonomicIncident] = []
        self._active_incident: Optional[Dict[str, Any]] = None
        self._incident_counter = 0

    def evaluate_frame(
        self,
        timestamp_sec: float,
        temporal_metrics: Dict[str, Any],
        risk_score: CumulativeRiskScore,
        frame: Optional[Any] = None,
        frame_index: Optional[int] = None,
    ) -> Optional[ErgonomicIncident]:
        """
        Evaluates current temporal metrics and risk score.
        Groups contiguous trigger conditions into a cohesive incident.
        """
        is_breached = self._check_breach_condition(temporal_metrics, risk_score)

        # 1. Condition breached -> open or extend active incident
        if is_breached:
            if self._active_incident is None:
                # Open new incident window
                self._incident_counter += 1
                inc_id = f"INC-{datetime.now().strftime('%Y%m%d')}-{self.camera_id.upper()}-{self._incident_counter:03d}"
                self._active_incident = {
                    "incident_id": inc_id,
                    "start_time": timestamp_sec,
                    "last_active": timestamp_sec,
                    "max_score": risk_score.score,
                    "worst_level": risk_score.level,
                    "metrics": dict(temporal_metrics),
                    "factors": set(risk_score.contributing_factors),
                    "keyframe": frame.copy() if frame is not None else None,
                    "keyframe_index": frame_index,
                    "keyframe_timestamp": timestamp_sec,
                }
            else:
                # Extend existing incident
                self._active_incident["last_active"] = timestamp_sec
                if risk_score.score > self._active_incident["max_score"]:
                    self._active_incident["max_score"] = risk_score.score
                    self._active_incident["worst_level"] = risk_score.level
                    self._active_incident["metrics"] = dict(temporal_metrics)
                    if frame is not None:
                        self._active_incident["keyframe"] = frame.copy()
                        self._active_incident["keyframe_index"] = frame_index
                        self._active_incident["keyframe_timestamp"] = timestamp_sec
                for f in risk_score.contributing_factors:
                    self._active_incident["factors"].add(f)

                # If continuous hazardous exposure exceeds 2.0 seconds, trigger incident promptly
                if (timestamp_sec - self._active_incident["start_time"]) >= 2.0:
                    return self._finalize_active_incident()

            return None

        # 2. Condition no longer breached -> finalize active incident after cooldown
        if self._active_incident is not None:
            if (timestamp_sec - self._active_incident["last_active"]) >= self.cooldown_sec:
                finalized = self._finalize_active_incident()
                return finalized

        return None

    def flush(self, final_timestamp_sec: Optional[float] = None) -> Optional[ErgonomicIncident]:
        """Finalizes any remaining open incident at session end."""
        if self._active_incident is not None:
            if final_timestamp_sec is not None:
                self._active_incident["last_active"] = max(
                    self._active_incident["last_active"], final_timestamp_sec
                )
            return self._finalize_active_incident()
        return None

    def _check_breach_condition(
        self,
        temporal_metrics: Dict[str, Any],
        risk_score: CumulativeRiskScore,
    ) -> bool:
        """Determines if current exposure breaches ergonomic safety thresholds with sustained evidence."""
        # Condition A: Cumulative risk score exceeds threshold
        if risk_score.score >= self.incident_threshold:
            return True

        awkward_duty = float(temporal_metrics.get("awkward_duty_cycle_pct", 0.0))
        window_dur = max(1.0, float(temporal_metrics.get("window_duration_sec", 20.0)))
        dur_awkward = (awkward_duty / 100.0) * window_dur

        # Condition B: Spinal compression exceeds NIOSH Action Limit (3,400 N) with sustained evidence
        if float(temporal_metrics.get("peak_spinal_compression_n", 0.0)) >= 3400.0 and (awkward_duty >= 30.0 or dur_awkward >= 2.0):
            return True

        # Condition C: High repetition lifting rate (> 6 reps/min) with awkward duty > 30% sustained
        if (
            float(temporal_metrics.get("rep_rate_per_min", 0.0)) >= 6.0
            and awkward_duty >= 30.0
            and dur_awkward >= 2.0
        ):
            return True

        # Condition D: Sustained trunk flexion >= 45° for > 3.0s
        bending_sec = float(temporal_metrics.get("duration_bending_sec", 0.0))
        peak_trunk = float(temporal_metrics.get("peak_trunk_flexion_deg", 0.0))
        if bending_sec >= 3.0 and peak_trunk >= 45.0:
            return True

        return False

    def _finalize_active_incident(self) -> ErgonomicIncident:
        """Constructs and registers a completed ErgonomicIncident object."""
        act = self._active_incident
        self._active_incident = None

        # Synthesize concise consolidated contributing factors from window metrics
        m = act["metrics"]
        factors = []

        peak_comp = m.get("peak_spinal_compression_n", 0.0)
        if peak_comp >= 3400.0:
            factors.append(f"Peak spinal compression ({peak_comp:.0f} N) exceeds NIOSH Action Limit (3,400 N)")
        elif peak_comp >= 2200.0:
            factors.append(f"Elevated spinal compression ({peak_comp:.0f} N) approaching safety threshold")

        peak_trunk = m.get("peak_trunk_flexion_deg", 0.0)
        if peak_trunk >= 45.0:
            factors.append(f"Excessive trunk flexion (peak {peak_trunk:.1f} deg exceeds 45 deg threshold)")
        elif peak_trunk >= 25.0:
            factors.append(f"Awkward torso forward flexion (peak {peak_trunk:.1f} deg)")

        awkward_pct = m.get("awkward_duty_cycle_pct", 0.0)
        if awkward_pct >= 40.0:
            factors.append(f"High awkward posture duty cycle ({awkward_pct:.1f}% in bending/lifting)")

        rep_rate = m.get("rep_rate_per_min", 0.0)
        if rep_rate >= 6.0:
            factors.append(f"Repetitive motion frequency ({rep_rate:.1f} reps/min, {m.get('repetition_count', 0)} cycles)")

        peak_torque = m.get("peak_lumbar_torque_nm", 0.0)
        if peak_torque >= 85.0:
            factors.append(f"Elevated peak lumbar moment ({peak_torque:.1f} Nm at L5/S1)")

        if not factors:
            factors.append("Cumulative ergonomic strain threshold exceeded")

        action = self._generate_recommendation(factors, m)

        # Save captured keyframe from actual video stream if available (MANDATORY PRIVACY DE-IDENTIFICATION)
        keyframe_path = None
        keyframe_idx = act.get("keyframe_index")
        if act.get("keyframe") is not None and cv2 is not None:
            try:
                self.keyframes_dir.mkdir(parents=True, exist_ok=True)
                kf_file = self.keyframes_dir / f"{act['incident_id']}_keyframe.jpg"
                deidentified_kf = privacy_guard.deidentify_frame(act["keyframe"])
                cv2.imwrite(str(kf_file), deidentified_kf)
                keyframe_path = str(kf_file)
            except Exception:
                keyframe_path = None

        incident = ErgonomicIncident(
            incident_id=act["incident_id"],
            worker_id=self.worker_id,
            camera_id=self.camera_id,
            station_id=self.station_id,
            start_time_sec=act["start_time"],
            end_time_sec=act["last_active"],
            risk_level=act["worst_level"],
            cumulative_risk_score=act["max_score"],
            exposure_metrics=m,
            contributing_factors=factors,
            recommended_action=action,
            event_id=act["incident_id"],
            session_id=self.session_id,
            source=self.source,
            keyframe_path=keyframe_path,
            keyframe_frame_index=keyframe_idx,
        )
        self.incidents.append(incident)
        return incident

    def _generate_recommendation(
        self, factors: List[str], metrics: Dict[str, Any]
    ) -> str:
        """Generates tailored ergonomic corrective actions based on observed root causes."""
        bending_time = metrics.get("duration_bending_sec", 0.0)
        rep_rate = metrics.get("rep_rate_per_min", 0.0)
        peak_comp = metrics.get("peak_spinal_compression_n", 0.0)
        peak_trunk = metrics.get("peak_trunk_flexion_deg", 0.0)

        recommendations = []

        if peak_comp >= 3400.0:
            recommendations.append(
                "Deploy mechanical lifting assistance (vacuum hoist or scissor lift table); "
                "enforce team two-person lifts for loads exceeding 15 kg."
            )
        elif peak_trunk >= 45.0 or bending_time >= 3.0:
            recommendations.append(
                "Reconfigure workstation and pallet height to waist level (75-90 cm) "
                "to eliminate deep trunk flexion; provide anti-fatigue matting."
            )
        elif rep_rate >= 6.0:
            recommendations.append(
                "Enforce mandatory task rotation (rotate worker every 20 minutes to non-manual sorting); "
                "limit continuous lifting rate to < 4 lifts/minute."
            )
        else:
            recommendations.append(
                "Mandate immediate 3-5 minute micro-break with spinal extension stretching; "
                "conduct on-site ergonomic workstation assessment."
            )

        return " ".join(recommendations)
