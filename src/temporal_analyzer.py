"""
KineticGuard - Temporal Analyzer Module
Implements sliding temporal window buffers (15–30s) to track posture durations,
duty cycles, mechanical torque impulse, and repetitive movement cycles.
"""

from collections import deque
from typing import Any, Dict, List, Optional, Tuple


class RepetitionCycleDetector:
    """
    Detects ergonomic movement cycles using a state machine:
    UPRIGHT -> BENDING / LIFTING -> UPRIGHT.
    Enforces minimum dwell time to eliminate transient frame jitter.
    """

    def __init__(self, min_dwell_sec: float = 0.25):
        self.min_dwell_sec = min_dwell_sec
        self.state = "UPRIGHT"
        self.state_enter_time = 0.0
        self.completed_lifts = 0
        self.completed_bends = 0
        self.current_cycle_type = None

    def update(self, posture: str, timestamp_sec: float) -> Tuple[bool, Optional[str]]:
        """
        Updates detector with current frame posture.
        Returns (cycle_completed, cycle_type) if a full ergonomic cycle finished.
        """
        cycle_finished = False
        finished_type = None

        if posture != self.state:
            dwell = timestamp_sec - self.state_enter_time

            # Transition out of UPRIGHT into awkward/active posture
            if self.state == "UPRIGHT":
                if posture in ("BENDING", "LIFTING", "SQUATTING"):
                    self.state = posture
                    self.current_cycle_type = posture
                    self.state_enter_time = timestamp_sec

            # Transition out of awkward/active posture back to UPRIGHT
            elif self.state in ("BENDING", "LIFTING", "SQUATTING"):
                if posture == "UPRIGHT" and dwell >= self.min_dwell_sec:
                    cycle_finished = True
                    finished_type = self.current_cycle_type

                    if finished_type == "LIFTING":
                        self.completed_lifts += 1
                    else:
                        self.completed_bends += 1

                    self.state = "UPRIGHT"
                    self.state_enter_time = timestamp_sec
                    self.current_cycle_type = None
                elif posture != self.state:
                    # Switch between active postures (e.g. Bending to Lifting)
                    self.state = posture
                    if posture == "LIFTING":
                        self.current_cycle_type = "LIFTING"

        return cycle_finished, finished_type


class TemporalSlidingWindow:
    """
    Maintains a sliding temporal window of frame telemetry over W seconds (15–30s).
    Aggregates posture durations, duty cycles, torque impulse, strain accumulation/decay, and movement kinematics.
    """

    def __init__(self, window_seconds: float = 20.0):
        self.window_seconds = max(5.0, min(60.0, window_seconds))
        self.buffer: deque = deque()
        self.cycle_detector = RepetitionCycleDetector()
        self.cycle_timestamps: List[Tuple[float, str]] = []
        self.cumulative_strain: float = 0.0
        self.sustained_awkward_sec: float = 0.0
        self.last_ts: Optional[float] = None

    def add_frame(self, frame_data: Dict[str, Any]) -> None:
        """Appends a frame record to the buffer, updates cycle detection, and tracks strain/recovery."""
        ts = frame_data.get("timestamp_sec", 0.0)
        posture = frame_data.get("posture", "UPRIGHT")

        # Dynamic strain accumulation and physiological recovery decay
        if self.last_ts is not None:
            dt = max(0.0, min(2.0, ts - self.last_ts))
            if posture in ("BENDING", "LIFTING", "SQUATTING"):
                self.sustained_awkward_sec += dt
                tq = max(0.0, frame_data.get("lumbar_torque_nm", 30.0) - 25.0)
                comp = max(0.0, frame_data.get("compression_force_n", 800.0) / 1500.0)
                delta_strain = (tq / 60.0) * comp * dt * 8.0
                self.cumulative_strain = min(100.0, self.cumulative_strain + delta_strain)
            else:
                # UPRIGHT posture provides physiological recovery
                self.sustained_awkward_sec = max(0.0, self.sustained_awkward_sec - dt * 2.0)
                import math
                self.cumulative_strain *= math.exp(-0.40 * dt)
                if self.cumulative_strain < 0.01:
                    self.cumulative_strain = 0.0

        self.last_ts = ts
        self.buffer.append(frame_data)

        # Update repetition cycle detector
        cycle_done, cycle_type = self.cycle_detector.update(posture, ts)
        if cycle_done and cycle_type:
            self.cycle_timestamps.append((ts, cycle_type))

        # Evict frames older than current timestamp - window_seconds
        cutoff = ts - self.window_seconds
        while self.buffer and self.buffer[0].get("timestamp_sec", 0.0) < cutoff:
            self.buffer.popleft()

    def get_metrics(self) -> Dict[str, Any]:
        """
        Computes aggregate exposure metrics across the current temporal window.
        """
        if not self.buffer:
            return {
                "window_duration_sec": 0.0,
                "frame_count": 0,
                "duration_upright_sec": 0.0,
                "duration_bending_sec": 0.0,
                "duration_squatting_sec": 0.0,
                "duration_lifting_sec": 0.0,
                "awkward_duty_cycle_pct": 0.0,
                "repetition_count": 0,
                "rep_rate_per_min": 0.0,
                "mean_lumbar_torque_nm": 0.0,
                "peak_lumbar_torque_nm": 0.0,
                "mean_spinal_compression_n": 0.0,
                "peak_spinal_compression_n": 0.0,
                "mean_trunk_flexion_deg": 0.0,
                "peak_trunk_flexion_deg": 0.0,
                "mean_phase2_risk_score": 0.0,
                "cumulative_torque_impulse_nms": 0.0,
                "sustained_awkward_sec": 0.0,
                "cumulative_strain_index": 0.0,
                "is_estimated_model_indicator": True,
                "disclaimer": "Temporal metrics and torque impulses are modeled biomechanical estimates and not direct medical or clinical measurements.",
            }

        first_ts = self.buffer[0].get("timestamp_sec", 0.0)
        last_ts = self.buffer[-1].get("timestamp_sec", 0.0)
        window_dur = max(0.01, last_ts - first_ts)

        # Duration per posture
        duration_by_posture = {"UPRIGHT": 0.0, "BENDING": 0.0, "SQUATTING": 0.0, "LIFTING": 0.0}

        torques = []
        compressions = []
        trunk_angles = []
        p2_scores = []
        torque_impulse = 0.0

        for i in range(len(self.buffer)):
            curr = self.buffer[i]
            posture = curr.get("posture", "UPRIGHT")

            # Time delta for this frame
            if i < len(self.buffer) - 1:
                next_ts = self.buffer[i + 1].get("timestamp_sec", last_ts)
                dt = max(0.0, next_ts - curr.get("timestamp_sec", first_ts))
            else:
                dt = window_dur / max(1, len(self.buffer))

            if posture in duration_by_posture:
                duration_by_posture[posture] += dt
            else:
                duration_by_posture["UPRIGHT"] += dt

            tq = curr.get("lumbar_torque_nm", 0.0)
            comp = curr.get("compression_force_n", 0.0)
            tr = curr.get("trunk_flexion_deg", 0.0)
            p2 = curr.get("risk_score", 0.0)

            torques.append(tq)
            compressions.append(comp)
            trunk_angles.append(tr)
            p2_scores.append(p2)

            torque_impulse += tq * dt

        awkward_time = (
            duration_by_posture["BENDING"]
            + duration_by_posture["LIFTING"]
            + duration_by_posture["SQUATTING"]
        )
        awkward_duty_pct = min(100.0, (awkward_time / window_dur) * 100.0)

        # Filter repetition events falling within current window
        cutoff = last_ts - self.window_seconds
        recent_cycles = [c for c in self.cycle_timestamps if c[0] >= cutoff]
        rep_count = len(recent_cycles)
        rep_rate_per_min = (rep_count / max(0.01, window_dur)) * 60.0

        return {
            "window_duration_sec": round(window_dur, 2),
            "frame_count": len(self.buffer),
            "duration_upright_sec": round(duration_by_posture["UPRIGHT"], 2),
            "duration_bending_sec": round(duration_by_posture["BENDING"], 2),
            "duration_squatting_sec": round(duration_by_posture["SQUATTING"], 2),
            "duration_lifting_sec": round(duration_by_posture["LIFTING"], 2),
            "awkward_duty_cycle_pct": round(awkward_duty_pct, 1),
            "repetition_count": rep_count,
            "rep_rate_per_min": round(rep_rate_per_min, 1),
            "mean_lumbar_torque_nm": round(float(sum(torques) / len(torques)), 1) if torques else 0.0,
            "peak_lumbar_torque_nm": round(float(max(torques)), 1) if torques else 0.0,
            "mean_spinal_compression_n": round(float(sum(compressions) / len(compressions)), 1) if compressions else 0.0,
            "peak_spinal_compression_n": round(float(max(compressions)), 1) if compressions else 0.0,
            "mean_trunk_flexion_deg": round(float(sum(trunk_angles) / len(trunk_angles)), 1) if trunk_angles else 0.0,
            "peak_trunk_flexion_deg": round(float(max(trunk_angles)), 1) if trunk_angles else 0.0,
            "mean_phase2_risk_score": round(float(sum(p2_scores) / len(p2_scores)), 1) if p2_scores else 0.0,
            "cumulative_torque_impulse_nms": round(torque_impulse, 1),
            "sustained_awkward_sec": round(self.sustained_awkward_sec, 2),
            "cumulative_strain_index": round(self.cumulative_strain, 1),
            "is_estimated_model_indicator": True,
            "disclaimer": "Temporal metrics and torque impulses are modeled biomechanical estimates and not direct medical or clinical measurements.",
        }
