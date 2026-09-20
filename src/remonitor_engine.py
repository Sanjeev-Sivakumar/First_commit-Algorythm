"""
KineticGuard - Re-Monitoring & Closed-Loop Effectiveness Engine
Tracks subsequent real video telemetry to verify whether applied safety interventions
succeeded in reducing ergonomic risk, lumbar torque, and spinal compression.

Strict Rules:
  - If insufficient post-intervention frames exist (< min_required_frames),
    strictly return WAITING_FOR_SUFFICIENT_DATA.
  - NEVER fabricate effectiveness data or assume success without actual measurements.
  - Update OpenSearch intervention record with true calculated before/after deltas.
"""

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from .local_opensearch import (
    KineticGuardOpenSearch,
    INDEX_INTERVENTIONS,
    opensearch_engine,
)


class ClosedLoopRemonitorEngine:
    """Evaluates the measurable outcome of safety interventions using real telemetry."""

    STATUS_WAITING = "WAITING_FOR_SUFFICIENT_DATA"
    STATUS_EFFECTIVE = "EFFECTIVE"
    STATUS_INEFFECTIVE = "INEFFECTIVE"

    def __init__(self, opensearch: Optional[KineticGuardOpenSearch] = None):
        self.opensearch = opensearch or opensearch_engine

    def evaluate_intervention_effectiveness(
        self,
        intervention_id: str,
        subsequent_frames: List[Dict[str, Any]],
        min_required_frames: int = 20,
    ) -> Dict[str, Any]:
        """
        Evaluates intervention effectiveness using real subsequent telemetry frames.
        Returns WAITING_FOR_SUFFICIENT_DATA if insufficient frames are observed.
        """
        # 1. Fetch intervention record from OpenSearch
        record = self.opensearch.get_document(INDEX_INTERVENTIONS, intervention_id)
        if not record:
            return {
                "error": f"Intervention {intervention_id} not found in OpenSearch index {INDEX_INTERVENTIONS}",
                "status": "NOT_FOUND",
            }

        pre_metrics = record.get("pre_intervention_metrics", {})
        frames_count = len(subsequent_frames)

        # 2. Insufficient Data Guard - NEVER fabricate effectiveness
        if frames_count < min_required_frames:
            waiting_result = {
                "intervention_id": intervention_id,
                "worker_id": record.get("worker_id"),
                "station_id": record.get("station_id"),
                "status": self.STATUS_WAITING,
                "message": (
                    f"Observed {frames_count} post-intervention frames; minimum {min_required_frames} "
                    f"required for statistically valid ergonomic comparison."
                ),
                "frames_observed": frames_count,
                "frames_required": min_required_frames,
                "is_effective": None,
                "risk_reduction_pct": None,
                "pre_intervention_metrics": pre_metrics,
                "post_intervention_metrics": None,
            }
            # Record state in OpenSearch
            record["effectiveness_status"] = self.STATUS_WAITING
            record["frames_observed"] = frames_count
            self.opensearch.index_intervention(record)
            return waiting_result

        # 3. Compute Real Post-Intervention Metrics from Actual Frames
        scores = []
        torques = []
        compressions = []
        awkward_frames = 0

        for f in subsequent_frames:
            s = f.get("risk_score") or f.get("mean_phase2_risk_score", 0.0)
            t = f.get("lumbar_torque_nm", 0.0)
            c = f.get("compression_force_n") or f.get("spinal_compression_n", 0.0)
            posture = f.get("posture", "UPRIGHT")

            scores.append(float(s))
            torques.append(float(t))
            compressions.append(float(c))
            if posture in ("BENDING", "LIFTING", "SQUATTING"):
                awkward_frames += 1

        post_mean_score = sum(scores) / max(1, len(scores))
        post_peak_torque = max(torques) if torques else 0.0
        post_mean_comp = sum(compressions) / max(1, len(compressions))
        post_awkward_duty = (awkward_frames / max(1, len(subsequent_frames))) * 100.0

        post_metrics = {
            "cumulative_risk_score": round(post_mean_score, 1),
            "peak_lumbar_torque_nm": round(post_peak_torque, 1),
            "mean_spinal_compression_n": round(post_mean_comp, 1),
            "awkward_duty_cycle_pct": round(post_awkward_duty, 1),
            "frames_analyzed": frames_count,
        }

        # 4. Comparative Effectiveness Analysis
        pre_score = float(pre_metrics.get("cumulative_risk_score", 50.0))
        risk_delta = round(post_mean_score - pre_score, 1)
        risk_reduction_pct = round(((pre_score - post_mean_score) / max(0.01, pre_score)) * 100.0, 1)

        # Objective criterion: >= 10% risk reduction OR dropping below moderate threshold (< 50)
        is_effective = (risk_reduction_pct >= 10.0) or (post_mean_score < 50.0 and pre_score >= 55.0)
        eff_status = self.STATUS_EFFECTIVE if is_effective else self.STATUS_INEFFECTIVE

        # 5. Update OpenSearch Persistent Intervention Record
        record["status"] = "RESOLVED"
        record["effectiveness_status"] = eff_status
        record["is_effective"] = is_effective
        record["risk_reduction_pct"] = risk_reduction_pct
        record["post_intervention_metrics"] = post_metrics
        record["evaluated_at"] = datetime.now(timezone.utc).isoformat()
        self.opensearch.index_intervention(record)

        return {
            "intervention_id": intervention_id,
            "worker_id": record.get("worker_id"),
            "station_id": record.get("station_id"),
            "status": eff_status,
            "is_effective": is_effective,
            "risk_reduction_pct": risk_reduction_pct,
            "risk_delta": risk_delta,
            "frames_analyzed": frames_count,
            "pre_intervention_metrics": pre_metrics,
            "post_intervention_metrics": post_metrics,
            "summary": (
                f"Intervention {intervention_id} is {eff_status}: Risk score changed from {pre_score:.1f} "
                f"to {post_mean_score:.1f} ({risk_reduction_pct:+.1f}% change across {frames_count} frames)."
            ),
        }
