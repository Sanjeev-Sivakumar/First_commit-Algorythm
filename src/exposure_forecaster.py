"""
KineticGuard - Near-Term Exposure Forecasting (Phase 6)
Forecasts worker ergonomic exposure accumulation over the next 30-60 minutes
using current telemetry, repetition rates, duty cycle trends, and worker passport memory.

Strict Rule: This is EXPOSURE FORECASTING, not medical injury prediction.
"""

from typing import Dict, Any, Optional, List


class ExposureForecaster:
    """Lightweight deterministic model projecting near-term ergonomic exposure trends."""

    def __init__(self):
        pass

    def forecast_exposure(
        self,
        current_score: float,
        duty_cycle_pct: float,
        repetition_rate_per_min: float,
        fatigue_index: float,
        risk_trend: Optional[List[Dict[str, Any]]] = None,
        worker_id: str = "WORKER-UNKNOWN",
        station_id: str = "STATION-UNKNOWN",
    ) -> Dict[str, Any]:
        """Projects exposure trajectory for the next 30 minutes of active operation."""
        # Check if insufficient data exists
        if current_score <= 0.0 and (not risk_trend or len(risk_trend) == 0):
            return {
                "forecast": "FORECAST: INSUFFICIENT DATA",
                "status": "INSUFFICIENT_DATA",
                "confidence": 0.0,
                "current_score": 0.0,
                "projected_score_30min": 0.0,
                "projected_fatigue_30min": 0.0,
                "reason": "Insufficient historical exposure or telemetry data exists to project trajectory.",
                "proactive_action": "Continue active video surveillance to accumulate operational baseline.",
                "worker_id": worker_id,
                "station_id": station_id,
            }

        # Evaluate trend velocity
        trend_velocity = 0.0
        if risk_trend and len(risk_trend) >= 2:
            s1 = float(risk_trend[-2].get("score", current_score))
            s2 = float(risk_trend[-1].get("score", current_score))
            trend_velocity = s2 - s1  # points per incident/measurement

        # Exposure ramp rate based on cadence and awkward duty cycle
        # Standard warehouse baseline: 6 reps/min, 40% duty cycle -> ~0 drift
        excess_cadence = max(0.0, repetition_rate_per_min - 8.0)
        excess_duty = max(0.0, duty_cycle_pct - 50.0)

        # Projected cumulative score after 30 minutes of continued exposure
        projected_drift = (excess_cadence * 0.4) + (excess_duty * 0.15) + (trend_velocity * 0.5)
        projected_score = min(100.0, max(0.0, current_score + projected_drift))

        # Projected fatigue index
        projected_fatigue = min(100.0, fatigue_index + (projected_drift * 0.8) + 5.0)

        # Categorize forecast
        if projected_score >= 80.0 or projected_fatigue >= 80.0:
            forecast_level = "CRITICAL_ACCUMULATION"
            confidence = 0.92
            reason = (
                f"High cadence of {repetition_rate_per_min:.1f} reps/min combined with {duty_cycle_pct:.1f}% awkward "
                f"duty cycle will drive cumulative exposure score to {projected_score:.1f} within 30 minutes. "
                f"Worker fatigue index will reach critical saturation ({projected_fatigue:.1f}/100)."
            )
            action = (
                "Schedule proactive worker rotation to non-manual sorting within the next 15 minutes before "
                "fatigue accumulation exceeds permissible threshold."
            )
        elif projected_score >= 60.0 or duty_cycle_pct >= 75.0:
            forecast_level = "ELEVATED_EXPOSURE"
            confidence = 0.88
            reason = (
                f"Current ergonomic duty cycle ({duty_cycle_pct:.1f}%) will lead to elevated fatigue accumulation "
                f"(projected score: {projected_score:.1f}) over the upcoming operational block."
            )
            action = "Introduce a 5-minute ergonomic micro-recovery break and reduce conveyor pacing."
        else:
            forecast_level = "STABLE_EXPOSURE"
            confidence = 0.94
            reason = (
                f"Biomechanical cadence and posture intervals remain within acceptable physiological recovery ranges "
                f"(projected score: {projected_score:.1f})."
            )
            action = "Continue standard shift operations and scheduled rest breaks."

        return {
            "forecast": forecast_level,
            "confidence": confidence,
            "worker_id": worker_id,
            "station_id": station_id,
            "current_score": round(current_score, 1),
            "projected_score_30min": round(projected_score, 1),
            "projected_fatigue_30min": round(projected_fatigue, 1),
            "reason": reason,
            "recommended_action": action,
            "proactive_action": action,
        }
