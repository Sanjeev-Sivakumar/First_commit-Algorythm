"""
KineticGuard - Cumulative Ergonomic Risk Model
Formulates a normalized 0–100 cumulative risk score combining instantaneous biomechanics,
sustained awkward posture durations, movement repetition frequency, and torque impulse.
"""

from typing import Any, Dict, List, Tuple


class CumulativeRiskScore:
    """Holds computed cumulative score, risk category, and contributing factors."""

    def __init__(
        self,
        score: float,
        level: str,
        sub_scores: Dict[str, float],
        contributing_factors: List[str],
    ):
        self.score = round(score, 1)
        self.level = level
        self.sub_scores = sub_scores
        self.contributing_factors = contributing_factors

    def to_dict(self) -> Dict[str, Any]:
        return {
            "cumulative_risk_score": self.score,
            "risk_level": self.level,
            "sub_scores": {k: round(v, 1) for k, v in self.sub_scores.items()},
            "contributing_factors": self.contributing_factors,
            "is_estimated_model_indicator": True,
            "disclaimer": "Cumulative ergonomic risk score is a modeled biomechanical indicator, not a clinical diagnostic measurement.",
        }


class CumulativeRiskModel:
    """
    Computes a multi-dimensional cumulative ergonomic risk score based on
    ergonomic biomechanical standards (NIOSH / RULA / ACGIH).
    """

    def __init__(
        self,
        w_biomech: float = 0.35,
        w_duration: float = 0.30,
        w_repetition: float = 0.20,
        w_impulse: float = 0.15,
    ):
        self.w_biomech = w_biomech
        self.w_duration = w_duration
        self.w_repetition = w_repetition
        self.w_impulse = w_impulse

    def evaluate(self, temporal_metrics: Dict[str, Any]) -> CumulativeRiskScore:
        """
        Calculates cumulative 0–100 risk score and identifies contributing risk factors.
        Enforces:
        - Movement != Risk
        - Single movement / brief bending (e.g. 2s) -> LOW (< 30.0)
        - 10s intermittent bending -> LOW / MODERATE (30.0 - 48.0)
        - 30s repeated bending -> MODERATE (50.0 - 65.0)
        - Sustained awkward posture + high load + repetition -> HIGH (>= 70.0)
        - Sustained high biomechanical exposure (>3400N NIOSH AL) -> HIGH / DANGEROUS (>= 80.0)
        - Never classify HIGH from a single frame or brief movement.
        """
        awkward_duty_pct = float(temporal_metrics.get("awkward_duty_cycle_pct", 0.0))
        rep_rate = float(temporal_metrics.get("rep_rate_per_min", 0.0))
        rep_count = int(temporal_metrics.get("repetition_count", 0))
        peak_comp = float(temporal_metrics.get("peak_spinal_compression_n", 0.0))
        mean_comp = float(temporal_metrics.get("mean_spinal_compression_n", 0.0))
        peak_torque = float(temporal_metrics.get("peak_lumbar_torque_nm", 0.0))
        peak_trunk = float(temporal_metrics.get("peak_trunk_flexion_deg", 0.0))
        impulse = float(temporal_metrics.get("cumulative_torque_impulse_nms", 0.0))
        window_dur = max(1.0, float(temporal_metrics.get("window_duration_sec", 20.0)))
        p2_score = float(temporal_metrics.get("mean_phase2_risk_score", 0.0))
        sustained_sec = float(temporal_metrics.get("sustained_awkward_sec", 0.0))
        strain_index = float(temporal_metrics.get("cumulative_strain_index", 0.0))

        # 1. Biomechanical Loading Sub-Score (0 - 100)
        # Prioritizes sustained spinal loading; incorporates peak when sustained
        s_comp_mean = min(100.0, max(0.0, (mean_comp - 750.0) / 2650.0) * 80.0)
        if peak_comp >= 3400.0 and (awkward_duty_pct >= 35.0 or sustained_sec >= 2.5):
            s_comp_peak = min(100.0, (peak_comp / 3400.0) * 85.0)
            s_biomech = max(s_comp_mean, s_comp_peak)
        elif peak_comp >= 2500.0 and awkward_duty_pct >= 25.0:
            s_comp_peak = min(100.0, (peak_comp / 3400.0) * 65.0)
            s_biomech = max(s_comp_mean, s_comp_peak)
        else:
            s_biomech = max(s_comp_mean, min(50.0, p2_score * 1.0))

        # 2. Posture Duration Sub-Score (0 - 100)
        # Non-linear exposure scaling: brief exposure (<15% duty) generates minimal score
        if awkward_duty_pct <= 15.0:
            s_duration = (awkward_duty_pct / 15.0) * 10.0
        elif awkward_duty_pct <= 50.0:
            s_duration = 10.0 + ((awkward_duty_pct - 15.0) / 35.0) * 45.0
        else:
            s_duration = 55.0 + ((awkward_duty_pct - 50.0) / 50.0) * 45.0

        # 3. Repetition Rate Sub-Score (0 - 100)
        # Ergonomic standard: <3 reps/min is baseline, 4-8 is moderate, >=10 is high
        if rep_rate <= 2.0:
            s_rep = (rep_rate / 2.0) * 15.0
        elif rep_rate <= 7.0:
            s_rep = 15.0 + ((rep_rate - 2.0) / 5.0) * 45.0
        else:
            s_rep = min(100.0, 60.0 + ((rep_rate - 7.0) / 7.0) * 40.0)

        # 4. Torque Impulse Sub-Score (0 - 100)
        # Accounts for active torque above neutral baseline (~20 Nm)
        eff_mean_torque = max(0.0, (impulse / window_dur) - 20.0)
        s_impulse = min(100.0, (eff_mean_torque / 75.0) * 100.0)

        # Base Weighted Score
        raw_score = (
            self.w_biomech * s_biomech
            + self.w_duration * s_duration
            + self.w_repetition * s_rep
            + self.w_impulse * s_impulse
        )

        # Temporal Persistence / Evidence Gate:
        # Prevent high risk from short bursts:
        # - Brief awkward duration (< 3s) without sustained strain: cap at LOW (< 30)
        # - Temporary exposure (< 10s, duty < 45%, non-critical compression): cap at MODERATE (< 60)
        dur_awkward = (awkward_duty_pct / 100.0) * window_dur
        if dur_awkward < 3.0 and strain_index < 10.0:
            final_score = min(28.0, raw_score)
        elif dur_awkward < 10.0 and awkward_duty_pct < 45.0 and peak_comp < 3400.0 and strain_index < 25.0:
            final_score = min(55.0, raw_score)
        else:
            final_score = max(0.0, min(100.0, raw_score))

        # Categorize Risk Level
        if final_score < 30.0:
            level = "LOW"
        elif final_score < 65.0:
            level = "MODERATE"
        elif final_score < 80.0:
            level = "HIGH"
        else:
            level = "DANGEROUS"

        # Identify Specific Contributing Factors
        factors = []
        if peak_comp >= 3400.0:
            factors.append(f"Peak spinal compression ({peak_comp:.0f} N) exceeds NIOSH Action Limit (3,400 N)")
        elif peak_comp >= 2500.0:
            factors.append(f"Elevated spinal compression ({peak_comp:.0f} N) approaching safety threshold")

        if peak_trunk >= 45.0:
            factors.append(f"Excessive trunk flexion (peak {peak_trunk:.1f} deg exceeds 45 deg threshold)")
        elif peak_trunk >= 30.0:
            factors.append(f"Awkward torso forward flexion (peak {peak_trunk:.1f} deg)")

        if awkward_duty_pct >= 40.0:
            factors.append(f"High awkward posture duty cycle ({awkward_duty_pct:.1f}% of window in bending/lifting)")

        if rep_rate >= 6.0:
            factors.append(f"Repetitive motion frequency ({rep_rate:.1f} reps/min, {rep_count} cycles in window)")

        if peak_torque >= 90.0:
            factors.append(f"High peak lumbar moment ({peak_torque:.1f} Nm at L5/S1)")

        if not factors:
            factors.append("Biomechanical parameters within normal ergonomic tolerance")

        return CumulativeRiskScore(
            score=final_score,
            level=level,
            sub_scores={
                "biomechanical_loading": s_biomech,
                "posture_duration": s_duration,
                "repetition_rate": s_rep,
                "torque_impulse": s_impulse,
            },
            contributing_factors=factors,
        )
