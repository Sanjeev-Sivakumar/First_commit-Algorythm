"""
KineticGuard - What-If Ergonomic Simulator (Phase 6)
Allows supervisors and safety engineers to simulate hypothetical workstation
and process adjustments, estimating risk reduction using deterministic Phase 2/3 formulas.

Notice: All output values are ESTIMATED / SIMULATED outcomes based on biomechanical
modeling, not actual measured field observations. No generative LLMs are used for math.
"""

import math
from typing import Dict, Any, Optional


class WhatIfSimulator:
    """Deterministic mathematical simulator for workstation and operational ergonomic modifications."""

    def __init__(self):
        self.g = 9.80665
        self.worker_mass_kg = 75.0
        self.torso_length_m = 0.55
        self.torso_com_dist_m = 0.28
        self.erector_moment_arm_m = 0.05  # 5 cm L5/S1 muscle lever arm

    def simulate(
        self,
        current_baseline: Optional[Dict[str, Any]] = None,
        modifications: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Calculates current vs simulated biomechanical and cumulative risk profile."""
        baseline = current_baseline or {}
        mods = modifications or {}

        # 1. Baseline parameters
        base_load_kg = float(baseline.get("load_weight_kg", 12.0))
        base_pallet_h_cm = float(baseline.get("pallet_height_cm", 45.0))
        base_reps_per_min = float(baseline.get("reps_per_minute", 18.0))
        base_duration_min = float(baseline.get("task_duration_minutes", 60.0))
        base_rotation = bool(baseline.get("worker_rotation_enabled", False))
        base_trunk_angle = float(baseline.get("trunk_flexion_deg", 40.6))
        base_duty_cycle = float(baseline.get("awkward_duty_cycle_pct", 93.5))

        # 2. Simulated parameters (apply user modifications)
        sim_load_kg = float(mods.get("load_weight_kg", base_load_kg))
        sim_pallet_h_cm = float(mods.get("pallet_height_cm", base_pallet_h_cm))
        sim_reps_per_min = float(mods.get("reps_per_minute", base_reps_per_min))
        sim_duration_min = float(mods.get("task_duration_minutes", base_duration_min))
        sim_rotation = bool(mods.get("worker_rotation_enabled", base_rotation))

        # Calculate current baseline biomechanics
        curr_res = self._compute_biomechanics(
            load_kg=base_load_kg,
            pallet_h_cm=base_pallet_h_cm,
            reps_min=base_reps_per_min,
            duration_min=base_duration_min,
            rotation=base_rotation,
            initial_trunk_deg=base_trunk_angle,
            initial_duty_pct=base_duty_cycle,
        )

        # Calculate simulated biomechanics
        sim_res = self._compute_biomechanics(
            load_kg=sim_load_kg,
            pallet_h_cm=sim_pallet_h_cm,
            reps_min=sim_reps_per_min,
            duration_min=sim_duration_min,
            rotation=sim_rotation,
            initial_trunk_deg=base_trunk_angle,
            initial_duty_pct=base_duty_cycle,
        )

        # Percentage change
        curr_score = curr_res["cumulative_risk_score"]
        sim_score = sim_res["cumulative_risk_score"]
        if curr_score > 0:
            change_pct = round(((sim_score - curr_score) / curr_score) * 100.0, 1)
        else:
            change_pct = 0.0

        return {
            "status": "SIMULATED_ESTIMATE",
            "label": "MODELLED / SIMULATED",
            "modeled_label": "MODELLED / SIMULATED",
            "disclaimer": "MODELLED / SIMULATED: All simulated metrics are deterministic mathematical projections based on biomechanical modeling, not measured clinical outcomes.",
            "current_configuration": {
                "load_weight_kg": base_load_kg,
                "pallet_height_cm": base_pallet_h_cm,
                "reps_per_minute": base_reps_per_min,
                "task_duration_minutes": base_duration_min,
                "worker_rotation_enabled": base_rotation,
                "estimated_trunk_flexion_deg": curr_res["trunk_flexion_deg"],
                "estimated_lumbar_torque_nm": curr_res["lumbar_torque_nm"],
                "estimated_spinal_compression_n": curr_res["spinal_compression_n"],
                "estimated_awkward_duty_cycle_pct": curr_res["awkward_duty_cycle_pct"],
                "cumulative_risk_score": curr_res["cumulative_risk_score"],
                "risk_category": curr_res["risk_category"],
            },
            "simulated_configuration": {
                "load_weight_kg": sim_load_kg,
                "pallet_height_cm": sim_pallet_h_cm,
                "reps_per_minute": sim_reps_per_min,
                "task_duration_minutes": sim_duration_min,
                "worker_rotation_enabled": sim_rotation,
                "estimated_trunk_flexion_deg": sim_res["trunk_flexion_deg"],
                "estimated_lumbar_torque_nm": sim_res["lumbar_torque_nm"],
                "estimated_spinal_compression_n": sim_res["spinal_compression_n"],
                "estimated_awkward_duty_cycle_pct": sim_res["awkward_duty_cycle_pct"],
                "cumulative_risk_score": sim_res["cumulative_risk_score"],
                "risk_category": sim_res["risk_category"],
            },
            "risk_reduction_pct": change_pct,
            "key_benefits": self._generate_benefits(curr_res, sim_res),
        }

    def _compute_biomechanics(
        self,
        load_kg: float,
        pallet_h_cm: float,
        reps_min: float,
        duration_min: float,
        rotation: bool,
        initial_trunk_deg: float,
        initial_duty_pct: float,
    ) -> Dict[str, Any]:
        """Calculates deterministic L5/S1 moments, compression, and cumulative score."""
        # 1. Trunk inclination adjusted by pallet elevation
        # Ground/low pallet is ~45 cm. Standing knuckle/waist height is ~85 cm.
        delta_h_cm = max(0.0, pallet_h_cm - 45.0)
        # Reduction in flexion angle: approx 0.55 deg per cm of pallet elevation
        angle_reduction = min(initial_trunk_deg - 8.0, delta_h_cm * 0.55)
        trunk_deg = max(8.0, initial_trunk_deg - angle_reduction)
        trunk_rad = math.radians(trunk_deg)

        # 2. Torso and Load forces
        f_trunk = 0.45 * self.worker_mass_kg * self.g
        f_load = load_kg * self.g

        # 3. Moment arms
        d_trunk = self.torso_com_dist_m * math.sin(trunk_rad)
        # Load distance from spine shortens as torso straightens
        d_load = max(0.20, 0.45 * math.cos(math.radians(angle_reduction * 0.5)))

        # 4. Lumbar moment & compressive load
        torque_nm = (f_trunk * d_trunk) + (f_load * d_load)
        compression_n = (torque_nm / self.erector_moment_arm_m) + ((f_trunk + f_load) * math.cos(trunk_rad))

        # 5. Duty cycle adjustment
        duty_pct = initial_duty_pct
        if angle_reduction > 15.0:
            duty_pct = max(15.0, duty_pct * 0.45)
        elif angle_reduction > 5.0:
            duty_pct = max(25.0, duty_pct * 0.70)

        # Worker rotation cuts sustained awkward duty cycle in half
        if rotation:
            duty_pct = max(12.0, duty_pct * 0.50)

        # Duration adjustment (scaling down for short tasks)
        if duration_min <= 20.0:
            duty_pct = max(10.0, duty_pct * 0.8)

        # 6. Cumulative Ergonomic Risk Score formula (Phase 3)
        s_duty = min(100.0, duty_pct)
        s_rep = min(100.0, (reps_min / 15.0) * 100.0)
        s_load = min(100.0, (compression_n / 3400.0) * 100.0)
        s_peak = min(100.0, (torque_nm / 170.0) * 100.0)

        score = (0.35 * s_duty) + (0.25 * s_rep) + (0.25 * s_load) + (0.15 * s_peak)
        score = round(min(100.0, max(0.0, score)), 1)

        # Category
        if score >= 85.0:
            cat = "DANGEROUS"
        elif score >= 60.0:
            cat = "HIGH"
        elif score >= 30.0:
            cat = "MODERATE"
        else:
            cat = "LOW"

        return {
            "trunk_flexion_deg": round(trunk_deg, 1),
            "lumbar_torque_nm": round(torque_nm, 1),
            "spinal_compression_n": round(compression_n, 1),
            "awkward_duty_cycle_pct": round(duty_pct, 1),
            "cumulative_risk_score": score,
            "risk_category": cat,
        }

    def _generate_benefits(self, curr: Dict[str, Any], sim: Dict[str, Any]) -> list:
        benefits = []
        torque_drop = curr["lumbar_torque_nm"] - sim["lumbar_torque_nm"]
        if torque_drop > 10.0:
            benefits.append(f"Reduces L5/S1 spinal moment by {torque_drop:.1f} Nm ({round((torque_drop/curr['lumbar_torque_nm'])*100, 1)}% reduction).")

        comp_drop = curr["spinal_compression_n"] - sim["spinal_compression_n"]
        if comp_drop > 200.0:
            benefits.append(f"Decreases compressive intervertebral load by {comp_drop:.0f} N.")

        duty_drop = curr["awkward_duty_cycle_pct"] - sim["awkward_duty_cycle_pct"]
        if duty_drop > 15.0:
            benefits.append(f"Lowers awkward posture exposure duration from {curr['awkward_duty_cycle_pct']}% to {sim['awkward_duty_cycle_pct']}%.")

        if sim["risk_category"] != curr["risk_category"]:
            benefits.append(f"Transitions overall ergonomic exposure from {curr['risk_category']} down to {sim['risk_category']}.")

        return benefits or ["Maintains ergonomic conditions within standard baseline tolerances."]
