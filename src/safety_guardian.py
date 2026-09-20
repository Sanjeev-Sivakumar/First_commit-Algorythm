"""
KineticGuard - Safety Guardian Agent (Phase 6)
Autonomous workplace safety decision layer that reasons over trusted telemetry,
Worker Ergonomic Passports, Station Profiles, and Safety Policies.

The agent does NOT calculate or override biomechanical measurements; Phase 2/3
deterministic calculations remain 100% authoritative.

Decisions:
  - LOG: Low risk, normal operational baseline.
  - NOTIFY_SUPERVISOR: Moderate risk or initial minor threshold breach.
  - RECOMMEND_INTERVENTION: High ergonomic exposure, duty cycle > 80%, sustained awkward posture.
  - ESCALATE: Critical spinal load (>= 3400 N NIOSH limit), fatigue index > 80, repeated high-risk events.

Produces structured JSON: decision, reason, risk_factors, recommended_action, evidence, confidence.
"""

import json
from typing import Dict, List, Any, Optional


class SafetyGuardianAgent:
    """Evaluates ergonomic incidents in organizational context and determines operational actions."""

    DECISION_LOG = "LOG"
    DECISION_NOTIFY = "NOTIFY_SUPERVISOR"
    DECISION_RECOMMEND = "RECOMMEND_INTERVENTION"
    DECISION_ESCALATE = "ESCALATE"

    def __init__(self):
        pass

    def evaluate(
        self,
        incident: Dict[str, Any],
        worker_passport: Optional[Dict[str, Any]] = None,
        station_profile: Optional[Dict[str, Any]] = None,
        recent_interventions: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Evaluates incident context and generates autonomous safety decision.
        
        Phase 2/3 deterministic numbers are treated as immutable ground truth.
        """
        auth = incident.get("authoritative_metrics", {})
        vlm = incident.get("vlm_analysis", {})

        score = float(auth.get("cumulative_risk_score", incident.get("cumulative_risk_score", 50.0)))
        risk_level = auth.get("risk_level", incident.get("risk_level", "MODERATE"))
        torque = float(auth.get("peak_lumbar_torque_nm", 75.0))
        comp = float(auth.get("peak_spinal_compression_n", 2000.0))
        duty_cycle = float(auth.get("awkward_duty_cycle_pct", 70.0))
        rep_rate = float(auth.get("repetition_rate_per_min", 10.0))

        worker_id = incident.get("worker_id", "WORKER-UNKNOWN")
        station_id = incident.get("station_id", "STATION-UNKNOWN")
        incident_id = incident.get("incident_id", "INC-UNKNOWN")

        passport = worker_passport or {}
        station = station_profile or {}
        interventions = recent_interventions or []

        fatigue_index = float(passport.get("fatigue_index", 30.0))
        worker_incidents = int(passport.get("total_incidents", 1))
        high_risk_incidents = int(passport.get("high_risk_incidents", 0))

        # Check for failed/ineffective prior interventions
        has_failed_intervention = any(
            i.get("effectiveness_status") in ("INEFFECTIVE", "ESCALATED") for i in interventions
        )

        evidence = [
            f"Authoritative Cumulative Risk Score: {score:.1f}/100 ({risk_level})",
            f"Peak Spinal Compressive Force: {comp:.0f} N (NIOSH Action Limit: 3400 N)",
            f"Awkward Posture Duty Cycle: {duty_cycle:.1f}%",
            f"Worker Cumulative Fatigue Index: {fatigue_index:.1f}/100",
            f"Station Total Incidents: {station.get('total_incidents', 1)}",
        ]

        risk_factors = []
        if comp >= 3400.0:
            risk_factors.append(f"Spinal compression ({comp:.0f} N) strictly breaches NIOSH Action Limit (3400 N)")
        elif comp >= 2200.0:
            risk_factors.append(f"Elevated spinal compression ({comp:.0f} N) nearing safety action limit")

        if duty_cycle >= 85.0:
            risk_factors.append(f"Excessive awkward duty cycle ({duty_cycle:.1f}%) in non-neutral posture")
        if torque >= 90.0:
            risk_factors.append(f"High lumbar moment ({torque:.1f} Nm) at L5/S1 vertebral disc")
        if rep_rate >= 15.0:
            risk_factors.append(f"Elevated movement frequency ({rep_rate:.1f} reps/min) with limited recovery interval")

        # Decision Logic
        if comp >= 3400.0 or fatigue_index >= 80.0 or (high_risk_incidents >= 2 and score >= 70.0) or has_failed_intervention:
            decision = self.DECISION_ESCALATE
            confidence = 0.98
            reason = (
                f"CRITICAL SAFETY THRESHOLD BREACH: Worker {worker_id} at {station_id} has exceeded critical ergonomic "
                f"limits (Spinal Load: {comp:.0f} N, Fatigue Index: {fatigue_index:.1f}/100). "
                f"{'Prior intervention failed to mitigate risk. ' if has_failed_intervention else ''}"
                f"Immediate administrative and engineering escalation required."
            )
            recommended_action = (
                "IMMEDIATE ACTION: Halt manual picking operations at this station immediately; enforce mandatory 20-minute "
                "postural recovery break for operative and deploy EHS supervisor for station mechanical lift inspection."
            )
        elif score >= 60.0 or duty_cycle >= 80.0 or risk_level == "HIGH":
            decision = self.DECISION_RECOMMEND
            confidence = 0.94
            reason = (
                f"HIGH RISK DETECTED: Operative {worker_id} is sustaining {duty_cycle:.1f}% awkward duty cycle "
                f"with peak lumbar torque of {torque:.1f} Nm. Cumulative muscle fatigue accumulation is accelerating."
            )
            recommended_action = (
                vlm.get("recommended_action")
                or "Enforce immediate task rotation to non-manual sorting and adjust pallet elevation by 35-45 cm."
            )
        elif score >= 30.0 or risk_level == "MODERATE":
            decision = self.DECISION_NOTIFY
            confidence = 0.91
            reason = (
                f"MODERATE EXPOSURE: Station {station_id} exhibits borderline ergonomic strain (Score: {score:.1f}/100). "
                f"Notifying shift supervisor for proactive station monitoring."
            )
            recommended_action = (
                "Log station metrics and schedule planned worker rotation within the next 45 minutes."
            )
        else:
            decision = self.DECISION_LOG
            confidence = 0.96
            reason = f"Normal ergonomic baseline observed for worker {worker_id} (Score: {score:.1f}/100)."
            recommended_action = "Continue routine shift surveillance."

        return {
            "decision": decision,
            "reason": reason,
            "risk_factors": risk_factors or ["Routine manual handling"],
            "recommended_action": recommended_action,
            "evidence": evidence,
            "confidence": confidence,
        }
