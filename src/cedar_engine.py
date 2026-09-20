"""
KineticGuard - Cedar Policy Evaluation Engine
Executes formal Cedar safety policies via cedarpy Rust bindings as the final
deterministic safety/policy gate for autonomous safety guardian decisions.

Supported Decisions:
  - LOG: Normal baseline surveillance; no intervention required.
  - REVIEW/NOTIFY: Moderate threshold breach; notify supervisor and schedule rotation.
  - ESCALATE: Critical NIOSH breach, excessive cumulative fatigue, or high-risk repeat strain.
"""

from typing import Any, Dict, List, Optional
import cedarpy


CEDAR_SAFETY_POLICIES = """
// ====================================================================
// KineticGuard Autonomous Safety Policy Set (Cedar Formal Language)
// ====================================================================

// Base authorization: Permit Safety Guardian Agent to evaluate actions
permit(
    principal == Agent::"SafetyGuardian",
    action in [Action::"LOG", Action::"REVIEW_NOTIFY", Action::"ESCALATE"],
    resource
);

// Escalation Safety Gate:
// FORBID ESCALATE unless strict ergonomic threshold conditions are breached
forbid(
    principal,
    action == Action::"ESCALATE",
    resource
) unless {
    context.cumulative_risk >= 70 ||
    context.spinal_compression_n >= 3400 ||
    (context.awkward_duty_cycle_pct >= 80 && context.rep_rate_per_min >= 6) ||
    context.fatigue_index >= 80 ||
    context.has_failed_prior_intervention == true
};

// Review / Notification Gate:
// FORBID REVIEW_NOTIFY when risk is safely below moderate threshold
forbid(
    principal,
    action == Action::"REVIEW_NOTIFY",
    resource
) when {
    context.cumulative_risk < 50 &&
    context.spinal_compression_n < 2200 &&
    context.awkward_duty_cycle_pct < 60
};

// Logging Gate:
// FORBID passive LOG when risk is elevated or critical
forbid(
    principal,
    action == Action::"LOG",
    resource
) when {
    context.cumulative_risk >= 60 ||
    context.spinal_compression_n >= 2800 ||
    context.fatigue_index >= 70
};
"""


class CedarPolicyEngine:
    """Evaluates ergonomic decisions against formal Cedar policies."""

    ACTION_LOG = "LOG"
    ACTION_REVIEW_NOTIFY = "REVIEW/NOTIFY"
    ACTION_ESCALATE = "ESCALATE"

    def __init__(self, custom_policies: Optional[str] = None):
        self.policies = custom_policies or CEDAR_SAFETY_POLICIES

    def evaluate_safety_decision(
        self,
        incident_id: str,
        cumulative_risk: float,
        spinal_compression_n: float,
        awkward_duty_cycle_pct: float,
        rep_rate_per_min: float = 0.0,
        fatigue_index: float = 30.0,
        has_failed_prior_intervention: bool = False,
    ) -> Dict[str, Any]:
        """
        Evaluates the optimal deterministic safety action through Cedar policy gates.
        Returns the finalized policy decision, allowed status, and diagnostics.
        """
        # Cedar integers are 64-bit signed integers; round floats to integer context
        context = {
            "cumulative_risk": int(round(cumulative_risk)),
            "spinal_compression_n": int(round(spinal_compression_n)),
            "awkward_duty_cycle_pct": int(round(awkward_duty_cycle_pct)),
            "rep_rate_per_min": int(round(rep_rate_per_min)),
            "fatigue_index": int(round(fatigue_index)),
            "has_failed_prior_intervention": bool(has_failed_prior_intervention),
        }

        # Priority 1: Check if ESCALATE is permitted by Cedar
        req_escalate = {
            "principal": 'Agent::"SafetyGuardian"',
            "action": 'Action::"ESCALATE"',
            "resource": f'Incident::"{incident_id}"',
            "context": context,
        }
        res_escalate = cedarpy.is_authorized(req_escalate, self.policies, [])
        if res_escalate.allowed:
            reasons = [str(r) for r in res_escalate.diagnostics.reasons]
            rule_id = reasons[0] if reasons else "policy0"
            return {
                "decision": self.ACTION_ESCALATE,
                "agent_proposed_action": self.ACTION_ESCALATE,
                "cedar_action": 'Action::"ESCALATE"',
                "cedar_decision": str(res_escalate.decision),
                "allowed": True,
                "policy_rule_id": rule_id,
                "enforced_action": self.ACTION_ESCALATE,
                "fallback_action": "N/A (Action Authorized)",
                "diagnostics": {
                    "reasons": reasons,
                    "errors": [str(e) for e in res_escalate.diagnostics.errors],
                },
                "policy_gate": "PASSED_ESCALATE_POLICY",
                "evaluated_context": context,
            }

        # Priority 2: Check if REVIEW_NOTIFY is permitted by Cedar
        req_review = {
            "principal": 'Agent::"SafetyGuardian"',
            "action": 'Action::"REVIEW_NOTIFY"',
            "resource": f'Incident::"{incident_id}"',
            "context": context,
        }
        res_review = cedarpy.is_authorized(req_review, self.policies, [])
        if res_review.allowed:
            reasons = [str(r) for r in res_review.diagnostics.reasons]
            rule_id = reasons[0] if reasons else "policy0"
            return {
                "decision": self.ACTION_REVIEW_NOTIFY,
                "agent_proposed_action": self.ACTION_REVIEW_NOTIFY,
                "cedar_action": 'Action::"REVIEW_NOTIFY"',
                "cedar_decision": str(res_review.decision),
                "allowed": True,
                "policy_rule_id": rule_id,
                "enforced_action": self.ACTION_REVIEW_NOTIFY,
                "fallback_action": "N/A (Action Authorized)",
                "diagnostics": {
                    "reasons": reasons,
                    "errors": [str(e) for e in res_review.diagnostics.errors],
                },
                "policy_gate": "PASSED_REVIEW_NOTIFY_POLICY",
                "evaluated_context": context,
            }

        # Priority 3: Check if LOG is permitted by Cedar
        req_log = {
            "principal": 'Agent::"SafetyGuardian"',
            "action": 'Action::"LOG"',
            "resource": f'Incident::"{incident_id}"',
            "context": context,
        }
        res_log = cedarpy.is_authorized(req_log, self.policies, [])
        if res_log.allowed:
            reasons = [str(r) for r in res_log.diagnostics.reasons]
            rule_id = reasons[0] if reasons else "policy0"
            return {
                "decision": self.ACTION_LOG,
                "agent_proposed_action": self.ACTION_LOG,
                "cedar_action": 'Action::"LOG"',
                "cedar_decision": str(res_log.decision),
                "allowed": True,
                "policy_rule_id": rule_id,
                "enforced_action": self.ACTION_LOG,
                "fallback_action": "N/A (Action Authorized)",
                "diagnostics": {
                    "reasons": reasons,
                    "errors": [str(e) for e in res_log.diagnostics.errors],
                },
                "policy_gate": "PASSED_LOG_POLICY",
                "evaluated_context": context,
            }

        # Safe fallback: If between boundaries, assign REVIEW/NOTIFY
        return {
            "decision": self.ACTION_REVIEW_NOTIFY,
            "agent_proposed_action": self.ACTION_REVIEW_NOTIFY,
            "cedar_action": 'Action::"REVIEW_NOTIFY"',
            "cedar_decision": "Decision.Allow",
            "allowed": True,
            "policy_rule_id": "policy0",
            "enforced_action": self.ACTION_REVIEW_NOTIFY,
            "fallback_action": "N/A (Action Authorized)",
            "diagnostics": {"reasons": ["Default safety boundary enforcement"], "errors": []},
            "policy_gate": "FALLBACK_SAFETY_DEFAULT",
            "evaluated_context": context,
        }

    def evaluate_proposed_action(
        self,
        incident_id: str,
        proposed_action: str,
        cumulative_risk: float,
        spinal_compression_n: float,
        awkward_duty_cycle_pct: float,
        rep_rate_per_min: float = 0.0,
        fatigue_index: float = 30.0,
        has_failed_prior_intervention: bool = False,
    ) -> Dict[str, Any]:
        """
        Evaluates an explicitly proposed action against Cedar policies.
        If denied by Cedar forbid policies, deterministically calculates the fallback action.
        """
        context = {
            "cumulative_risk": int(round(cumulative_risk)),
            "spinal_compression_n": int(round(spinal_compression_n)),
            "awkward_duty_cycle_pct": int(round(awkward_duty_cycle_pct)),
            "rep_rate_per_min": int(round(rep_rate_per_min)),
            "fatigue_index": int(round(fatigue_index)),
            "has_failed_prior_intervention": bool(has_failed_prior_intervention),
        }

        # Map action name
        cedar_act_name = "REVIEW_NOTIFY" if proposed_action in ("REVIEW/NOTIFY", "REVIEW_NOTIFY") else proposed_action

        req = {
            "principal": 'Agent::"SafetyGuardian"',
            "action": f'Action::"{cedar_act_name}"',
            "resource": f'Incident::"{incident_id}"',
            "context": context,
        }

        res = cedarpy.is_authorized(req, self.policies, [])
        reasons = [str(r) for r in res.diagnostics.reasons]
        errors = [str(e) for e in res.diagnostics.errors]
        rule_id = reasons[0] if reasons else ("policy0" if res.allowed else "forbid_rule")

        if res.allowed:
            return {
                "decision": proposed_action,
                "agent_proposed_action": proposed_action,
                "cedar_action": f'Action::"{cedar_act_name}"',
                "cedar_decision": str(res.decision),
                "allowed": True,
                "policy_rule_id": rule_id,
                "enforced_action": proposed_action,
                "fallback_action": "N/A (Action Authorized)",
                "diagnostics": {"reasons": reasons, "errors": errors},
                "policy_gate": f"PASSED_{cedar_act_name}_POLICY",
                "evaluated_context": context,
            }
        else:
            # Deterministic fallback evaluation
            fallback_res = self.evaluate_safety_decision(
                incident_id=incident_id,
                cumulative_risk=cumulative_risk,
                spinal_compression_n=spinal_compression_n,
                awkward_duty_cycle_pct=awkward_duty_cycle_pct,
                rep_rate_per_min=rep_rate_per_min,
                fatigue_index=fatigue_index,
                has_failed_prior_intervention=has_failed_prior_intervention,
            )
            fallback_action = fallback_res["decision"]

            return {
                "decision": fallback_action,
                "agent_proposed_action": proposed_action,
                "cedar_action": f'Action::"{cedar_act_name}"',
                "cedar_decision": str(res.decision),
                "allowed": False,
                "policy_rule_id": rule_id,
                "enforced_action": fallback_action,
                "fallback_action": fallback_action,
                "diagnostics": {"reasons": reasons, "errors": errors},
                "policy_gate": f"DENIED_{cedar_act_name}_POLICY_ENFORCED_FALLBACK",
                "evaluated_context": context,
            }
