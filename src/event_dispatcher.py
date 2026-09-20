"""
KineticGuard - Local Safety Event & Action Dispatch Layer
Executes local event routing and automated safety interventions based on Strands + Cedar decisions.

Dispatches:
  - LOG -> Indexes incident and decision into OpenSearch; no operational interruption.
  - REVIEW/NOTIFY -> Generates SUPERVISOR_NOTIFICATION_EVENT, logs supervisor alert, and creates intervention record.
  - ESCALATE -> Generates CRITICAL_ESCALATION_EVENT, issues immediate safety halt/rotation alert, and creates intervention record.

Zero AWS cloud dependencies (SNS, EventBridge, Lambda replaced with local typed event dispatcher and OpenSearch).
"""

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from rich.console import Console
from .local_opensearch import (
    KineticGuardOpenSearch,
    INDEX_INCIDENTS,
    INDEX_AGENT_DECISIONS,
    INDEX_INTERVENTIONS,
    INDEX_WORKER_EXPOSURE,
    INDEX_STATION_RISK,
    opensearch_engine,
)

console = Console()


class SafetyEventDispatcher:
    """Local event broker and intervention manager for KineticGuard."""

    def __init__(self, opensearch: Optional[KineticGuardOpenSearch] = None, output_dir: Optional[Path] = None):
        self.opensearch = opensearch or opensearch_engine
        self.output_dir = Path(output_dir or "output")
        self.events_dir = self.output_dir / "events"
        self.events_dir.mkdir(parents=True, exist_ok=True)
        self.events_log_file = self.events_dir / "dispatched_events.jsonl"

    def dispatch_decision(self, agent_decision: Dict[str, Any]) -> Dict[str, Any]:
        """
        Dispatches action based on Cedar policy decision and persists to OpenSearch.
        """
        inc_id = agent_decision.get("incident_id", f"INC-{int(time.time())}")
        worker_id = agent_decision.get("worker_identity", {}).get("worker_id", "WORKER-UNKNOWN")
        station_id = agent_decision.get("worker_identity", {}).get("station_id", "STATION-UNKNOWN")
        camera_id = agent_decision.get("worker_identity", {}).get("camera_id", "camera_01")
        decision = agent_decision.get("decision", "LOG")
        rec_action = agent_decision.get("recommended_action", "Continue routine monitoring")
        bio = agent_decision.get("authoritative_biomechanics", {})

        # 1. Store Decision in OpenSearch
        self.opensearch.index_agent_decision(agent_decision)

        # 2. Store or Update Worker & Station in OpenSearch
        self.opensearch.index_worker_exposure(worker_id, {
            "worker_id": worker_id,
            "station_id": station_id,
            "last_decision": decision,
            "last_risk_score": bio.get("cumulative_risk_score", 50.0),
            "peak_compression_n": bio.get("peak_spinal_compression_n", 2000.0),
            "updated_at": datetime.now(timezone.utc).isoformat(),
        })

        self.opensearch.index_station_risk(station_id, {
            "station_id": station_id,
            "camera_id": camera_id,
            "last_decision": decision,
            "last_risk_score": bio.get("cumulative_risk_score", 50.0),
            "updated_at": datetime.now(timezone.utc).isoformat(),
        })

        # 3. Handle Decision Routing
        event_record = None
        intervention_record = None

        if decision == "ESCALATE":
            event_record = self._dispatch_escalation(inc_id, worker_id, station_id, camera_id, bio, rec_action)
            intervention_record = self._create_intervention_record(
                inc_id, worker_id, station_id, camera_id, "MANDATORY_ROTATION_AND_REST", rec_action, bio
            )

        elif decision in ("REVIEW/NOTIFY", "REVIEW_NOTIFY"):
            event_record = self._dispatch_review_notify(inc_id, worker_id, station_id, camera_id, bio, rec_action)
            intervention_record = self._create_intervention_record(
                inc_id, worker_id, station_id, camera_id, "SUPERVISOR_REVIEW", rec_action, bio
            )

        else:  # LOG
            event_record = {
                "event_id": f"EVT-LOG-{int(time.time())}-{camera_id}",
                "event_type": "AUDIT_LOG_EVENT",
                "severity": "INFO",
                "incident_id": inc_id,
                "worker_id": worker_id,
                "station_id": station_id,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "details": "Routine ergonomic surveillance record indexed in OpenSearch; no operational intervention required.",
            }

        # 4. Append to local events log for auditability & debugging
        if event_record:
            with open(self.events_log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(event_record) + "\n")

        return {
            "incident_id": inc_id,
            "decision": decision,
            "event_dispatched": event_record,
            "intervention_created": intervention_record,
            "opensearch_indexed": True,
            "indexes_updated": [INDEX_AGENT_DECISIONS, INDEX_WORKER_EXPOSURE, INDEX_STATION_RISK]
            + ([INDEX_INTERVENTIONS] if intervention_record else []),
        }

    def _dispatch_escalation(
        self, inc_id: str, worker_id: str, station_id: str, camera_id: str, bio: Dict[str, Any], action: str
    ) -> Dict[str, Any]:
        """Dispatches high-priority safety escalation event."""
        comp = bio.get("peak_spinal_compression_n", 2200.0)
        score = bio.get("cumulative_risk_score", 70.0)
        evt_id = f"EVT-ESC-{int(time.time())}-{camera_id}"

        console.print(f"\n[bold red][ALERT: CRITICAL ESCALATION EVENT][/bold red] {evt_id}")
        console.print(f"  Worker: [cyan]{worker_id}[/cyan] at [cyan]{station_id}[/cyan] | Risk: [bold red]{score:.1f}/100[/bold red]")
        console.print(f"  Spinal Load: [bold red]{comp:.0f} N[/bold red] (NIOSH AL: 3400 N)")
        console.print(f"  Action: [yellow]{action}[/yellow]")

        return {
            "event_id": evt_id,
            "event_type": "CRITICAL_ESCALATION_EVENT",
            "severity": "CRITICAL",
            "channel": "LOCAL_SAFETY_BROKER",
            "incident_id": inc_id,
            "worker_id": worker_id,
            "station_id": station_id,
            "camera_id": camera_id,
            "authoritative_metrics": bio,
            "mandated_action": action,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "status": "DISPATCHED",
        }

    def _dispatch_review_notify(
        self, inc_id: str, worker_id: str, station_id: str, camera_id: str, bio: Dict[str, Any], action: str
    ) -> Dict[str, Any]:
        """Dispatches moderate supervisor notification event."""
        score = bio.get("cumulative_risk_score", 55.0)
        evt_id = f"EVT-NOTIFY-{int(time.time())}-{camera_id}"

        console.print(f"\n[bold yellow][NOTICE: SUPERVISOR NOTIFICATION EVENT][/bold yellow] {evt_id}")
        console.print(f"  Worker: [cyan]{worker_id}[/cyan] at [cyan]{station_id}[/cyan] | Risk: [bold yellow]{score:.1f}/100[/bold yellow]")
        console.print(f"  Action: [green]{action}[/green]")

        return {
            "event_id": evt_id,
            "event_type": "SUPERVISOR_NOTIFICATION_EVENT",
            "severity": "WARNING",
            "channel": "LOCAL_SAFETY_BROKER",
            "incident_id": inc_id,
            "worker_id": worker_id,
            "station_id": station_id,
            "camera_id": camera_id,
            "authoritative_metrics": bio,
            "recommended_action": action,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "status": "DISPATCHED",
        }

    def _create_intervention_record(
        self,
        inc_id: str,
        worker_id: str,
        station_id: str,
        camera_id: str,
        action_type: str,
        action_description: str,
        bio: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Creates and indexes an active intervention record in OpenSearch."""
        int_id = f"INTV-{inc_id}"
        record = {
            "intervention_id": int_id,
            "incident_id": inc_id,
            "worker_id": worker_id,
            "station_id": station_id,
            "camera_id": camera_id,
            "action_type": action_type,
            "action_description": action_description,
            "status": "APPLIED",
            "effectiveness_status": "WAITING_FOR_SUFFICIENT_DATA",
            "applied_at": datetime.now(timezone.utc).isoformat(),
            "pre_intervention_metrics": {
                "cumulative_risk_score": bio.get("cumulative_risk_score", 50.0),
                "peak_lumbar_torque_nm": bio.get("peak_lumbar_torque_nm", 75.0),
                "peak_spinal_compression_n": bio.get("peak_spinal_compression_n", 2200.0),
                "awkward_duty_cycle_pct": bio.get("awkward_duty_cycle_pct", 60.0),
            },
            "post_intervention_metrics": None,
            "is_effective": None,
            "risk_reduction_pct": None,
            "@timestamp": datetime.now(timezone.utc).isoformat(),
        }

        self.opensearch.index_intervention(record)
        return record
