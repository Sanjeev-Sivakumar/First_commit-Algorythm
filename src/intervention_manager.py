"""
KineticGuard - Closed-Loop Intervention System (Phase 6)
Manages the closed-loop safety lifecycle:
Risk Detected -> Recommendation -> Intervention Recorded -> Re-monitoring -> Before vs After -> Effectiveness

Tracks interventions (Task rotation, Recovery break, Pallet elevation, Height adjustment, Repetition reduction)
and evaluates whether risk was successfully reduced or requires escalation.
"""

import json
import os
import time
from datetime import datetime, timezone
from decimal import Decimal
from typing import Dict, List, Any, Optional

import boto3
from botocore.exceptions import ClientError


class InterventionManager:
    """Manages recording, re-monitoring, and effectiveness scoring of ergonomic interventions."""

    TABLE_NAME = "kineticguard-interventions"

    INTERVENTION_TYPES = [
        "TASK_ROTATION",
        "RECOVERY_BREAK",
        "PALLET_ELEVATION",
        "HEIGHT_ADJUSTMENT",
        "REPETITION_REDUCTION",
        "WORKFLOW_MODIFICATION",
    ]

    def __init__(
        self,
        region_name: str = "us-east-1",
        mock_mode: bool = False,
        mock_file: str = "output/mock_aws_state/dynamodb/interventions.json",
    ):
        self.region_name = region_name
        self.mock_mode = mock_mode
        self.mock_file = mock_file

        if self.mock_mode:
            os.makedirs(os.path.dirname(self.mock_file), exist_ok=True)
            if not os.path.exists(self.mock_file):
                with open(self.mock_file, "w", encoding="utf-8") as f:
                    json.dump({}, f)
        else:
            self.dynamodb = boto3.resource("dynamodb", region_name=self.region_name)
            self.table = self.dynamodb.Table(self.TABLE_NAME)

    def record_intervention(
        self,
        worker_id: str,
        station_id: str,
        triggering_incident_id: str,
        intervention_type: str,
        pre_intervention_risk: float,
        description: str,
        notes: str = "",
    ) -> Dict[str, Any]:
        """Creates and stores a new intervention record."""
        intervention_id = f"INT-{int(time.time()*1000)}"
        now_iso = datetime.now(timezone.utc).isoformat()

        record = {
            "intervention_id": intervention_id,
            "worker_id": worker_id,
            "station_id": station_id,
            "triggering_incident_id": triggering_incident_id,
            "intervention_type": intervention_type,
            "description": description,
            "timestamp": now_iso,
            "pre_intervention_risk": round(float(pre_intervention_risk), 1),
            "post_intervention_risk": None,
            "risk_change_pct": None,
            "effectiveness_status": "WAITING FOR SUFFICIENT POST-INTERVENTION DATA",
            "notes": notes,
            "last_updated": now_iso,
        }

        self._save_record(record)
        return record

    def evaluate_effectiveness(
        self,
        intervention_id: str,
        post_intervention_risk: float,
        evaluator_notes: str = "",
    ) -> Dict[str, Any]:
        """Evaluates before vs after risk to determine intervention effectiveness."""
        record = self.get_intervention(intervention_id)
        if not record:
            raise ValueError(f"Intervention not found: {intervention_id}")

        pre_risk = float(record.get("pre_intervention_risk", 50.0))
        post_risk = round(float(post_intervention_risk), 1)

        # Calculate percentage change: negative is improvement
        if pre_risk > 0:
            change_pct = round(((post_risk - pre_risk) / pre_risk) * 100.0, 1)
        else:
            change_pct = 0.0

        # Effectiveness classification
        # Significant reduction (>= 20% drop) -> EFFECTIVE
        # Moderate drop (5% to 20% drop) -> MODERATELY_EFFECTIVE
        # No improvement or worsening -> INEFFECTIVE (triggers escalation)
        if change_pct <= -20.0:
            status = "EFFECTIVE"
            summary_note = "Significant ergonomic strain reduction confirmed. Closed-loop control successful."
        elif change_pct <= -5.0:
            status = "MODERATELY_EFFECTIVE"
            summary_note = "Partial risk reduction observed. Continued monitoring recommended."
        else:
            status = "ESCALATED"
            summary_note = "Intervention failed to mitigate hazardous biomechanical loading. Escalated to EHS engineering."

        record["post_intervention_risk"] = post_risk
        record["risk_change_pct"] = change_pct
        record["effectiveness_status"] = status
        record["last_updated"] = datetime.now(timezone.utc).isoformat()
        if evaluator_notes:
            record["notes"] = f"{record.get('notes', '')} | {evaluator_notes}".strip(" |")
        else:
            record["notes"] = f"{record.get('notes', '')} | {summary_note}".strip(" |")

        self._save_record(record)
        return record

    def get_intervention(self, intervention_id: str) -> Optional[Dict[str, Any]]:
        """Retrieves an intervention record by ID."""
        if self.mock_mode:
            data = self._read_mock()
            return data.get(intervention_id)
        else:
            try:
                res = self.table.get_item(Key={"intervention_id": intervention_id})
                item = res.get("Item")
                return self._decimal_to_float(item) if item else None
            except ClientError:
                return None

    def get_by_worker(self, worker_id: str) -> List[Dict[str, Any]]:
        """Retrieves all interventions for a specific worker."""
        records = self.list_all()
        return [r for r in records if r.get("worker_id") == worker_id]

    def get_by_station(self, station_id: str) -> List[Dict[str, Any]]:
        """Retrieves all interventions for a specific station."""
        records = self.list_all()
        return [r for r in records if r.get("station_id") == station_id]

    def list_all(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Lists all intervention records sorted by timestamp desc."""
        if self.mock_mode:
            data = self._read_mock()
            records = list(data.values())
        else:
            try:
                res = self.table.scan()
                records = [self._decimal_to_float(i) for i in res.get("Items", [])]
            except ClientError:
                records = []

        records.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
        return records[:limit]

    # -------------------------------------------------------------------------
    # Internal helpers
    # -------------------------------------------------------------------------
    def _save_record(self, record: Dict[str, Any]):
        if self.mock_mode:
            data = self._read_mock()
            data[record["intervention_id"]] = record
            with open(self.mock_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
        else:
            item = self._float_to_decimal(record)
            self.table.put_item(Item=item)

    def _read_mock(self) -> Dict[str, Any]:
        try:
            with open(self.mock_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}

    def _float_to_decimal(self, obj):
        if isinstance(obj, float):
            return Decimal(str(round(obj, 4)))
        elif isinstance(obj, dict):
            return {k: self._float_to_decimal(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._float_to_decimal(v) for v in obj]
        return obj

    def _decimal_to_float(self, obj):
        if isinstance(obj, Decimal):
            return float(obj) if obj % 1 != 0 else int(obj)
        elif isinstance(obj, dict):
            return {k: self._decimal_to_float(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._decimal_to_float(v) for v in obj]
        return obj
