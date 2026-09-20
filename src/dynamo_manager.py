"""
KineticGuard - DynamoDB Storage & Ergonomic Passport Manager (Phase 5)
Manages storage and retrieval for:
1. kineticguard-incidents: Audit ledger of all biomechanical & VLM incidents
2. kineticguard-worker-passport: Cumulative worker ergonomic profile and fatigue index
3. kineticguard-station-stats: Station-level ergonomic safety metrics and active hazards
"""

import json
import os
import time
from datetime import datetime, timezone
from decimal import Decimal
from typing import Dict, List, Any, Optional

import boto3
from botocore.exceptions import ClientError


def float_to_decimal(obj):
    """Recursively converts float types to Decimal for DynamoDB serialization."""
    if isinstance(obj, float):
        return Decimal(str(round(obj, 4)))
    elif isinstance(obj, dict):
        return {k: float_to_decimal(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [float_to_decimal(v) for v in obj]
    return obj


def decimal_to_float(obj):
    """Recursively converts Decimal back to standard Python float/int."""
    if isinstance(obj, Decimal):
        return float(obj) if obj % 1 != 0 else int(obj)
    elif isinstance(obj, dict):
        return {k: decimal_to_float(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [decimal_to_float(v) for v in obj]
    return obj


class DynamoDBErgonomicsManager:
    """Interface for DynamoDB incident storage and Worker Ergonomic Passport computation."""

    TABLE_INCIDENTS = "kineticguard-incidents"
    TABLE_PASSPORT = "kineticguard-worker-passport"
    TABLE_STATIONS = "kineticguard-station-stats"

    def __init__(
        self,
        region_name: str = "us-east-1",
        mock_mode: bool = False,
        mock_state_dir: str = "output/mock_aws_state/dynamodb",
    ):
        self.region_name = region_name
        self.mock_mode = mock_mode
        self.mock_state_dir = mock_state_dir

        if self.mock_mode:
            os.makedirs(self.mock_state_dir, exist_ok=True)
            self.incidents_file = os.path.join(self.mock_state_dir, "incidents.json")
            self.passport_file = os.path.join(self.mock_state_dir, "passports.json")
            self.stations_file = os.path.join(self.mock_state_dir, "stations.json")
            self._init_mock_storage()
        else:
            self.dynamodb = boto3.resource("dynamodb", region_name=self.region_name)
            self.table_incidents = self.dynamodb.Table(self.TABLE_INCIDENTS)
            self.table_passport = self.dynamodb.Table(self.TABLE_PASSPORT)
            self.table_stations = self.dynamodb.Table(self.TABLE_STATIONS)

    def _init_mock_storage(self):
        for fpath in [self.incidents_file, self.passport_file, self.stations_file]:
            if not os.path.exists(fpath):
                with open(fpath, "w", encoding="utf-8") as f:
                    json.dump({}, f)

    def _read_mock(self, fpath: str) -> Dict[str, Any]:
        try:
            with open(fpath, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}

    def _write_mock(self, fpath: str, data: Dict[str, Any]):
        with open(fpath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    # -------------------------------------------------------------------------
    # Incident Storage
    # -------------------------------------------------------------------------
    def save_incident(self, incident: Dict[str, Any]) -> Dict[str, Any]:
        """Saves an incident record to DynamoDB and updates worker passport & station metrics."""
        incident_id = incident.get("incident_id")
        if not incident_id:
            incident_id = f"INC-{int(time.time())}"
            incident["incident_id"] = incident_id

        if "timestamp" not in incident:
            incident["timestamp"] = datetime.now(timezone.utc).isoformat()

        # Update Worker Passport
        passport = self.update_worker_passport(incident)
        # Update Station Statistics
        station = self.update_station_stats(incident)

        if self.mock_mode:
            incidents = self._read_mock(self.incidents_file)
            incidents[incident_id] = incident
            self._write_mock(self.incidents_file, incidents)
        else:
            item = float_to_decimal(incident)
            self.table_incidents.put_item(Item=item)

        return {
            "incident_id": incident_id,
            "status": "STORED",
            "passport_status": passport.get("current_ergonomic_status"),
            "station_risk": station.get("station_risk_level"),
        }

    # -------------------------------------------------------------------------
    # Worker Ergonomic Passport 2.0 Engine
    # -------------------------------------------------------------------------
    def update_worker_passport(self, incident: Dict[str, Any]) -> Dict[str, Any]:
        """Recalculates worker cumulative ergonomic health passport upon incident ingestion."""
        worker_id = incident.get("worker_id", "WORKER-DEFAULT")
        auth = incident.get("authoritative_metrics", {})
        score = float(auth.get("cumulative_risk_score", incident.get("cumulative_risk_score", 50.0)))
        risk_level = auth.get("risk_level", incident.get("risk_level", "MODERATE"))
        station_id = incident.get("station_id", "STATION-UNKNOWN")
        torque = float(auth.get("peak_lumbar_torque_nm", 75.0))
        comp = float(auth.get("peak_spinal_compression_n", 2000.0))
        reps = int(auth.get("repetition_count", 1))
        rep_rate = float(auth.get("repetition_rate_per_min", 10.0))
        duration = float(auth.get("duration_sec", 1.5))
        now_iso = datetime.now(timezone.utc).isoformat()

        current = self.get_worker_passport(worker_id)
        if not current:
            current = {
                "worker_id": worker_id,
                "assigned_station": station_id,
                "total_incidents": 0,
                "high_risk_incidents": 0,
                "mean_risk_score": 0.0,
                "fatigue_index": 0.0,
                "cumulative_exposure_hours": 0.5,
                "awkward_exposure_duration_sec": 0.0,
                "repetition_count": 0,
                "mean_rep_rate_per_min": rep_rate,
                "mean_lumbar_torque_nm": torque,
                "peak_lumbar_torque_nm": torque,
                "peak_compression_n": comp,
                "current_ergonomic_status": "HEALTHY",
                "recommended_rotations": [],
                "risk_trend": [],
                "created_at": now_iso,
                "last_seen_timestamp": now_iso,
            }

        # Update core tallies
        total = current.get("total_incidents", 0) + 1
        high_risk = current.get("high_risk_incidents", 0) + (1 if risk_level in ("HIGH", "DANGEROUS") else 0)
        old_mean = float(current.get("mean_risk_score", 0.0))
        new_mean = ((old_mean * (total - 1)) + score) / total

        # Biomechanical accumulations
        awk_duration = float(current.get("awkward_exposure_duration_sec", 0.0)) + duration
        total_reps = int(current.get("repetition_count", 0)) + reps
        peak_t = max(float(current.get("peak_lumbar_torque_nm", 0.0)), torque)
        peak_c = max(float(current.get("peak_compression_n", 0.0)), comp)
        mean_t = round(((float(current.get("mean_lumbar_torque_nm", torque)) * (total - 1)) + torque) / total, 1)

        # Risk status determination
        if high_risk >= 2 or new_mean >= 70.0 or peak_c >= 3400.0:
            status = "ACTION_REQUIRED"
            rotations = [
                "Mandatory immediate task rotation to non-manual sorting station.",
                "15-minute active postural recovery break recommended."
            ]
        elif high_risk == 1 or new_mean >= 50.0:
            status = "AT_RISK"
            rotations = ["Schedule station rotation within next 30 minutes."]
        else:
            status = "HEALTHY"
            rotations = ["Standard shift monitoring."]

        # Risk trend history (keep last 30 data points)
        trend = current.get("risk_trend", [])
        trend.append({
            "timestamp": now_iso,
            "score": round(score, 1),
            "risk_level": risk_level,
            "station_id": station_id,
        })
        if len(trend) > 30:
            trend = trend[-30:]

        current["total_incidents"] = total
        current["high_risk_incidents"] = high_risk
        current["mean_risk_score"] = round(new_mean, 1)
        current["fatigue_index"] = round(min(100.0, (new_mean * 0.65) + (total * 7.5) + (awk_duration * 1.5)), 1)
        current["awkward_exposure_duration_sec"] = round(awk_duration, 1)
        current["repetition_count"] = total_reps
        current["mean_rep_rate_per_min"] = round(rep_rate, 1)
        current["mean_lumbar_torque_nm"] = mean_t
        current["peak_lumbar_torque_nm"] = round(peak_t, 1)
        current["peak_compression_n"] = round(peak_c, 1)
        current["current_ergonomic_status"] = status
        current["assigned_station"] = station_id
        current["recommended_rotations"] = rotations
        current["risk_trend"] = trend
        current["last_seen_timestamp"] = now_iso
        current["last_updated"] = now_iso

        if self.mock_mode:
            passports = self._read_mock(self.passport_file)
            passports[worker_id] = current
            self._write_mock(self.passport_file, passports)
        else:
            item = float_to_decimal(current)
            self.table_passport.put_item(Item=item)

        return current

    def get_worker_passport(self, worker_id: str) -> Optional[Dict[str, Any]]:
        """Retrieves a Worker Ergonomic Passport by worker_id."""
        if self.mock_mode:
            passports = self._read_mock(self.passport_file)
            return passports.get(worker_id)
        else:
            try:
                res = self.table_passport.get_item(Key={"worker_id": worker_id})
                item = res.get("Item")
                return decimal_to_float(item) if item else None
            except ClientError:
                return None

    def get_worker_trend(self, worker_id: str) -> List[Dict[str, Any]]:
        """Retrieves historical risk trend for a specific worker."""
        passport = self.get_worker_passport(worker_id)
        if not passport:
            return []
        return passport.get("risk_trend", [])

    def list_all_passports(self) -> List[Dict[str, Any]]:
        """Returns passports for all monitored workers from live state."""
        if self.mock_mode:
            passports = self._read_mock(self.passport_file)
            return list(passports.values())
        else:
            try:
                res = self.table_passport.scan()
                return [decimal_to_float(item) for item in res.get("Items", [])]
            except ClientError:
                return []

    # -------------------------------------------------------------------------
    # Station Risk Intelligence Engine
    # -------------------------------------------------------------------------
    def update_station_stats(self, incident: Dict[str, Any]) -> Dict[str, Any]:
        """Updates aggregate station risk and safety metrics."""
        station_id = incident.get("station_id", "STATION-UNKNOWN")
        camera_id = incident.get("camera_id", "camera_01")
        worker_id = incident.get("worker_id", "WORKER-UNKNOWN")
        auth = incident.get("authoritative_metrics", {})
        vlm = incident.get("vlm_analysis", {})
        torque = float(auth.get("peak_lumbar_torque_nm", 75.0))
        comp = float(auth.get("peak_spinal_compression_n", 2000.0))
        duty = float(auth.get("awkward_duty_cycle_pct", 70.0))
        score = float(auth.get("cumulative_risk_score", 50.0))
        duration = float(auth.get("duration_sec", 1.5))
        now_iso = datetime.now(timezone.utc).isoformat()

        current = self.get_station_stats(station_id)
        if not current:
            current = {
                "station_id": station_id,
                "camera_id": camera_id,
                "workers_affected": [],
                "total_incidents": 0,
                "high_risk_incident_count": 0,
                "cumulative_awkward_exposure_sec": 0.0,
                "peak_lumbar_torque_nm": torque,
                "peak_spinal_compression_n": comp,
                "mean_duty_cycle_pct": duty,
                "station_risk_level": "MODERATE",
                "dominant_risk_factor": "Torso forward flexion with extended lever arm",
                "recommended_engineering_intervention": "Elevate workpiece presentation height to 85 cm",
                "risk_trend": [],
                "last_updated": now_iso,
            }

        # Track affected workers
        workers = set(current.get("workers_affected", []))
        workers.add(worker_id)
        current["workers_affected"] = list(workers)

        current["total_incidents"] = current.get("total_incidents", 0) + 1
        if score >= 60.0 or comp >= 2300.0:
            current["high_risk_incident_count"] = current.get("high_risk_incident_count", 0) + 1

        current["cumulative_awkward_exposure_sec"] = round(
            float(current.get("cumulative_awkward_exposure_sec", 0.0)) + duration, 1
        )
        current["peak_lumbar_torque_nm"] = round(max(float(current.get("peak_lumbar_torque_nm", 0.0)), torque), 1)
        current["peak_spinal_compression_n"] = round(max(float(current.get("peak_spinal_compression_n", 0.0)), comp), 1)
        current["mean_duty_cycle_pct"] = round((float(current.get("mean_duty_cycle_pct", duty)) + duty) / 2.0, 1)

        # Dominant risk factor and engineering recommendation
        if current["peak_spinal_compression_n"] >= 3400.0:
            current["station_risk_level"] = "ACTION_REQUIRED"
            current["dominant_risk_factor"] = "Critical L5/S1 spinal compressive overload breaching NIOSH Action Limit"
            current["recommended_engineering_intervention"] = "Mandatory zero-gravity mechanical crane assist and 2-person lift rule"
        elif current["peak_spinal_compression_n"] >= 2200.0 or current["mean_duty_cycle_pct"] >= 80.0:
            current["station_risk_level"] = "HIGH"
            factor = vlm.get("root_cause")
            if not factor and vlm.get("risk_factors"):
                factor = vlm["risk_factors"][0]
            if not factor and incident.get("contributing_factors"):
                factor = incident["contributing_factors"][0]
            current["dominant_risk_factor"] = factor or "Awkward torso flexion and spinal compressive loading"
            current["recommended_engineering_intervention"] = vlm.get("recommended_action") or incident.get("recommended_action") or "Install adjustable hydraulic scissor lift table (40 cm elevation)"
        else:
            current["station_risk_level"] = "MODERATE"
            current["dominant_risk_factor"] = "Routine picking repetition"
            current["recommended_engineering_intervention"] = "Anti-fatigue matting and periodic job rotation"

        # Risk trend
        trend = current.get("risk_trend", [])
        trend.append({"timestamp": now_iso, "score": round(score, 1)})
        if len(trend) > 30:
            trend = trend[-30:]
        current["risk_trend"] = trend
        current["last_updated"] = now_iso

        if self.mock_mode:
            stations = self._read_mock(self.stations_file)
            stations[station_id] = current
            self._write_mock(self.stations_file, stations)
        else:
            item = float_to_decimal(current)
            self.table_stations.put_item(Item=item)

        return current

    def get_station_stats(self, station_id: str) -> Optional[Dict[str, Any]]:
        """Retrieves stats for a specific station."""
        if self.mock_mode:
            stations = self._read_mock(self.stations_file)
            return stations.get(station_id)
        else:
            try:
                res = self.table_stations.get_item(Key={"station_id": station_id})
                item = res.get("Item")
                return decimal_to_float(item) if item else None
            except ClientError:
                return None

    def get_station_trend(self, station_id: str) -> List[Dict[str, Any]]:
        """Retrieves historical risk trend for a specific station."""
        st = self.get_station_stats(station_id)
        if not st:
            return []
        return st.get("risk_trend", [])


    def list_all_stations(self) -> List[Dict[str, Any]]:
        """Returns statistics for all monitored stations."""
        if self.mock_mode:
            stations = self._read_mock(self.stations_file)
            return list(stations.values())
        else:
            try:
                res = self.table_stations.scan()
                return [decimal_to_float(item) for item in res.get("Items", [])]
            except ClientError:
                return []

    # -------------------------------------------------------------------------
    # Incident Queries
    # -------------------------------------------------------------------------
    def get_latest_incidents(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Returns the most recent incidents."""
        if self.mock_mode:
            incidents = self._read_mock(self.incidents_file)
            items = list(incidents.values())
            # Sort by timestamp desc
            items.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
            return items[:limit]
        else:
            try:
                res = self.table_incidents.scan(Limit=limit)
                items = [decimal_to_float(i) for i in res.get("Items", [])]
                items.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
                return items
            except ClientError:
                return []

    def get_incident_history(
        self,
        worker_id: Optional[str] = None,
        station_id: Optional[str] = None,
        limit: int = 20,
    ) -> List[Dict[str, Any]]:
        """Filters incident history by worker or station."""
        if self.mock_mode:
            incidents = self._read_mock(self.incidents_file)
            items = list(incidents.values())
            if worker_id:
                items = [i for i in items if i.get("worker_id") == worker_id]
            if station_id:
                items = [i for i in items if i.get("station_id") == station_id]
            items.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
            return items[:limit]
        else:
            # Live scan / filter
            try:
                res = self.table_incidents.scan()
                items = [decimal_to_float(i) for i in res.get("Items", [])]
                if worker_id:
                    items = [i for i in items if i.get("worker_id") == worker_id]
                if station_id:
                    items = [i for i in items if i.get("station_id") == station_id]
                items.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
                return items[:limit]
            except ClientError:
                return []
