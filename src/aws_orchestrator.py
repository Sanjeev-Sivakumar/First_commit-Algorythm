"""
KineticGuard - AWS Lambda Orchestrator & REST API Handler (Phase 6)
Autonomous Safety Intelligence Engine running inside AWS Lambda or local mock environment.

Pipeline:
  Incident -> Lambda -> DynamoDB -> AI analysis (SageMaker/LoRA) -> Safety Guardian Agent -> Risk/Action -> SNS

REST APIs:
  - GET  /passport/{worker_id}               -> Worker Ergonomic Passport 2.0
  - GET  /passport/{worker_id}/trend         -> Worker Risk Trend
  - GET  /passport/{worker_id}/interventions -> Worker Interventions History
  - GET  /stations                           -> Multi-Station Ergonomic Risk Dashboard
  - GET  /stations/{station_id}              -> Station Safety Stats
  - GET  /stations/{station_id}/trend        -> Station Historical Risk Trend
  - POST /incidents                          -> Ingest Edge Incident + Safety Guardian Decision
  - GET  /incidents                          -> Incident History Ledger (filterable)
  - GET  /incidents/latest                   -> Latest Active Incidents
  - POST /interventions                      -> Record New Ergonomic Intervention
  - POST /interventions/{id}/evaluate        -> Evaluate Intervention Effectiveness (Before vs After)
  - GET  /interventions/{id}                 -> Get Specific Intervention Record
  - GET  /interventions/worker/{worker_id}   -> Worker Interventions
  - GET  /interventions/station/{station_id} -> Station Interventions
  - GET  /interventions                      -> List All Interventions
  - POST /forecast                           -> Near-Term Exposure Forecasting
  - POST /simulator/whatif                   -> What-If Ergonomic Simulation
  - GET  /reports/shift                      -> Executive Shift Safety Report
"""

import base64
import json
import os
import queue
import time
from datetime import datetime, timezone
from typing import Dict, Any, Optional, List

import boto3
from botocore.exceptions import ClientError

from src.dynamo_manager import DynamoDBErgonomicsManager
from src.sagemaker_service import SageMakerErgonomicService
from src.intervention_manager import InterventionManager
from src.safety_guardian import SafetyGuardianAgent
from src.exposure_forecaster import ExposureForecaster
from src.whatif_simulator import WhatIfSimulator
from src.shift_reporter import ShiftReporter


class KineticGuardOrchestrator:
    """Central orchestration controller executing the end-to-end KineticGuard pipeline."""

    def __init__(
        self,
        region_name: str = "us-east-1",
        mock_mode: bool = False,
        s3_bucket: Optional[str] = None,
        sns_topic_arn: Optional[str] = None,
    ):
        self.region_name = os.environ.get("AWS_REGION", region_name)
        self.mock_mode = mock_mode or os.environ.get("MOCK_AWS", "false").lower() == "true"
        self.s3_bucket = s3_bucket or os.environ.get("S3_BUCKET_NAME", "kineticguard-artifacts")
        self.sns_topic_arn = sns_topic_arn or os.environ.get("SNS_TOPIC_ARN")

        # Event pub/sub for real-time dashboard updates
        self._listeners: List[queue.Queue] = []
        self.latest_events: List[Dict[str, Any]] = []

        # Phase 5 & 6 Modules
        self.dynamo = DynamoDBErgonomicsManager(
            region_name=self.region_name,
            mock_mode=self.mock_mode,
        )
        self.sagemaker = SageMakerErgonomicService(
            region_name=self.region_name,
            mock_mode=self.mock_mode,
        )
        self.interventions = InterventionManager(
            region_name=self.region_name,
            mock_mode=self.mock_mode,
        )
        self.guardian = SafetyGuardianAgent()
        self.forecaster = ExposureForecaster()
        self.simulator = WhatIfSimulator()
        self.reporter = ShiftReporter(
            region_name=self.region_name,
            mock_mode=self.mock_mode,
        )

        if not self.mock_mode:
            self.s3 = boto3.client("s3", region_name=self.region_name)
            self.sns = boto3.client("sns", region_name=self.region_name)
        else:
            self.mock_sns_alerts_file = "output/mock_aws_state/sns_alerts.json"
            os.makedirs(os.path.dirname(self.mock_sns_alerts_file), exist_ok=True)

    def register_listener(self) -> queue.Queue:
        q = queue.Queue(maxsize=100)
        self._listeners.append(q)
        return q

    def unregister_listener(self, q: queue.Queue):
        if q in self._listeners:
            self._listeners.remove(q)

    def broadcast_event(self, event_data: Dict[str, Any]):
        event_data["broadcast_timestamp"] = datetime.now(timezone.utc).isoformat()
        self.latest_events.append(event_data)
        if len(self.latest_events) > 100:
            self.latest_events.pop(0)

        dead = []
        for q in self._listeners:
            try:
                q.put_nowait(event_data)
            except Exception:
                dead.append(q)
        for d in dead:
            self.unregister_listener(d)

    def route_request(self, method: str, path: str, query_params: Dict[str, str], body: Any) -> Dict[str, Any]:
        """Routes HTTP requests to appropriate handlers."""
        method = method.upper()

        if path == "/health" and method == "GET":
            return self._response(200, {
                "status": "HEALTHY",
                "system": "KineticGuard Autonomous Safety Intelligence Platform",
                "phase": "Phase 6",
                "mode": "MOCK" if self.mock_mode else "AWS_LIVE",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            })

        # --- LIVE OVERVIEW & METRICS API ---
        if path in ("/overview", "/api/overview") and method == "GET":
            return self.handle_get_overview()

        # --- INCIDENTS API ---
        if (path in ("/incidents", "/api/incidents")) and method == "POST":
            return self.handle_post_incident(body)

        if (path in ("/incidents/latest", "/api/incidents/latest")) and method == "GET":
            limit = int(query_params.get("limit", 10))
            return self._response(200, {"incidents": self.dynamo.get_latest_incidents(limit)})

        if (path in ("/incidents", "/api/incidents")) and method == "GET":
            worker_id = query_params.get("worker_id")
            station_id = query_params.get("station_id")
            limit = int(query_params.get("limit", 20))
            history = self.dynamo.get_incident_history(worker_id=worker_id, station_id=station_id, limit=limit)
            return self._response(200, {
                "count": len(history),
                "worker_id_filter": worker_id,
                "station_id_filter": station_id,
                "incidents": history,
            })

        # --- WORKER PASSPORT 2.0 API ---
        pure_path = path.split("?")[0]
        clean_parts = [p for p in pure_path.strip("/").split("/") if p and p != "api"]

        # --- WORKER PASSPORT 2.0 API ---
        if clean_parts and clean_parts[0] in ("passports", "passport"):
            if len(clean_parts) == 1 and method == "GET":
                return self._response(200, {"passports": self.dynamo.list_all_passports()})
            elif len(clean_parts) >= 2:
                worker_id = clean_parts[1]
                if len(clean_parts) >= 3 and clean_parts[2] == "trend":
                    return self._response(200, {"worker_id": worker_id, "trend": self.dynamo.get_worker_trend(worker_id)})
                elif len(clean_parts) >= 3 and clean_parts[2] == "interventions":
                    return self._response(200, {"worker_id": worker_id, "interventions": self.interventions.get_by_worker(worker_id)})
                else:
                    passport = self.dynamo.get_worker_passport(worker_id)
                    if not passport:
                        return self._response(404, {"error": f"Worker passport not found for worker_id: {worker_id}"})
                    return self._response(200, {"passport": passport})

        if path in ("/workers", "/api/workers") and method == "GET":
            passports = self.dynamo.list_all_passports()
            return self._response(200, {"workers": [p.get("worker_id") for p in passports if p.get("worker_id")]})

        # --- STATION RISK INTELLIGENCE API ---
        if clean_parts and clean_parts[0] in ("stations", "station"):
            if len(clean_parts) == 1 and method == "GET":
                return self._response(200, {"stations": self.dynamo.list_all_stations()})
            elif len(clean_parts) >= 2:
                station_id = clean_parts[1]
                if len(clean_parts) >= 3 and clean_parts[2] == "trend":
                    return self._response(200, {"station_id": station_id, "trend": self.dynamo.get_station_trend(station_id)})
                else:
                    stats = self.dynamo.get_station_stats(station_id)
                    if not stats:
                        return self._response(404, {"error": f"Station stats not found for station_id: {station_id}"})
                    return self._response(200, {"station": stats})

        # --- CLOSED-LOOP INTERVENTIONS API ---
        if path in ("/interventions", "/api/interventions") and method == "POST":
            return self.handle_post_intervention(body)

        if path.startswith("/interventions/") or path.startswith("/api/interventions/"):
            parts = path.rstrip("/").split("/")
            target = parts[-1]
            if len(parts) >= 4 and parts[-1] == "evaluate":
                int_id = parts[-2]
                return self.handle_evaluate_intervention(int_id, body)
            if parts[-2] == "worker":
                return self._response(200, {"interventions": self.interventions.get_by_worker(target)})
            if parts[-2] == "station":
                return self._response(200, {"interventions": self.interventions.get_by_station(target)})
            
            # Single intervention GET
            record = self.interventions.get_intervention(target)
            if not record:
                return self._response(404, {"error": f"Intervention not found: {target}"})
            return self._response(200, {"intervention": record})

        if path in ("/interventions", "/api/interventions") and method == "GET":
            limit = int(query_params.get("limit", 50))
            return self._response(200, {"interventions": self.interventions.list_all(limit=limit)})

        # --- NEAR-TERM EXPOSURE FORECASTING API ---
        if clean_parts and clean_parts[0] == "forecast":
            q = dict(query_params or {})
            if len(clean_parts) >= 2 and "worker_id" not in q:
                q["worker_id"] = clean_parts[1]
            return self.handle_forecast(body or q, q)

        # --- WHAT-IF SIMULATOR API ---
        if clean_parts and (clean_parts[0] in ("simulator", "simulate") or (len(clean_parts) > 1 and clean_parts[1] == "whatif")):
            sim_input = body if (body and isinstance(body, dict)) else dict(query_params or {})
            return self.handle_simulator_whatif(sim_input)

        # --- SHIFT SAFETY REPORT API ---
        if path in ("/reports/shift", "/api/reports/shift") and method in ("GET", "POST"):
            report_res = self.reporter.generate_report()
            return self._response(200, report_res["report"])

        return self._response(404, {"error": f"Route not found: {method} {path}"})

    def handle_get_overview(self) -> Dict[str, Any]:
        """Returns dynamically aggregated platform overview metrics."""
        stations = self.dynamo.list_all_stations()
        passports = self.dynamo.list_all_passports()
        incidents = self.dynamo.get_latest_incidents(limit=50)

        high_risk_workers = [p for p in passports if p.get("current_ergonomic_status") in ("ACTION_REQUIRED", "AT_RISK")]
        high_risk_stations = [s for s in stations if s.get("station_risk_level") in ("HIGH", "DANGEROUS", "ACTION_REQUIRED")]

        latest_action = "MONITORING"
        latest_timestamp = None
        freshness = "NO DATA"

        if incidents:
            latest_inc = incidents[0]
            latest_timestamp = latest_inc.get("timestamp")
            guardian = latest_inc.get("safety_guardian", {})
            latest_action = guardian.get("decision") or latest_inc.get("risk_level", "MONITORING")

            try:
                t = datetime.fromisoformat(latest_timestamp.replace("Z", "+00:00"))
                diff_sec = (datetime.now(timezone.utc) - t).total_seconds()
                freshness = "LIVE" if diff_sec <= 60 else "STALE"
            except Exception:
                freshness = "LIVE"

        return self._response(200, {
            "stations_count": len(stations),
            "workers_count": len(passports),
            "incidents_count": len(incidents),
            "high_risk_workers_count": len(high_risk_workers),
            "high_risk_stations_count": len(high_risk_stations),
            "latest_guardian_action": latest_action,
            "latest_event_timestamp": latest_timestamp,
            "freshness": freshness,
            "stations": stations,
            "passports": passports,
        })

    # -------------------------------------------------------------------------
    # Core Ingestion Loop with Safety Guardian Decision
    # -------------------------------------------------------------------------
    def handle_post_incident(self, body: Any) -> Dict[str, Any]:
        """Ingests edge incident, runs VLM, invokes Safety Guardian, updates DynamoDB, alerts via SNS."""
        if isinstance(body, str):
            try:
                payload = json.loads(body)
            except Exception:
                return self._response(400, {"error": "Invalid JSON body"})
        elif isinstance(body, dict):
            payload = body
        else:
            return self._response(400, {"error": "Missing incident payload"})

        incident_id = payload.get("incident_id") or f"INC-{int(time.time()*1000)}"
        payload["incident_id"] = incident_id
        camera_id = payload.get("camera_id", "camera_01")
        clean_cam = camera_id.replace("camera_", "").upper()
        worker_id = payload.get("worker_id") or f"W-CAM{clean_cam}-{incident_id[-4:]}"
        station_id = payload.get("station_id") or f"STATION-{clean_cam}"
        payload["worker_id"] = worker_id
        payload["station_id"] = station_id

        # 1. Upload Keyframe to S3 if base64 provided
        s3_keyframe_uri = None
        keyframe_b64 = payload.pop("keyframe_base64", None)
        if keyframe_b64:
            s3_keyframe_uri = self._upload_keyframe_to_s3(incident_id, keyframe_b64)
            payload["s3_keyframe_uri"] = s3_keyframe_uri

        # 2. Execute / Verify AI Explanation via SageMaker / VLM
        if "vlm_analysis" not in payload:
            ai_result = self.sagemaker.invoke_inference(payload)
            payload["vlm_analysis"] = ai_result.get("vlm_analysis", {})
            payload["authoritative_metrics"] = ai_result.get("authoritative_metrics", payload.get("exposure_metrics", {}))

        # 3. Fetch current memory state for Worker & Station
        current_passport = self.dynamo.get_worker_passport(worker_id)
        current_station = self.dynamo.get_station_stats(station_id)
        recent_interventions = self.interventions.get_by_worker(worker_id)

        # 4. Invoke Safety Guardian Agent (Autonomous Decision Layer)
        guardian_decision = self.guardian.evaluate(
            incident=payload,
            worker_passport=current_passport,
            station_profile=current_station,
            recent_interventions=recent_interventions,
        )
        payload["safety_guardian"] = guardian_decision

        # 5. Store in DynamoDB (Incident, Passport 2.0, Station stats)
        storage_res = self.dynamo.save_incident(payload)

        # 6. Trigger SNS Alert if Safety Guardian decides ESCALATE or NOTIFY_SUPERVISOR
        decision_val = guardian_decision.get("decision")
        sns_sent = False
        if decision_val in ("ESCALATE", "RECOMMEND_INTERVENTION", "NOTIFY_SUPERVISOR"):
            sns_sent = self._publish_safety_alert(payload, guardian_decision)
            payload["sns_alert_sent"] = sns_sent

        # 7. Broadcast event to live dashboard listeners
        self.broadcast_event({"type": "INCIDENT", "incident": payload})

        # 8. Return complete confirmation
        return self._response(201, {
            "incident_id": incident_id,
            "status": "PROCESSED",
            "safety_guardian_decision": decision_val,
            "safety_guardian_reason": guardian_decision.get("reason"),
            "recommended_action": guardian_decision.get("recommended_action"),
            "worker_id": worker_id,
            "station_id": station_id,
            "passport_status": storage_res.get("passport_status"),
            "sns_alert_published": sns_sent,
            "s3_keyframe_uri": s3_keyframe_uri,
        })

    def handle_post_intervention(self, body: Any) -> Dict[str, Any]:
        """Records a new ergonomic intervention."""
        data = json.loads(body) if isinstance(body, str) else (body or {})
        worker_id = data.get("worker_id")
        station_id = data.get("station_id")
        if not worker_id or not station_id:
            return self._response(400, {"error": "worker_id and station_id are required"})

        try:
            record = self.interventions.record_intervention(
                worker_id=worker_id,
                station_id=station_id,
                triggering_incident_id=data.get("triggering_incident_id", "MANUAL-DISPATCH"),
                intervention_type=data.get("intervention_type", "TASK_ROTATION"),
                pre_intervention_risk=float(data.get("pre_intervention_risk", 70.0)),
                description=data.get("description", "Ergonomic control administered"),
                notes=data.get("notes", ""),
            )
            self.broadcast_event({"type": "INTERVENTION", "intervention": record})
            return self._response(201, {"intervention": record})
        except Exception as e:
            return self._response(400, {"error": str(e)})

    def handle_evaluate_intervention(self, int_id: str, body: Any) -> Dict[str, Any]:
        """Evaluates before vs after effectiveness of an intervention."""
        data = json.loads(body) if isinstance(body, str) else (body or {})
        post_risk = float(data.get("post_intervention_risk", 45.0))
        notes = data.get("notes", "")
        try:
            record = self.interventions.evaluate_effectiveness(
                intervention_id=int_id,
                post_intervention_risk=post_risk,
                evaluator_notes=notes,
            )
            self.broadcast_event({"type": "INTERVENTION_EVALUATED", "evaluated_intervention": record})
            return self._response(200, {"evaluated_intervention": record})
        except Exception as e:
            return self._response(400, {"error": str(e)})

    def handle_forecast(self, body: Any, query: Dict[str, str]) -> Dict[str, Any]:
        """Runs near-term exposure forecasting."""
        data = json.loads(body) if isinstance(body, str) else (body or {})
        worker_id = data.get("worker_id") or query.get("worker_id")
        if not worker_id:
            return self._response(400, {"error": "worker_id is required for exposure forecasting"})

        passport = self.dynamo.get_worker_passport(worker_id) or {}
        score = float(data.get("current_score", passport.get("mean_risk_score", 0.0)))
        duty = float(data.get("duty_cycle_pct", passport.get("awkward_exposure_duration_sec", 0.0)))
        reps = float(data.get("repetition_rate_per_min", passport.get("mean_rep_rate_per_min", 0.0)))
        fatigue = float(data.get("fatigue_index", passport.get("fatigue_index", 0.0)))
        trend = passport.get("risk_trend", [])
        station_id = passport.get("assigned_station", data.get("station_id", "STATION-UNKNOWN"))

        res = self.forecaster.forecast_exposure(
            current_score=score,
            duty_cycle_pct=duty,
            repetition_rate_per_min=reps,
            fatigue_index=fatigue,
            risk_trend=trend,
            worker_id=worker_id,
            station_id=station_id,
        )
        return self._response(200, res)

    def handle_simulator_whatif(self, body: Any) -> Dict[str, Any]:
        """Runs deterministic What-If simulator modeling."""
        data = json.loads(body) if isinstance(body, str) else (body or {})
        baseline = data.get("current_baseline")
        modifications = data.get("modifications") or data
        sim_res = self.simulator.simulate(current_baseline=baseline, modifications=modifications)
        return self._response(200, sim_res)

    def _upload_keyframe_to_s3(self, incident_id: str, b64_data: str) -> str:
        s3_key = f"incidents/keyframes/{incident_id}.jpg"
        if self.mock_mode:
            mock_s3_dir = "output/mock_aws_state/s3/incidents/keyframes"
            os.makedirs(mock_s3_dir, exist_ok=True)
            try:
                img_bytes = base64.b64decode(b64_data)
                with open(os.path.join(mock_s3_dir, f"{incident_id}.jpg"), "wb") as f:
                    f.write(img_bytes)
            except Exception:
                pass
            return f"s3://{self.s3_bucket}/{s3_key}"

        try:
            img_bytes = base64.b64decode(b64_data)
            self.s3.put_object(
                Bucket=self.s3_bucket,
                Key=s3_key,
                Body=img_bytes,
                ContentType="image/jpeg",
            )
            return f"s3://{self.s3_bucket}/{s3_key}"
        except Exception:
            return f"s3://{self.s3_bucket}/{s3_key}"

    def _publish_safety_alert(self, incident: Dict[str, Any], guardian: Dict[str, Any]) -> bool:
        incident_id = incident.get("incident_id")
        worker_id = incident.get("worker_id")
        station_id = incident.get("station_id")
        auth = incident.get("authoritative_metrics", {})
        decision = guardian.get("decision", "RECOMMEND_INTERVENTION")
        reason = guardian.get("reason", "")
        action = guardian.get("recommended_action", "")

        subject = f"[SAFETY GUARDIAN: {decision}] Ergonomic Action Alert at {station_id}"
        message = (
            f"KINETICGUARD AUTONOMOUS SAFETY INTELLIGENCE\n"
            f"--------------------------------------------------\n"
            f"Decision:            {decision}\n"
            f"Incident ID:         {incident_id}\n"
            f"Worker ID:           {worker_id}\n"
            f"Station ID:          {station_id}\n"
            f"Spinal Comp Load:    {auth.get('peak_spinal_compression_n', 'N/A')} N\n"
            f"Lumbar Torque:       {auth.get('peak_lumbar_torque_nm', 'N/A')} Nm\n"
            f"Awkward Duty Cycle:  {auth.get('awkward_duty_cycle_pct', 'N/A')}%\n\n"
            f"Guardian Reasoning:\n{reason}\n\n"
            f"Prescribed Corrective Action:\n{action}\n"
            f"--------------------------------------------------\n"
            f"Automated safety directive from KineticGuard Closed-Loop Cloud."
        )

        if self.mock_mode or not self.sns_topic_arn:
            alerts = []
            if os.path.exists(self.mock_sns_alerts_file):
                try:
                    with open(self.mock_sns_alerts_file, "r", encoding="utf-8") as f:
                        alerts = json.load(f)
                except Exception:
                    alerts = []
            alerts.append({
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "subject": subject,
                "incident_id": incident_id,
                "worker_id": worker_id,
                "station_id": station_id,
                "decision": decision,
                "message": message,
            })
            with open(self.mock_sns_alerts_file, "w", encoding="utf-8") as f:
                json.dump(alerts, f, indent=2)
            return True

        try:
            self.sns.publish(
                TopicArn=self.sns_topic_arn,
                Subject=subject[:100],
                Message=message,
            )
            return True
        except ClientError:
            return False

    def _response(self, status_code: int, body_dict: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "statusCode": status_code,
            "headers": {
                "Content-Type": "application/json",
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Methods": "GET,POST,OPTIONS",
                "Access-Control-Allow-Headers": "Content-Type,Authorization",
            },
            "body": json.dumps(body_dict),
        }


# Global orchestrator instance for AWS Lambda container reuse
_orchestrator = None


def lambda_handler(event: Dict[str, Any], context: Any) -> Dict[str, Any]:
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = KineticGuardOrchestrator()

    http_ctx = event.get("requestContext", {}).get("http", {})
    method = http_ctx.get("method") or event.get("httpMethod", "GET")
    raw_path = event.get("rawPath") or event.get("path", "/health")
    query_params = event.get("queryStringParameters") or {}

    body = event.get("body")
    if event.get("isBase64Encoded", False) and body:
        body = base64.b64decode(body).decode("utf-8")

    return _orchestrator.route_request(
        method=method,
        path=raw_path,
        query_params=query_params,
        body=body,
    )
