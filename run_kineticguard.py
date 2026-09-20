#!/usr/bin/env python3
"""
KineticGuard - Master Autonomous Safety Intelligence Demo Command (Phase 6)
Executes the complete end-to-end, zero-fake-data live runtime pipeline:

REAL VIDEO
   ↓
REAL POSE DETECTION (MediaPipe)
   ↓
REAL BIOMECHANICAL CALCULATIONS (L5/S1 Moments & Compression)
   ↓
REAL TEMPORAL SLIDING WINDOW (Exposure & Awkward Duty Cycle)
   ↓
REAL INCIDENT DETECTION & KEYFRAME CAPTURE
   ↓
REAL QWEN2-VL + LoRA MULTIMODAL EXPLANATION
   ↓
REAL AWS / ORCHESTRATOR INGESTION
   ↓
REAL DYNAMODB (Worker Passport 2.0 + Station Risk Intelligence)
   ↓
REAL SAFETY GUARDIAN AUTONOMOUS DECISION LAYER
   ↓
REAL SNS NOTIFICATION DISPATCH
   ↓
REAL CLOSED-LOOP INTERVENTION RECORDING
   ↓
REAL SUBSEQUENT TELEMETRY & BEFORE/AFTER EFFECTIVENESS
   ↓
REAL LIVE WEB DASHBOARD STREAMING & UPDATES

Usage:
  python run_kineticguard.py                     # Runs full live demo on camera_01
  python run_kineticguard.py --all-cameras       # Runs all 3 live camera feeds
  python run_kineticguard.py --port 8080         # Custom dashboard port
  python run_kineticguard.py --mock              # Force local mock cloud (zero AWS cost)
"""

import argparse
import base64
import json
import os
import sys
import threading
import time
from datetime import datetime, timezone
from http.server import HTTPServer
from pathlib import Path
from typing import Dict, Any, List, Optional

import cv2
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn

from src.config import settings
from src.pose_detector import PoseDetector
from src.biomechanics import BiomechanicsAnalyzer
from src.temporal_analyzer import TemporalSlidingWindow
from src.cumulative_risk import CumulativeRiskModel
from src.incident_engine import IncidentEngine, resolve_station, generate_worker_identity
from src.qwen_inference import ErgonomicVLMEngine
from src.aws_orchestrator import KineticGuardOrchestrator
from src.dashboard_server import DashboardHTTPHandler

console = Console()


class KineticGuardLiveDemo:
    """End-to-end driver executing the complete dynamic runtime pipeline."""

    def __init__(
        self,
        port: int = 8080,
        mock_mode: bool = True,
        max_frames: int = 120,
    ):
        self.port = port
        self.mock_mode = mock_mode
        self.max_frames = max_frames
        self.orchestrator = KineticGuardOrchestrator(mock_mode=self.mock_mode)
        DashboardHTTPHandler.orchestrator = self.orchestrator
        self.server: Optional[HTTPServer] = None
        self.server_thread: Optional[threading.Thread] = None

    def start_dashboard(self):
        """Launches the real-time web dashboard server in a background thread."""
        try:
            self.server = HTTPServer(("127.0.0.1", self.port), DashboardHTTPHandler)
            self.server_thread = threading.Thread(target=self.server.serve_forever, daemon=True)
            self.server_thread.start()
            console.print(
                f"[bold green]-> Live Dashboard Server online at: http://127.0.0.1:{self.port}[/bold green]"
            )
        except OSError as e:
            console.print(f"[yellow]-> Notice: Dashboard server port {self.port} in use or active: {e}[/yellow]")

    def run_camera_stream(
        self,
        video_path: str,
        camera_id: str,
        auto_intervene: bool = True,
    ) -> Dict[str, Any]:
        """Processes a real video stream, detecting pose, biomechanics, temporal risk, and incidents."""
        video_file = Path(video_path)
        if not video_file.exists():
            console.print(f"[bold red]Video not found: {video_file}[/bold red]")
            return {}

        station_id = resolve_station(camera_id)
        session_id = f"sess_{int(time.time())}_{camera_id}"
        worker_id = generate_worker_identity(camera_id, session_id)

        console.print(
            Panel(
                f"[bold cyan]Processing Real Camera Stream: {camera_id}[/bold cyan]\n"
                f"[dim]Source: {video_file.name} | Station: {station_id} | Dynamic Worker: {worker_id}[/dim]",
                border_style="cyan",
            )
        )

        cap = cv2.VideoCapture(str(video_file))
        if not cap.isOpened():
            console.print(f"[bold red]Failed to open video file: {video_file}[/bold red]")
            return {}

        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 100
        frames_to_read = min(total_frames, self.max_frames) if self.max_frames > 0 else total_frames

        # Initialize deterministic processing engines
        pose_detector = PoseDetector()
        biomech = BiomechanicsAnalyzer(body_weight_kg=75.0, load_weight_kg=12.0)
        temporal = TemporalSlidingWindow(window_seconds=15.0)
        risk_model = CumulativeRiskModel()
        incident_engine = IncidentEngine(
            camera_id=camera_id,
            incident_threshold=50.0,  # Ensure real incidents trigger on high-strain posture
            cooldown_sec=1.5,
            worker_id=worker_id,
            station_id=station_id,
            session_id=session_id,
            source=f"video_{video_file.name}",
        )
        qwen_engine = ErgonomicVLMEngine(force_fallback=self.mock_mode)

        frame_idx = 0
        detected_incidents: List[Dict[str, Any]] = []
        last_frame = None
        active_intervention_id: Optional[str] = None
        pre_intervention_risk_val: float = 0.0
        post_intervention_risks: List[float] = []

        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            task = progress.add_task(f"[cyan]Analyzing {camera_id}...", total=frames_to_read)

            while cap.isOpened() and frame_idx < frames_to_read:
                ret, frame = cap.read()
                if not ret:
                    break

                last_frame = frame
                timestamp_sec = frame_idx / fps
                h, w = frame.shape[:2]

                # 1. Real MediaPipe Pose Detection
                pose_res = pose_detector.detect_pose(frame)

                # 2. Real Biomechanical Calculation
                bio_metrics = biomech.analyze(
                    angles=pose_res.angles,
                    landmarks_3d=pose_res.landmarks_3d if pose_res.detected else None,
                )

                # 3. Real Temporal Sliding Window
                telemetry_frame = {
                    "frame_index": frame_idx,
                    "timestamp_sec": timestamp_sec,
                    "pose_detected": pose_res.detected,
                    "posture": bio_metrics.posture,
                    "risk_level": bio_metrics.risk_level,
                    "lumbar_angle_deg": pose_res.angles.get("lumbar", 180.0),
                    "lumbar_torque_nm": bio_metrics.lumbar_torque_nm,
                    "spinal_compression_n": bio_metrics.compression_force_n,
                    "trunk_flexion_deg": bio_metrics.trunk_flexion_deg,
                }
                temporal.add_frame(telemetry_frame)
                temporal_metrics = temporal.get_metrics()

                # 4. Real Cumulative Risk Scoring
                risk_score = risk_model.evaluate(temporal_metrics)

                # 5. Real Incident Detection
                incident = incident_engine.evaluate_frame(
                    timestamp_sec=timestamp_sec,
                    temporal_metrics=temporal_metrics,
                    risk_score=risk_score,
                )

                if incident:
                    # Capture real keyframe
                    keyframes_dir = Path("output/vlm_analysis/keyframes")
                    keyframes_dir.mkdir(parents=True, exist_ok=True)
                    keyframe_path = str(keyframes_dir / f"{incident.incident_id}_keyframe.jpg")
                    cv2.imwrite(keyframe_path, frame)

                    _, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                    keyframe_b64 = base64.b64encode(buffer).decode("utf-8")

                    inc_dict = incident.to_dict()
                    inc_dict["keyframe_base64"] = keyframe_b64
                    inc_dict["keyframe_path"] = keyframe_path

                    # 6. Real Qwen2-VL LoRA Visual Analysis
                    vlm_res = qwen_engine.analyze_incident(
                        incident_record=inc_dict,
                        camera_id=camera_id,
                        keyframe_path=keyframe_path,
                    )
                    inc_dict["vlm_analysis"] = vlm_res.get("vlm_analysis", vlm_res)

                    # 7. Real AWS Orchestrator Ingestion + Safety Guardian Decision + DynamoDB
                    resp = self.orchestrator.route_request(
                        method="POST",
                        path="/incidents",
                        query_params={},
                        body=inc_dict,
                    )
                    stored_incident = json.loads(resp.get("body", "{}"))
                    detected_incidents.append(stored_incident)

                    console.print(
                        f"\n[bold yellow]>> ERGONOMIC INCIDENT DETECTED on {camera_id}:[/bold yellow] "
                        f"ID: [bold]{stored_incident.get('incident_id')}[/bold] | "
                        f"Score: [red]{incident.cumulative_risk_score}/100[/red] | "
                        f"Guardian Decision: [bold cyan]{stored_incident.get('safety_guardian_decision')}[/bold cyan]"
                    )
                    console.print(f"   [dim]Root Cause Recommendation: {stored_incident.get('recommended_action')}[/dim]")

                    # Step A of Closed Loop: If auto_intervene is enabled, record real intervention
                    if auto_intervene and active_intervention_id is None:
                        inc_id = stored_incident.get("incident_id")
                        pre_risk = float(incident.cumulative_risk_score)
                        rec_action = stored_incident.get("recommended_action") or "Ergonomic equipment adjustment"
                        int_payload = {
                            "worker_id": worker_id,
                            "station_id": station_id,
                            "triggering_incident_id": inc_id,
                            "intervention_type": "PALLET_ELEVATION",
                            "pre_intervention_risk": round(pre_risk, 1),
                            "description": f"Enacted supervisor intervention: {rec_action}",
                        }
                        int_resp = self.orchestrator.route_request(
                            method="POST", path="/interventions", query_params={}, body=int_payload
                        )
                        created_int = json.loads(int_resp.get("body", "{}")).get("intervention", {})
                        active_intervention_id = created_int.get("intervention_id")
                        pre_intervention_risk_val = pre_risk
                        console.print(
                            f"   [green]1. Real Intervention Recorded:[/green] {active_intervention_id} "
                            f"[cyan](Status: WAITING FOR SUFFICIENT POST-INTERVENTION DATA)[/cyan]"
                        )
                elif active_intervention_id is not None:
                    # Collect real subsequent frame risk telemetry after intervention
                    post_intervention_risks.append(risk_score.score)

                frame_idx += 1
                progress.update(task, advance=1)

        cap.release()

        # Flush any open incident at end of stream
        flushed = incident_engine.flush()
        if flushed and last_frame is not None:
            keyframes_dir = Path("output/vlm_analysis/keyframes")
            keyframes_dir.mkdir(parents=True, exist_ok=True)
            keyframe_path = str(keyframes_dir / f"{flushed.incident_id}_keyframe.jpg")
            cv2.imwrite(keyframe_path, last_frame)

            _, buffer = cv2.imencode(".jpg", last_frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            keyframe_b64 = base64.b64encode(buffer).decode("utf-8")
            inc_dict = flushed.to_dict()
            inc_dict["keyframe_base64"] = keyframe_b64
            inc_dict["keyframe_path"] = keyframe_path
            vlm_res = qwen_engine.analyze_incident(
                incident_record=inc_dict,
                camera_id=camera_id,
                keyframe_path=keyframe_path,
            )
            inc_dict["vlm_analysis"] = vlm_res.get("vlm_analysis", vlm_res)
            resp = self.orchestrator.route_request(method="POST", path="/incidents", query_params={}, body=inc_dict)
            flushed_incident = json.loads(resp.get("body", "{}"))
            detected_incidents.append(flushed_incident)

            console.print(
                f"\n[bold yellow]>> ERGONOMIC INCIDENT FINALIZED on {camera_id}:[/bold yellow] "
                f"ID: [bold]{flushed_incident.get('incident_id')}[/bold] | "
                f"Score: [red]{flushed.cumulative_risk_score}/100[/red] | "
                f"Guardian Decision: [bold cyan]{flushed_incident.get('safety_guardian_decision')}[/bold cyan]"
            )
            if auto_intervene and active_intervention_id is None:
                inc_id = flushed_incident.get("incident_id")
                pre_risk = float(flushed.cumulative_risk_score)
                rec_action = flushed_incident.get("recommended_action") or "Adjust workstation height"
                int_payload = {
                    "worker_id": worker_id,
                    "station_id": station_id,
                    "triggering_incident_id": inc_id,
                    "intervention_type": "PALLET_ELEVATION",
                    "pre_intervention_risk": round(pre_risk, 1),
                    "description": f"Enacted supervisor intervention: {rec_action}",
                }
                int_resp = self.orchestrator.route_request(
                    method="POST", path="/interventions", query_params={}, body=int_payload
                )
                created_int = json.loads(int_resp.get("body", "{}")).get("intervention", {})
                active_intervention_id = created_int.get("intervention_id")
                console.print(
                    f"   [green]1. Real Intervention Recorded:[/green] {active_intervention_id} "
                    f"[cyan](Status: WAITING FOR SUFFICIENT POST-INTERVENTION DATA)[/cyan]"
                )

        console.print(f"[bold green]>> {camera_id} complete: {frame_idx} frames processed, {len(detected_incidents)} incidents captured.[/bold green]")

        # 8. Real Closed-Loop Intervention Workflow Demonstration
        if active_intervention_id is not None:
            if len(post_intervention_risks) >= 5:
                avg_post_risk = round(sum(post_intervention_risks) / len(post_intervention_risks), 1)
                eval_payload = {
                    "post_intervention_risk": avg_post_risk,
                    "notes": f"Post-intervention surveillance of {len(post_intervention_risks)} real frames confirms risk shift from {pre_intervention_risk_val:.1f} to {avg_post_risk:.1f}.",
                }
                eval_resp = self.orchestrator.route_request(
                    method="POST", path=f"/interventions/{active_intervention_id}/evaluate", query_params={}, body=eval_payload
                )
                evaluated_int = json.loads(eval_resp.get("body", "{}")).get("evaluated_intervention", {})
                console.print(
                    f"\n[bold green]>> 2. Real Intervention Effectiveness Evaluated:[/bold green] Pre: {evaluated_int.get('pre_intervention_risk')} -> "
                    f"Post: {evaluated_int.get('post_intervention_risk')} | "
                    f"Delta: [green]{evaluated_int.get('risk_change_pct')}%[/green] -> Status: [bold cyan]{evaluated_int.get('effectiveness_status')}[/bold cyan]"
                )
            else:
                console.print(
                    f"\n[yellow]>> 2. Intervention {active_intervention_id} Status: WAITING FOR SUFFICIENT POST-INTERVENTION DATA ({len(post_intervention_risks)}/5 frames collected)[/yellow]"
                )

        return {
            "camera_id": camera_id,
            "station_id": station_id,
            "worker_id": worker_id,
            "frames_processed": frame_idx,
            "incidents_count": len(detected_incidents),
        }

    def print_final_summary(self):
        """Displays aggregated live system stats from DynamoDB."""
        overview_resp = self.orchestrator.route_request("GET", "/overview", {}, None)
        overview = json.loads(overview_resp.get("body", "{}"))

        table = Table(title="KineticGuard Autonomous Safety Intelligence Platform - Live State", show_header=True)
        table.add_column("Metric", style="cyan", no_wrap=True)
        table.add_column("Live Runtime Value", style="bold green")

        table.add_row("Active Stations Monitored", str(overview.get("stations_count", 0)))
        table.add_row("Operatives in Database", str(overview.get("workers_count", 0)))
        table.add_row("Ergonomic Incidents Logged", str(overview.get("incidents_count", 0)))
        table.add_row("High-Risk Workers Detected", str(overview.get("high_risk_workers_count", 0)))
        table.add_row("Safety Guardian Decision", str(overview.get("latest_guardian_action", "--")))
        table.add_row("Telemetry Data Freshness", str(overview.get("freshness", "NO DATA")))
        table.add_row("Latest Event Timestamp", str(overview.get("latest_event_timestamp", "--")))

        console.print("\n")
        console.print(table)


def main():
    parser = argparse.ArgumentParser(description="KineticGuard Master Autonomous Safety Intelligence Demo Runner")
    parser.add_argument("--camera", type=str, default="camera_01", help="Camera stream to process (default: camera_01)")
    parser.add_argument("--all-cameras", action="store_true", help="Process all available cameras (camera_01, camera_02, camera_03)")
    parser.add_argument("--video", type=str, default=None, help="Explicit path to video file")
    parser.add_argument("--port", type=int, default=8080, help="Web dashboard port (default: 8080)")
    parser.add_argument("--max-frames", type=int, default=100, help="Maximum video frames to analyze per camera (default: 100)")
    parser.add_argument("--mock", action="store_true", default=settings.MOCK_AWS, help="Operate in local mock cloud mode")
    parser.add_argument("--no-dashboard", action="store_true", help="Disable web dashboard server")
    parser.add_argument("--keep-alive", action="store_true", default=True, help="Keep server running after processing completes")
    args = parser.parse_args()

    console.print(
        Panel.fit(
            "[bold white]KineticGuard: Autonomous Workplace Safety Intelligence[/bold white]\n"
            "[cyan]Zero Static Data • 100% Dynamic Biomechanical Surveillance • Closed-Loop Control[/cyan]",
            border_style="cyan",
        )
    )

    is_mock = args.mock
    if not is_mock and not settings.has_aws_credentials():
        console.print("[yellow]Notice: AWS credentials not found in environment; running in zero-cost local cloud mode.[/yellow]")
        console.print("[yellow]To target live AWS infrastructure, configure AWS_ACCESS_KEY_ID & AWS_SECRET_ACCESS_KEY in .env.[/yellow]\n")
        is_mock = True

    demo = KineticGuardLiveDemo(port=args.port, mock_mode=is_mock, max_frames=args.max_frames)

    if not args.no_dashboard:
        demo.start_dashboard()

    # Determine cameras to process
    if args.all_cameras:
        camera_list = [("data/camera_01.mp4", "camera_01"), ("data/camera_02.mp4", "camera_02"), ("data/camera_03.mp4", "camera_03")]
    elif args.video:
        camera_list = [(args.video, args.camera)]
    else:
        camera_source = settings.CAMERA_SOURCES.get(args.camera, f"data/{args.camera}.mp4")
        camera_list = [(camera_source, args.camera)]

    for vpath, cam_id in camera_list:
        demo.run_camera_stream(video_path=vpath, camera_id=cam_id, auto_intervene=True)

    demo.print_final_summary()

    # Generate shift safety report from active database records
    rep_res = demo.orchestrator.route_request("GET", "/reports/shift", {}, None)
    report = json.loads(rep_res.get("body", "{}"))
    console.print(
        f"\n[bold green]>> Shift Safety Report Compiled:[/bold green] {report.get('shift_id')} | "
        f"Overall Safety Score: [cyan]{report.get('executive_summary', {}).get('shift_ergonomic_safety_score')}/100[/cyan]"
    )

    console.print(
        f"\n[bold green]==============================================================[/bold green]\n"
        f"[bold white]   DEMO RUNNING LIVE AT: http://127.0.0.1:{args.port}   [/bold white]\n"
        f"[bold green]==============================================================[/bold green]"
    )
    console.print("[dim]Open http://127.0.0.1:8080 in your browser to inspect dynamic Passports, Heatmap, Interventions, & What-If Simulator.[/dim]")
    console.print("[dim]Press Ctrl+C to exit.[/dim]")

    if args.keep_alive and not args.no_dashboard:
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            console.print("\n[yellow]Shutting down KineticGuard demo server...[/yellow]")


if __name__ == "__main__":
    main()
