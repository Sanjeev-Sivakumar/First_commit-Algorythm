"""
KineticGuard - Phase 3: Temporal & Cumulative Ergonomic Risk CLI
Implements:
1. Live real-time analysis directly from video streams (OpenCV -> MediaPipe -> Biomechanics -> Temporal Window -> Incidents).
2. Sliding temporal window (15–30s) tracking posture durations and awkward duty cycle.
3. Repetitive movement cycle detection and rep rate calculation.
4. Lumbar torque impulse and cumulative spinal load accumulation.
5. Peak, mean, and cumulative 0–100 ergonomic risk score.
6. Dynamic incident triggering with cooldown and real keyframe capture from the video.
7. Dynamic worker/session/station/camera identity (no hardcoded identities).
8. Export of kineticguard.strands.v1 incident packets for Strands Agents SDK.
9. Backward-compatible evaluation from telemetry files when specified.
10. Zero AWS dependencies - 100% local execution.
"""

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from src.config import settings
from src.pose_detector import PoseDetector
from src.biomechanics import BiomechanicsAnalyzer
from src.cumulative_risk import CumulativeRiskModel
from src.incident_engine import IncidentEngine, resolve_station, generate_worker_identity
from src.temporal_analyzer import TemporalSlidingWindow

console = Console()


def load_telemetry_frames(telemetry_path: Path) -> List[Dict[str, Any]]:
    """Loads frame telemetry records from JSON or CSV file."""
    if not telemetry_path.exists():
        raise FileNotFoundError(f"Telemetry file not found: {telemetry_path}")

    if telemetry_path.suffix == ".json":
        with open(telemetry_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data.get("frames", [])

    elif telemetry_path.suffix == ".csv":
        frames = []
        with open(telemetry_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rec = {}
                for k, v in row.items():
                    if k in ("frame_index",):
                        rec[k] = int(v)
                    elif k in ("pose_detected",):
                        rec[k] = v.lower() in ("true", "1")
                    elif k in ("camera_id", "posture", "risk_level"):
                        rec[k] = v
                    else:
                        try:
                            rec[k] = float(v)
                        except ValueError:
                            rec[k] = v
                frames.append(rec)
        return frames

    raise ValueError(f"Unsupported telemetry format: {telemetry_path.suffix}")


def evaluate_camera_live(
    video_path: str,
    camera_id: str,
    output_dir: Path,
    window_sec: float = 20.0,
    incident_threshold: float = 50.0,
    max_frames: Optional[int] = None,
    session_id: Optional[str] = None,
    worker_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Evaluates real camera video directly through the live pipeline:
    OpenCV -> MediaPipe -> Biomechanics -> Temporal Sliding Window -> Cumulative Risk -> Real Incidents (with keyframe capture).
    Zero precomputed or static files.
    """
    video_file = Path(video_path)
    if not video_file.exists():
        console.print(f"[bold red]Video file not found: {video_file}[/bold red]")
        return {}

    station_id = resolve_station(camera_id)
    session_id = session_id or f"sess_{int(time.time())}_{camera_id}"
    worker_id = worker_id or generate_worker_identity(camera_id, session_id)

    cap = cv2.VideoCapture(str(video_file))
    if not cap.isOpened():
        console.print(f"[bold red]Failed to open video: {video_file}[/bold red]")
        return {}

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 100
    frames_to_read = min(total_video_frames, max_frames) if (max_frames and max_frames > 0) else total_video_frames

    pose_detector = PoseDetector()
    biomech = BiomechanicsAnalyzer(body_weight_kg=75.0, load_weight_kg=12.0)
    sliding_window = TemporalSlidingWindow(window_seconds=window_sec)
    risk_model = CumulativeRiskModel()

    keyframes_dir = output_dir / "incidents" / "keyframes"
    keyframes_dir.mkdir(parents=True, exist_ok=True)

    incident_engine = IncidentEngine(
        camera_id=camera_id,
        incident_threshold=incident_threshold,
        cooldown_sec=1.5,
        worker_id=worker_id,
        station_id=station_id,
        session_id=session_id,
        source=f"video_{video_file.name}",
        keyframes_dir=keyframes_dir,
    )

    all_scores: List[float] = []
    latest_metrics: Dict[str, Any] = {}
    fidx = 0
    last_timestamp = 0.0

    while cap.isOpened() and fidx < frames_to_read:
        ret, frame = cap.read()
        if not ret:
            break

        timestamp_sec = fidx / fps
        last_timestamp = timestamp_sec

        # 1. Pose detection
        pose_res = pose_detector.detect_pose(frame)

        # 2. Biomechanical calculation
        bio_metrics = biomech.analyze(
            angles=pose_res.angles,
            landmarks_3d=pose_res.landmarks_3d if pose_res.detected else None,
        )

        # 3. Temporal sliding window
        telemetry_frame = {
            "frame_index": fidx,
            "timestamp_sec": timestamp_sec,
            "pose_detected": pose_res.detected,
            "posture": bio_metrics.posture,
            "risk_level": bio_metrics.risk_level,
            "risk_score": bio_metrics.risk_score,
            "lumbar_torque_nm": bio_metrics.lumbar_torque_nm,
            "compression_force_n": bio_metrics.compression_force_n,
            "trunk_flexion_deg": bio_metrics.trunk_flexion_deg,
        }
        sliding_window.add_frame(telemetry_frame)
        temporal_metrics = sliding_window.get_metrics()
        latest_metrics = temporal_metrics

        # 4. Cumulative risk score
        risk_score = risk_model.evaluate(temporal_metrics)
        all_scores.append(risk_score.score)

        # 5. Incident engine evaluation with frame passing for real keyframe capture
        incident_engine.evaluate_frame(
            timestamp_sec=timestamp_sec,
            temporal_metrics=temporal_metrics,
            risk_score=risk_score,
            frame=frame,
            frame_index=fidx,
        )
        fidx += 1

    cap.release()

    # Flush any active incident at end of stream
    incident_engine.flush(final_timestamp_sec=last_timestamp)

    # Export Incidents JSON & Strands JSON
    incidents_dir = output_dir / "incidents"
    incidents_dir.mkdir(parents=True, exist_ok=True)
    incidents_file = incidents_dir / f"{camera_id}_incidents.json"
    strands_file = incidents_dir / f"{camera_id}_strands_incidents.json"

    inc_dicts = []
    strands_packets = []
    for inc in incident_engine.incidents:
        d = inc.to_dict()
        s_pkt = inc.to_strands_incident_packet()
        d["strands_packet"] = s_pkt
        inc_dicts.append(d)
        strands_packets.append(s_pkt)

    incidents_payload = {
        "metadata": {
            "system": "KineticGuard Incident Engine (Live Build It)",
            "phase": "Phase 3",
            "schema_version": "kineticguard.strands.v1",
            "camera_id": camera_id,
            "worker_id": incident_engine.worker_id,
            "station_id": incident_engine.station_id,
            "session_id": session_id,
            "window_seconds": window_sec,
            "incident_threshold": incident_threshold,
            "total_incidents_count": len(incident_engine.incidents),
            "is_estimated_model_indicator": True,
            "disclaimer": "Biomechanical indicators (torque/compression) are modeled estimates and not direct clinical diagnostic measurements.",
        },
        "incidents": inc_dicts,
    }

    with open(incidents_file, "w", encoding="utf-8") as f:
        json.dump(incidents_payload, f, indent=2)

    with open(strands_file, "w", encoding="utf-8") as f:
        json.dump({
            "schema_version": "kineticguard.strands.v1",
            "event_type": "ERGONOMIC_INCIDENT_BATCH",
            "camera_id": camera_id,
            "station_id": incident_engine.station_id,
            "worker_id": incident_engine.worker_id,
            "session_id": session_id,
            "incidents": strands_packets,
        }, f, indent=2)

    peak_score = max(all_scores) if all_scores else 0.0
    mean_score = sum(all_scores) / max(1, len(all_scores))

    session_stat = {
        "camera_id": camera_id,
        "worker_id": incident_engine.worker_id,
        "station_id": incident_engine.station_id,
        "session_id": session_id,
        "session_duration_sec": round(last_timestamp, 2),
        "total_frames_analyzed": fidx,
        "window_size_sec": window_sec,
        "awkward_duty_cycle_pct": latest_metrics.get("awkward_duty_cycle_pct", 0.0),
        "total_repetition_cycles": sliding_window.cycle_detector.completed_lifts + sliding_window.cycle_detector.completed_bends,
        "completed_lifts": sliding_window.cycle_detector.completed_lifts,
        "completed_bends": sliding_window.cycle_detector.completed_bends,
        "rep_rate_per_min": latest_metrics.get("rep_rate_per_min", 0.0),
        "mean_lumbar_torque_nm": latest_metrics.get("mean_lumbar_torque_nm", 0.0),
        "peak_lumbar_torque_nm": latest_metrics.get("peak_lumbar_torque_nm", 0.0),
        "mean_spinal_compression_n": latest_metrics.get("mean_spinal_compression_n", 0.0),
        "peak_spinal_compression_n": latest_metrics.get("peak_spinal_compression_n", 0.0),
        "mean_cumulative_score": round(mean_score, 1),
        "peak_cumulative_score": round(peak_score, 1),
        "risk_category": "DANGEROUS" if peak_score >= 80 else ("HIGH" if peak_score >= 60 else ("MODERATE" if peak_score >= 30 else "LOW")),
        "incidents_triggered": len(incident_engine.incidents),
        "incidents_file": str(incidents_file),
        "strands_file": str(strands_file),
        "is_estimated_model_indicator": True,
    }

    _display_camera_report(camera_id, session_stat, incident_engine.incidents)
    return session_stat


def evaluate_camera_stream(
    camera_id: str,
    telemetry_file: Path,
    output_dir: Path,
    window_sec: float = 20.0,
    incident_threshold: float = 60.0,
    session_id: Optional[str] = None,
    worker_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Applies sliding temporal window analysis, evaluates cumulative risk,
    and records ergonomic incidents from a telemetry file (backward-compatibility mode).
    """
    frames = load_telemetry_frames(telemetry_file)
    if not frames:
        console.print(f"[yellow]Warning: No telemetry frames found in {telemetry_file}[/yellow]")
        return {}

    session_id = session_id or f"sess_{camera_id}"
    worker_id = worker_id or generate_worker_identity(camera_id, session_id)
    station_id = resolve_station(camera_id)

    sliding_window = TemporalSlidingWindow(window_seconds=window_sec)
    risk_model = CumulativeRiskModel()
    incident_engine = IncidentEngine(
        camera_id=camera_id,
        incident_threshold=incident_threshold,
        session_id=session_id,
        worker_id=worker_id,
        station_id=station_id,
    )

    all_scores: List[float] = []
    latest_metrics: Dict[str, Any] = {}
    last_timestamp = 0.0

    for frame in frames:
        ts = frame.get("timestamp_sec", 0.0)
        last_timestamp = max(last_timestamp, ts)

        sliding_window.add_frame(frame)
        temporal_metrics = sliding_window.get_metrics()
        latest_metrics = temporal_metrics

        risk_score = risk_model.evaluate(temporal_metrics)
        all_scores.append(risk_score.score)

        incident_engine.evaluate_frame(ts, temporal_metrics, risk_score)

    incident_engine.flush(final_timestamp_sec=last_timestamp)

    incidents_dir = output_dir / "incidents"
    incidents_dir.mkdir(parents=True, exist_ok=True)
    incidents_file = incidents_dir / f"{camera_id}_incidents.json"
    strands_file = incidents_dir / f"{camera_id}_strands_incidents.json"

    inc_dicts = []
    strands_packets = []
    for inc in incident_engine.incidents:
        d = inc.to_dict()
        s_pkt = inc.to_strands_incident_packet()
        d["strands_packet"] = s_pkt
        inc_dicts.append(d)
        strands_packets.append(s_pkt)

    incidents_payload = {
        "metadata": {
            "system": "KineticGuard Incident Engine",
            "phase": "Phase 3",
            "schema_version": "kineticguard.strands.v1",
            "camera_id": camera_id,
            "worker_id": incident_engine.worker_id,
            "station_id": incident_engine.station_id,
            "session_id": session_id,
            "window_seconds": window_sec,
            "incident_threshold": incident_threshold,
            "total_incidents_count": len(incident_engine.incidents),
            "is_estimated_model_indicator": True,
            "disclaimer": "Biomechanical indicators (torque/compression) are modeled estimates and not direct clinical diagnostic measurements.",
        },
        "incidents": inc_dicts,
    }

    with open(incidents_file, "w", encoding="utf-8") as f:
        json.dump(incidents_payload, f, indent=2)

    with open(strands_file, "w", encoding="utf-8") as f:
        json.dump({
            "schema_version": "kineticguard.strands.v1",
            "event_type": "ERGONOMIC_INCIDENT_BATCH",
            "camera_id": camera_id,
            "station_id": incident_engine.station_id,
            "worker_id": incident_engine.worker_id,
            "session_id": session_id,
            "incidents": strands_packets,
        }, f, indent=2)

    peak_score = max(all_scores) if all_scores else 0.0
    mean_score = sum(all_scores) / max(1, len(all_scores))

    session_stat = {
        "camera_id": camera_id,
        "worker_id": incident_engine.worker_id,
        "station_id": incident_engine.station_id,
        "session_id": session_id,
        "session_duration_sec": round(last_timestamp, 2),
        "total_frames_analyzed": len(frames),
        "window_size_sec": window_sec,
        "awkward_duty_cycle_pct": latest_metrics.get("awkward_duty_cycle_pct", 0.0),
        "total_repetition_cycles": sliding_window.cycle_detector.completed_lifts + sliding_window.cycle_detector.completed_bends,
        "completed_lifts": sliding_window.cycle_detector.completed_lifts,
        "completed_bends": sliding_window.cycle_detector.completed_bends,
        "rep_rate_per_min": latest_metrics.get("rep_rate_per_min", 0.0),
        "mean_lumbar_torque_nm": latest_metrics.get("mean_lumbar_torque_nm", 0.0),
        "peak_lumbar_torque_nm": latest_metrics.get("peak_lumbar_torque_nm", 0.0),
        "mean_spinal_compression_n": latest_metrics.get("mean_spinal_compression_n", 0.0),
        "peak_spinal_compression_n": latest_metrics.get("peak_spinal_compression_n", 0.0),
        "mean_cumulative_score": round(mean_score, 1),
        "peak_cumulative_score": round(peak_score, 1),
        "risk_category": "DANGEROUS" if peak_score >= 80 else ("HIGH" if peak_score >= 60 else ("MODERATE" if peak_score >= 30 else "LOW")),
        "incidents_triggered": len(incident_engine.incidents),
        "incidents_file": str(incidents_file),
        "strands_file": str(strands_file),
        "is_estimated_model_indicator": True,
    }

    _display_camera_report(camera_id, session_stat, incident_engine.incidents)
    return session_stat


def _display_camera_report(
    camera_id: str,
    stats: Dict[str, Any],
    incidents: List[Any],
):
    """Renders formatted tables for exposure metrics and triggered incidents."""
    table = Table(
        title=f"Temporal & Cumulative Ergonomic Exposure - {camera_id.upper()} ({stats['station_id']})",
        show_header=True,
        header_style="bold cyan",
        expand=False,
    )
    table.add_column("Exposure Parameter", style="bold white", width=28)
    table.add_column("Observed Value", style="cyan", width=34)

    table.add_row("Worker ID / Station", f"{stats['worker_id']} | {stats['station_id']}")
    table.add_row("Session Duration", f"{stats['session_duration_sec']}s ({stats['total_frames_analyzed']} frames)")
    table.add_row("Sliding Window Duration", f"{stats['window_size_sec']} seconds")
    table.add_row("Awkward Posture Duty Cycle", f"{stats['awkward_duty_cycle_pct']}% (Bending + Lifting + Squatting)")
    table.add_row("Repetition Rate", f"{stats['rep_rate_per_min']:.1f} reps/min ({stats['total_repetition_cycles']} total: {stats['completed_lifts']} lifts, {stats['completed_bends']} bends)")
    table.add_row("Peak Lumbar Torque (Model)", f"{stats['peak_lumbar_torque_nm']} Nm (Mean: {stats['mean_lumbar_torque_nm']} Nm)")
    table.add_row("Peak Spinal Compression", f"{stats['peak_spinal_compression_n']} N (NIOSH AL: 3400 N)")
    table.add_row("Cumulative Risk Score", f"{stats['peak_cumulative_score']:.1f} / 100 [{stats['risk_category']}]")
    table.add_row("Incidents Triggered", f"{stats['incidents_triggered']} event(s)")

    console.print("\n")
    console.print(table)

    if incidents:
        inc_table = Table(
            title=f"Triggered ERGONOMIC_INCIDENT Ledger - {camera_id.upper()}",
            show_header=True,
            header_style="bold red",
            expand=False,
        )
        inc_table.add_column("Incident ID", style="bold white", width=24)
        inc_table.add_column("Time", style="cyan", width=12)
        inc_table.add_column("Severity", justify="center", width=11)
        inc_table.add_column("Keyframe", style="dim", width=20)
        inc_table.add_column("Corrective Action", style="green")

        for inc in incidents:
            d = inc.to_dict() if hasattr(inc, "to_dict") else inc
            time_str = f"{d['start_time_sec']:.1f}s - {d['end_time_sec']:.1f}s"
            severity_style = (
                "[bold red]DANGEROUS[/bold red]"
                if d["risk_level"] == "DANGEROUS"
                else "[bold yellow]HIGH[/bold yellow]"
            )
            kf_name = Path(d["keyframe_path"]).name if d.get("keyframe_path") else "stream_ref"
            inc_table.add_row(
                d["incident_id"],
                time_str,
                severity_style,
                kf_name,
                d["recommended_action"][:65] + ("..." if len(d["recommended_action"]) > 65 else ""),
            )

        console.print(inc_table)
    else:
        console.print(f"  [green][OK] No ergonomic threshold breaches triggered for {camera_id}.[/green]")


def _save_session_summaries(
    output_dir: Path, session_stats: List[Dict[str, Any]]
) -> Tuple[str, str]:
    """Exports cross-camera session summaries in JSON and CSV."""
    summary_dir = output_dir / "risk_summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    json_path = summary_dir / "session_summary.json"
    csv_path = summary_dir / "session_summary.csv"

    payload = {
        "metadata": {
            "system": "KineticGuard Cumulative Ergonomics Platform",
            "phase": "Phase 3 (Build It)",
            "schema_version": "kineticguard.strands.v1",
            "evaluated_stations_count": len(session_stats),
            "is_estimated_model_indicator": True,
            "disclaimer": "Biomechanical indicators are modeled estimates and not direct clinical diagnostic measurements.",
        },
        "station_summaries": session_stats,
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    if session_stats:
        keys = list(session_stats[0].keys())
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            for s in session_stats:
                writer.writerow(s)

    return str(json_path), str(csv_path)


def main():
    parser = argparse.ArgumentParser(
        description="KineticGuard Phase 3: Temporal & Cumulative Ergonomic Risk Engine (Build It Architecture)"
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--camera",
        type=str,
        default="camera_01",
        help="Target camera ID (camera_01, camera_02, camera_03)",
    )
    group.add_argument(
        "--all",
        action="store_true",
        help="Evaluate all 3 camera streams in batch and generate cross-station risk summary",
    )
    parser.add_argument(
        "--video",
        type=str,
        default=None,
        help="Path to specific video file to analyze",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Limit number of frames to analyze per camera (default: full video)",
    )
    parser.add_argument(
        "--telemetry-file",
        type=str,
        default=None,
        help="Path to existing telemetry JSON/CSV (fallback file mode)",
    )
    parser.add_argument(
        "--session-id",
        type=str,
        default=None,
        help="Custom session identifier (default: auto-generated dynamic)",
    )
    parser.add_argument(
        "--worker-id",
        type=str,
        default=None,
        help="Custom worker identifier (default: dynamic derived from camera & session)",
    )
    parser.add_argument(
        "--window-sec",
        type=float,
        default=20.0,
        help="Sliding temporal window size in seconds (15-30s, default: 20.0)",
    )
    parser.add_argument(
        "--incident-threshold",
        type=float,
        default=50.0,
        help="Cumulative risk score threshold to trigger ERGONOMIC_INCIDENT (default: 50.0)",
    )
    parser.add_argument(
        "--telemetry-dir",
        type=str,
        default="output/telemetry",
        help="Directory containing Phase 2 telemetry files (default: output/telemetry)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output",
        help="Directory to save incidents and risk summaries (default: output)",
    )

    args = parser.parse_args()

    header = Panel(
        "[bold cyan]KineticGuard Phase 3: Temporal + Cumulative Ergonomic Risk Engine[/bold cyan]\n"
        "[dim]Real Video -> Pose -> Biomechanics -> Temporal Risk -> Real Incidents -> Strands Packets[/dim]\n"
        "[dim yellow]Model Indicators: Modeled estimates, non-medical. Zero AWS cloud dependency.[/dim yellow]",
        border_style="cyan",
        expand=False,
    )
    console.print(header)

    output_dir = Path(args.output_dir)
    session_results = []

    # Check if user explicitly asked for a telemetry file
    if args.telemetry_file:
        tfile = Path(args.telemetry_file)
        cid = args.camera or tfile.stem.replace("_telemetry", "")
        console.print(f"\n[bold cyan]-> Evaluating from telemetry file: {tfile.name}...[/bold cyan]")
        res = evaluate_camera_stream(
            camera_id=cid,
            telemetry_file=tfile,
            output_dir=output_dir,
            window_sec=args.window_sec,
            incident_threshold=args.incident_threshold,
            session_id=args.session_id,
            worker_id=args.worker_id,
        )
        if res:
            session_results.append(res)
    else:
        # Live video processing mode (Build It Architecture)
        targets: List[Tuple[str, str]] = []
        if args.all:
            for cid in settings.CAMERA_IDS:
                targets.append((cid, settings.CAMERA_SOURCES[cid]))
        else:
            cid = args.camera
            vpath = args.video or settings.CAMERA_SOURCES.get(cid, f"data/{cid}.mp4")
            targets.append((cid, vpath))

        for cid, vpath in targets:
            console.print(f"\n[bold cyan]-> Live Processing Video Stream: {cid} ({Path(vpath).name})...[/bold cyan]")
            res = evaluate_camera_live(
                video_path=vpath,
                camera_id=cid,
                output_dir=output_dir,
                window_sec=args.window_sec,
                incident_threshold=args.incident_threshold,
                max_frames=args.max_frames,
                session_id=args.session_id,
                worker_id=args.worker_id,
            )
            if res:
                session_results.append(res)

    # Cross-Camera Summary if multiple processed
    if session_results:
        json_sum, csv_sum = _save_session_summaries(output_dir, session_results)

        if len(session_results) > 1:
            comp_table = Table(
                title="Cross-Station Cumulative Ergonomic Risk Ranking (Live)",
                show_header=True,
                header_style="bold cyan",
                expand=False,
            )
            comp_table.add_column("Station", style="bold white", width=18)
            comp_table.add_column("Worker ID", style="cyan", width=16)
            comp_table.add_column("Awkward Duty", justify="right", width=14)
            comp_table.add_column("Rep Rate", justify="right", width=14)
            comp_table.add_column("Peak Torque", justify="right", width=13)
            comp_table.add_column("Peak Comp.", justify="right", width=15)
            comp_table.add_column("Risk Score", justify="center", width=12)
            comp_table.add_column("Category", justify="center", width=12)
            comp_table.add_column("Incidents", justify="center", width=11)

            sorted_stats = sorted(session_results, key=lambda x: x["peak_cumulative_score"], reverse=True)
            for s in sorted_stats:
                cat_style = (
                    "[bold red]DANGEROUS[/bold red]"
                    if s["risk_category"] == "DANGEROUS"
                    else ("[bold yellow]HIGH[/bold yellow]" if s["risk_category"] == "HIGH" else "[green]LOW/MOD[/green]")
                )
                comp_table.add_row(
                    s["station_id"],
                    s["worker_id"],
                    f"{s['awkward_duty_cycle_pct']:.1f}%",
                    f"{s['rep_rate_per_min']:.1f}/m",
                    f"{s['peak_lumbar_torque_nm']:.1f} Nm",
                    f"{s['peak_spinal_compression_n']:.0f} N",
                    f"{s['peak_cumulative_score']:.1f}",
                    cat_style,
                    str(s["incidents_triggered"]),
                )

            console.print("\n")
            console.print(comp_table)

        console.print(f"\n[green][OK] Session Summaries saved to:[/green]")
        console.print(f"  - JSON: {json_sum}")
        console.print(f"  - CSV:  {csv_sum}")

    console.print("\n[bold green]KineticGuard Phase 3 Live Temporal Evaluation completed successfully![/bold green]")


if __name__ == "__main__":
    main()
