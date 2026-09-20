"""
KineticGuard - Phase 2: Pose Estimation & Biomechanical Analysis CLI
Processes video frames with MediaPipe Pose, computes joint angles, classifies postures,
calculates L5/S1 spinal torque & compressive forces (estimated model indicators),
exports structured telemetry for Strands Agents SDK, and renders annotated output videos.
"""

import argparse
import sys
import time
from pathlib import Path
from typing import List, Optional
import cv2
from rich.console import Console
from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.table import Table

from src.biomechanics import BiomechanicsAnalyzer
from src.config import settings
from src.logger import logger
from src.pose_detector import PoseDetector
from src.telemetry_exporter import TelemetryExporter
from src.visualizer import BiomechanicalVisualizer

console = Console()


def analyze_single_stream(
    video_path: str,
    camera_id: str,
    output_dir: Path,
    session_id: Optional[str] = None,
    worker_id: Optional[str] = None,
    sample_stride: int = 1,
    body_weight_kg: float = 75.0,
    load_weight_kg: float = 10.0,
    max_frames: int = 0,
    save_video: bool = True,
) -> dict:
    """Processes a video feed, extracts pose landmarks, computes biomechanics, and exports results."""
    input_file = Path(video_path)
    if not input_file.exists():
        console.print(f"[bold red]Error: Video file not found: {input_file}[/bold red]")
        return {}

    # Dynamic session and worker identification (Zero hardcoded identities)
    sess_id = session_id or f"sess_{int(time.time())}_{camera_id}"
    clean_cam = camera_id.replace("camera_", "C").upper()
    wrk_id = worker_id or f"W-{clean_cam}-{sess_id[-4:]}"

    console.print(
        f"\n[bold cyan]-> Starting Biomechanical Analysis on {camera_id} ({input_file.name})...[/bold cyan]\n"
        f"   [dim]Session: {sess_id} | Operative: {wrk_id} | Stride: {sample_stride}[/dim]"
    )

    cap = cv2.VideoCapture(str(input_file))
    if not cap.isOpened():
        console.print(f"[bold red]Failed to open video: {input_file}[/bold red]")
        return {}

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 100

    if max_frames > 0:
        frames_to_process = min(max_frames, total_frames)
    else:
        frames_to_process = total_frames

    # Prepare output paths
    telemetry_dir = output_dir / "telemetry"
    telemetry_dir.mkdir(parents=True, exist_ok=True)
    json_path = telemetry_dir / f"{camera_id}_telemetry.json"
    csv_path = telemetry_dir / f"{camera_id}_telemetry.csv"
    video_out_path = output_dir / f"{camera_id}_annotated.mp4"

    # Initialize components
    pose_detector = PoseDetector()
    biomech_analyzer = BiomechanicsAnalyzer(
        body_weight_kg=body_weight_kg,
        load_weight_kg=load_weight_kg,
    )
    exporter = TelemetryExporter(
        camera_id=camera_id,
        session_id=sess_id,
        worker_id=wrk_id,
    )
    visualizer = BiomechanicalVisualizer(camera_id=camera_id)

    video_writer = None
    if save_video:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video_writer = cv2.VideoWriter(
            str(video_out_path), fourcc, fps, (width, height)
        )

    start_time = time.time()
    raw_frame_idx = 0
    sampled_frame_idx = 0

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task(f"Analyzing {camera_id}", total=frames_to_process)

        while raw_frame_idx < frames_to_process:
            ret, frame = cap.read()
            if not ret or frame is None:
                break

            raw_frame_idx += 1
            progress.update(task, advance=1)

            if sample_stride > 1 and ((raw_frame_idx - 1) % sample_stride != 0):
                continue

            sampled_frame_idx += 1
            timestamp_sec = raw_frame_idx / fps

            # 1. Real 3D Pose Landmark Detection
            pose_res = pose_detector.detect_pose(frame)

            # 2. Deterministic Biomechanics & Posture Analysis
            biomech = biomech_analyzer.analyze(
                angles=pose_res.angles,
                landmarks_3d=pose_res.landmarks_3d if pose_res.detected else None,
            )

            # 3. Record Telemetry (Estimated Model Indicators)
            exporter.record_frame(
                frame_index=raw_frame_idx,
                timestamp_sec=timestamp_sec,
                pose_detected=pose_res.detected,
                angles=pose_res.angles,
                biomech_dict=biomech.to_dict(),
                keypoints=pose_res.landmarks_3d if pose_res.detected else None,
            )

            # 4. Render Video Overlay
            if save_video and video_writer is not None:
                annotated_frame = visualizer.draw_frame(
                    frame_bgr=frame,
                    pose_res=pose_res,
                    biomech=biomech,
                    frame_idx=raw_frame_idx,
                    fps=fps,
                )
                video_writer.write(annotated_frame)

    cap.release()
    if video_writer:
        video_writer.release()

    elapsed = time.time() - start_time
    proc_fps = raw_frame_idx / max(0.001, elapsed)

    # Save telemetry files
    saved_json = exporter.save_json(str(json_path))
    saved_csv = exporter.save_csv(str(csv_path))

    summary = exporter.get_summary()
    summary["processing_time_sec"] = round(elapsed, 2)
    summary["processing_fps"] = round(proc_fps, 1)
    summary["output_video"] = str(video_out_path) if save_video else None
    summary["output_json"] = str(json_path)
    summary["output_csv"] = str(csv_path)

    # Render Summary Card
    _display_summary_table(camera_id, summary, proc_fps, elapsed)

    return summary


def _display_summary_table(camera_id: str, summary: dict, proc_fps: float, elapsed: float):
    """Prints a styled telemetry summary table for the analyzed video feed."""
    table = Table(
        title=f"Biomechanical Telemetry Summary - {camera_id.upper()}",
        show_header=True,
        header_style="bold cyan",
        expand=False,
    )
    table.add_column("Metric", style="bold white", width=28)
    table.add_column("Value", style="cyan", width=42)

    table.add_row("Operative ID", str(summary.get("worker_id", "--")))
    table.add_row("Session ID", str(summary.get("session_id", "--")))
    table.add_row("Frames Processed", f"{summary.get('total_frames')} frames ({proc_fps:.1f} FPS, {elapsed:.1f}s)")
    table.add_row("Pose Detection Rate", f"{summary.get('detection_rate_pct')}% ({summary.get('detected_frames')} frames)")
    table.add_row("Mean Lumbar Torque (L5/S1)", f"{summary.get('mean_lumbar_torque_nm')} Nm [Estimated Model Indicator]")
    table.add_row("Peak Lumbar Torque", f"{summary.get('peak_lumbar_torque_nm')} Nm [Estimated Model Indicator]")
    table.add_row("Mean Spinal Compression", f"{summary.get('mean_compression_force_n')} N [Estimated Model Indicator]")
    table.add_row("Peak Spinal Compression", f"{summary.get('peak_compression_force_n')} N (NIOSH Action Limit: 3400 N)")
    table.add_row("Clinical Disclaimer", "Non-medical; estimated model indicator")

    # Postures breakdown
    postures = summary.get("posture_breakdown", {})
    posture_str = " | ".join([f"{k}: {v}" for k, v in postures.items()])
    table.add_row("Posture Classification", posture_str)

    # Risk breakdown
    risks = summary.get("risk_level_breakdown", {})
    risk_str = " | ".join([f"{k}: {v}" for k, v in risks.items()])
    table.add_row("Ergonomic Risk Levels", risk_str)

    # Output paths
    table.add_row("Telemetry JSON", summary.get("output_json"))
    table.add_row("Telemetry CSV", summary.get("output_csv"))
    if summary.get("output_video"):
        table.add_row("Annotated Video", summary.get("output_video"))

    console.print("\n")
    console.print(table)


def main():
    parser = argparse.ArgumentParser(
        description="KineticGuard Phase 2: Pose Estimation & Biomechanical Analysis"
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--camera",
        type=str,
        default="camera_01",
        help="Camera ID (camera_01, camera_02, camera_03) or custom video path",
    )
    group.add_argument(
        "--all",
        action="store_true",
        help="Process all 3 cameras (camera_01, camera_02, camera_03) in sequence",
    )

    parser.add_argument(
        "--session-id",
        type=str,
        default=None,
        help="Custom session identifier (default: dynamically generated)",
    )
    parser.add_argument(
        "--worker-id",
        type=str,
        default=None,
        help="Custom operative identifier (default: dynamically generated from camera/session)",
    )
    parser.add_argument(
        "--sample-stride",
        type=int,
        default=1,
        help="Frame sampling stride (1 = process every frame, 2 = every second frame; default: 1)",
    )
    parser.add_argument(
        "--load-kg",
        type=float,
        default=10.0,
        help="Estimated external load weight in kg (default: 10.0)",
    )
    parser.add_argument(
        "--body-weight-kg",
        type=float,
        default=75.0,
        help="Worker body weight in kg (default: 75.0)",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="Limit frames analyzed (0 = process entire video)",
    )
    parser.add_argument(
        "--no-video",
        action="store_true",
        help="Skip annotated video creation (extract telemetry JSON/CSV only)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output",
        help="Directory to save telemetry and annotated videos (default: output)",
    )

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    header = Panel(
        "[bold cyan]KineticGuard Phase 2: Pose Estimation & Biomechanical Analysis[/bold cyan]\n"
        "[dim]Calculates Joint Angles, Ergonomic Posture, L5/S1 Lumbar Torque & Spinal Compression (Estimated Model Indicators)[/dim]",
        border_style="cyan",
        expand=False,
    )
    console.print(header)

    targets = []
    if args.all:
        for cam_id in settings.CAMERA_IDS:
            targets.append((cam_id, settings.CAMERA_SOURCES[cam_id]))
    else:
        cam_id = args.camera
        if cam_id in settings.CAMERA_SOURCES:
            video_path = settings.CAMERA_SOURCES[cam_id]
        else:
            video_path = cam_id
            cam_id = Path(cam_id).stem
        targets.append((cam_id, video_path))

    for cid, vpath in targets:
        analyze_single_stream(
            video_path=vpath,
            camera_id=cid,
            output_dir=output_dir,
            session_id=args.session_id,
            worker_id=args.worker_id,
            sample_stride=args.sample_stride,
            body_weight_kg=args.body_weight_kg,
            load_weight_kg=args.load_kg,
            max_frames=args.max_frames,
            save_video=not args.no_video,
        )

    console.print(
        "\n[bold green][OK] KineticGuard Phase 2 Analysis completed successfully![/bold green]"
    )


if __name__ == "__main__":
    main()
