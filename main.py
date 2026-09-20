"""
KineticGuard - Phase 1: Multi-Camera Video Ingestion (Build It Architecture)
Main entry point for concurrent multi-camera video ingestion using real OpenCV decoding,
dynamic session tracking, configurable frame sampling, and zero AWS cloud dependencies.
"""

import argparse
import signal
import sys
import time
from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from src.config import settings
from src.logger import log_event, logger
from src.producer import MultiCameraIngestionManager
from src.video_source import SyntheticVideoGenerator

console = Console()


def print_banner(use_kvs: bool, mock_mode: bool, sample_stride: int):
    """Display startup banner and configuration summary."""
    if use_kvs:
        status_text = (
            "[bold yellow]AWS KVS SIMULATION / MOCK[/bold yellow] (Local video decoding & simulated KVS)"
            if mock_mode
            else f"[bold green]LIVE AWS KVS STREAMING[/bold green] (Region: {settings.AWS_REGION})"
        )
        platform_text = "AWS Kinesis Video Streams (KVS)"
    else:
        status_text = "[bold green]LOCAL BUILD IT INGESTION[/bold green] (Zero AWS Credentials Required • Real OpenCV Decoding)"
        platform_text = "Build It Local Pipeline (Strands / OpenSearch / LocalStack Ready)"

    details = f"""
[bold cyan]Phase 1:[/bold cyan] Multi-Camera Video Ingestion (Build It Audit & Implementation)
[bold cyan]Platform:[/bold cyan] {platform_text}
[bold cyan]Mode:[/bold cyan] {status_text}
[bold cyan]Cameras:[/bold cyan] {', '.join(settings.CAMERA_IDS)}
[bold cyan]Target FPS:[/bold cyan] {settings.FPS_TARGET} (Sampling Stride: {sample_stride})
[bold cyan]AWS Dependency:[/bold cyan] [bold yellow]NONE (100% Local Processing)[/bold yellow]
"""
    panel = Panel(
        details.strip(),
        title="[bold white]KineticGuard Video Ingestion Engine (Build It Architecture)[/bold white]",
        border_style="cyan",
        expand=False,
    )
    console.print(panel)


def render_dashboard(manager: MultiCameraIngestionManager):
    """Renders a real-time status table of active camera streams."""
    metrics_list = manager.get_dashboard_metrics()

    table = Table(
        title="Active Multi-Camera Ingestion Feeds",
        show_header=True,
        header_style="bold cyan",
        expand=False,
    )
    table.add_column("Camera", style="bold white", width=10)
    table.add_column("Session ID", style="cyan", width=26)
    table.add_column("Status", justify="center", width=12)
    table.add_column("Mode", justify="center", width=8)
    table.add_column("FPS", justify="right", width=6)
    table.add_column("Frames Read", justify="right", width=12)
    table.add_column("MB Decoded", justify="right", width=11)
    table.add_column("Uptime", justify="right", width=8)

    for m in metrics_list:
        status = m.get("status", "UNKNOWN")
        if status == "LIVE":
            status_style = "[bold green]LIVE[/bold green]"
        elif status == "STALE":
            status_style = "[bold yellow]STALE[/bold yellow]"
        elif status == "NO DATA":
            status_style = "[bold red]NO DATA[/bold red]"
        elif status in ("STREAMING", "CONNECTING"):
            status_style = "[bold green]STREAMING[/bold green]"
        elif status == "RECONNECTING":
            status_style = "[bold yellow]RECONNECT[/bold yellow]"
        else:
            status_style = f"[dim]{status}[/dim]"

        table.add_row(
            m["camera_id"],
            m.get("session_id", "--"),
            status_style,
            m.get("mode", "LOCAL"),
            f"{m['fps']:.1f}",
            str(m["frames_read"]),
            f"{m['bytes_sent_mb']:.2f}",
            f"{m.get('uptime_sec', 0)}s",
        )

    console.print(table)


def main():
    parser = argparse.ArgumentParser(
        description="KineticGuard Phase 1: Multi-Camera Video Ingestion (Build It Architecture)"
    )
    parser.add_argument(
        "--use-kvs",
        action="store_true",
        default=False,
        help="Enable AWS Kinesis Video Streams ingestion (default: False, runs local Build It ingestion)",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        default=False,
        help="Run KVS in local simulation mode (only applies if --use-kvs is enabled)",
    )
    parser.add_argument(
        "--duration",
        type=int,
        default=0,
        help="Run ingestion for N seconds then exit gracefully (0 = run indefinitely)",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=None,
        help=f"Target frame rate for video playback (default: {settings.FPS_TARGET})",
    )
    parser.add_argument(
        "--sample-stride",
        type=int,
        default=settings.SAMPLE_STRIDE,
        help="Frame sampling stride (1 = process every frame, 2 = every second frame; default: 1)",
    )
    parser.add_argument(
        "--session-prefix",
        type=str,
        default=None,
        help="Optional custom prefix for dynamic runtime session IDs",
    )
    parser.add_argument(
        "--refresh-interval",
        type=int,
        default=3,
        help="Dashboard stats refresh interval in seconds (default: 3)",
    )

    args = parser.parse_args()

    if args.fps:
        settings.FPS_TARGET = args.fps

    use_kvs = args.use_kvs
    mock_mode = args.mock or settings.MOCK_AWS or not settings.has_aws_credentials()

    print_banner(use_kvs=use_kvs, mock_mode=mock_mode, sample_stride=args.sample_stride)

    # 1. Ensure camera video files exist
    console.print("[cyan]Verifying camera video source files...[/cyan]")
    for cam_id in settings.CAMERA_IDS:
        vpath = Path(settings.CAMERA_SOURCES[cam_id])
        if vpath.exists():
            console.print(f"  [green]Found real camera source: {vpath} ({vpath.stat().st_size / (1024*1024):.2f} MB)[/green]")
        else:
            console.print(f"  [yellow]Generating sample camera feed: {vpath}...[/yellow]")
            SyntheticVideoGenerator.ensure_all_sample_videos()
            break

    # 2. Initialize Ingestion Manager
    manager = MultiCameraIngestionManager(
        mock_mode=mock_mode,
        use_kvs=use_kvs,
        sample_stride=args.sample_stride,
        target_fps=settings.FPS_TARGET,
        session_id_prefix=args.session_prefix,
    )
    manager.initialize_producers()

    # Register graceful shutdown handlers
    def shutdown_signal_handler(sig, frame):
        console.print("\n[bold yellow]Interrupt signal received. Shutting down camera streams...[/bold yellow]")
        manager.stop()
        console.print("[bold green]Shutdown complete. Exiting.[/bold green]")
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown_signal_handler)
    signal.signal(signal.SIGTERM, shutdown_signal_handler)

    # 3. Start ingestion
    manager.start()

    console.print(
        "\n[bold green]Ingestion active. Press Ctrl+C at any time to stop.[/bold green]\n"
    )

    start_time = time.time()
    last_render_time = start_time

    try:
        while True:
            time.sleep(1)
            now = time.time()

            if now - last_render_time >= args.refresh_interval:
                render_dashboard(manager)
                last_render_time = now

            if args.duration > 0 and (now - start_time) >= args.duration:
                console.print(
                    f"\n[cyan]Duration limit ({args.duration}s) reached. Terminating ingestion demo...[/cyan]"
                )
                break

    except KeyboardInterrupt:
        console.print("\n[bold yellow]Stopping streams...[/bold yellow]")
    finally:
        manager.stop()
        render_dashboard(manager)
        console.print("[bold green]KineticGuard Multi-Camera Ingestion Finished.[/bold green]")


if __name__ == "__main__":
    main()
