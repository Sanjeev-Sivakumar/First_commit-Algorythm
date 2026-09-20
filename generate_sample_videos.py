"""
KineticGuard - Sample Video Generator
Generates simulated CCTV surveillance video clips for camera_01, camera_02, and camera_03.
"""

import argparse
import sys
from pathlib import Path
from rich.console import Console

from src.config import settings
from src.video_source import SyntheticVideoGenerator

console = Console()


def main():
    parser = argparse.ArgumentParser(
        description="Generate simulated CCTV surveillance videos for KineticGuard Phase 1"
    )
    parser.add_argument(
        "--duration",
        type=int,
        default=10,
        help="Duration of each simulated video in seconds (default: 10)",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=25,
        help="Video frame rate in frames per second (default: 25)",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=640,
        help="Video width in pixels (default: 640)",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=480,
        help="Video height in pixels (default: 480)",
    )
    args = parser.parse_args()

    console.print(
        "[bold cyan]======================================================[/bold cyan]"
    )
    console.print(
        "[bold cyan]   KineticGuard: Generating Simulated Camera Feeds    [/bold cyan]"
    )
    console.print(
        "[bold cyan]======================================================[/bold cyan]\n"
    )

    cameras = [
        ("camera_01", settings.CAMERA_SOURCES["camera_01"], "Front Entrance CCTV"),
        ("camera_02", settings.CAMERA_SOURCES["camera_02"], "Warehouse Corridor CCTV"),
        ("camera_03", settings.CAMERA_SOURCES["camera_03"], "Perimeter Gate CCTV"),
    ]

    for camera_id, path_str, label in cameras:
        out_path = Path(path_str)
        console.print(
            f"[yellow]-> Generating {camera_id} ({label}) -> {out_path}...[/yellow]"
        )

        created_file = SyntheticVideoGenerator.generate_camera_video(
            output_path=str(out_path),
            camera_id=camera_id,
            title=label,
            duration_sec=args.duration,
            fps=args.fps,
            width=args.width,
            height=args.height,
        )

        size_kb = Path(created_file).stat().st_size / 1024
        console.print(
            f"  [green][OK] Successfully created {out_path.name} ({size_kb:.1f} KB, {args.duration}s @ {args.fps} FPS)[/green]"
        )

    console.print(
        "\n[bold green]All simulated camera video files generated successfully![/bold green]"
    )


if __name__ == "__main__":
    main()
