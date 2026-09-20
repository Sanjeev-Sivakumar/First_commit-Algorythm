#!/usr/bin/env python3
"""
KineticGuard - Generate Ergonomic Dataset (Phase 4 CLI)
Generates the 6-class local ergonomic dataset for Qwen2-VL LoRA fine-tuning.

Classes:
  1. UPRIGHT: Neutral spinal alignment, low torque & compression.
  2. BENDING: Forward stoop bending, extended moment arm.
  3. LIFTING: Manual crate lift with load held away from body.
  4. AWKWARD_POSTURE: Over-reaching or severe torso deflection (> 45 deg).
  5. REPETITIVE_MOTION: High cycle frequency (> 15 reps/min) and > 85% duty cycle.
  6. HIGH_RISK_LIFTING: Heavy load with deep stoop bending breaching NIOSH limits (> 3400 N).

Usage:
  python generate_dataset.py
  python generate_dataset.py --samples-per-class 3 --output-dir data/ergonomic_dataset
"""

import argparse
import sys
from pathlib import Path
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from src.dataset_generator import ErgonomicDatasetGenerator, get_dataset_stats


def main():
    parser = argparse.ArgumentParser(
        description="KineticGuard - Generate Local Ergonomic Dataset for Qwen2-VL LoRA"
    )
    parser.add_argument(
        "--samples-per-class",
        type=int,
        default=2,
        help="Number of variations to generate per ergonomic category (default: 2)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/ergonomic_dataset",
        help="Target output directory for dataset annotations and images",
    )
    args = parser.parse_args()

    console = Console()
    console.print(
        Panel(
            "[bold cyan]KineticGuard: Local Ergonomic Dataset Generator[/bold cyan]\n"
            "[dim]Phase 4: Qwen2-VL + LoRA Multimodal Fine-Tuning Corpus[/dim]",
            border_style="cyan",
        )
    )

    console.print(f"[dim]Generating {args.samples_per_class} sample(s) per category into '{args.output_dir}'...[/dim]")

    generator = ErgonomicDatasetGenerator(output_dir=args.output_dir)
    results = generator.generate_all(num_samples_per_class=args.samples_per_class)

    stats = get_dataset_stats(results["annotations_file"])

    table = Table(title="Generated Ergonomic Dataset Summary", border_style="green")
    table.add_column("Category", style="cyan", no_wrap=True)
    table.add_column("Biomechanics Focus", style="white")
    table.add_column("Risk Threshold", style="magenta")

    table.add_row("UPRIGHT", "Neutral spinal posture (Trunk < 10 deg)", "LOW (< 30)")
    table.add_row("BENDING", "Stoop bending flexion (Trunk ~40 deg)", "HIGH (60-80)")
    table.add_row("LIFTING", "Manual parcel lift with moment arm (~45 cm)", "HIGH (65-80)")
    table.add_row("AWKWARD_POSTURE", "Over-reach across table (> 50 cm, Trunk > 45 deg)", "HIGH (75-85)")
    table.add_row("REPETITIVE_MOTION", "High cycle rate (> 15 reps/min, > 90% duty cycle)", "HIGH (80-88)")
    table.add_row("HIGH_RISK_LIFTING", "Heavy payload (> 15 kg), spinal compression > 3400 N", "DANGEROUS (> 90)")

    console.print(table)

    console.print(
        Panel(
            f"[bold green]Dataset Generation Complete![/bold green]\n\n"
            f"- Total Samples: [bold white]{results['total_samples']}[/bold white]\n"
            f"- Annotations JSON: [yellow]{results['annotations_file']}[/yellow]\n"
            f"- ChatML Format: [yellow]{results['chatml_file']}[/yellow]\n"
            f"- Keyframe Images: [yellow]{results['images_dir']}[/yellow]",
            border_style="green",
        )
    )


if __name__ == "__main__":
    main()
