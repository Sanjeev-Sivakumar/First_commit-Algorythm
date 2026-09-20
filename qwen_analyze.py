#!/usr/bin/env python3
"""
KineticGuard - Qwen2-VL Multimodal Ergonomic Incident Analyzer (Phase 4 CLI)
Executes vision-language analysis on Phase 3 ergonomic incidents:
- Extracts incident keyframe from camera video feed
- Ingests authoritative Phase 3 telemetry context
- Performs Qwen2-VL + LoRA multimodal inference (or robust offline fallback)
- Outputs structured JSON: posture, risk_factors, ergonomic_observation, root_cause, recommended_action, confidence
- Preserves authoritative Phase 2/3 numerical calculations

Usage:
  python qwen_analyze.py --incident output/incidents/camera_01_incidents.json
  python qwen_analyze.py --all
  python qwen_analyze.py --incident output/incidents/camera_02_incidents.json --output-dir output/vlm_analysis
"""

import argparse
import glob
import json
import os
import sys
from pathlib import Path
from typing import List, Dict, Any

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text

from src.qwen_inference import ErgonomicVLMEngine


def process_incident_file(
    file_path: Path,
    engine: ErgonomicVLMEngine,
    output_dir: Path,
    console: Console,
) -> List[Dict[str, Any]]:
    """Analyzes all incidents inside an incident JSON file."""
    if not file_path.exists():
        console.print(f"[bold red]Error: Incident file not found: {file_path}[/bold red]")
        return []

    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    camera_id = data.get("metadata", {}).get("camera_id", file_path.stem.replace("_incidents", ""))
    incidents = data.get("incidents", [])

    if not incidents:
        console.print(f"[yellow]No incidents found in {file_path.name}. Skipping.[/yellow]")
        return []

    results = []

    for inc in incidents:
        incident_id = inc.get("incident_id", "INC-UNKNOWN")
        console.print(f"\n[cyan]>> Processing Incident: [bold]{incident_id}[/bold] ({camera_id})[/cyan]")

        analysis = engine.analyze_incident(
            incident_record=inc,
            camera_id=camera_id,
        )

        # Write individual incident analysis JSON
        out_file = output_dir / f"{incident_id}_analysis.json"
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(analysis, f, indent=2)

        results.append(analysis)
        _display_analysis(analysis, console, out_file)

    return results


def _display_analysis(analysis: Dict[str, Any], console: Console, out_file: Path):
    """Renders a formatted Rich panel for the analyzed incident."""
    vlm = analysis["vlm_analysis"]
    auth = analysis["authoritative_metrics"]

    score_color = "red" if auth["risk_level"] in ("HIGH", "DANGEROUS") else "yellow"

    # Risk factor bullet points
    factors_txt = "\n".join([f"  - {f}" for f in vlm["risk_factors"]])

    content = (
        f"[bold white]Incident ID:[/bold white] {analysis['incident_id']}  |  "
        f"[bold white]Station:[/bold white] {analysis['station_id']}  |  "
        f"[bold white]Worker:[/bold white] {analysis['worker_id']}\n"
        f"[bold white]Authoritative Biomechanics:[/bold white] [{score_color}]Risk Score: {auth['cumulative_risk_score']:.1f} ({auth['risk_level']})[/{score_color}] | "
        f"Duty Cycle: {auth['awkward_duty_cycle_pct']:.1f}% | Torque: {auth['peak_lumbar_torque_nm']:.1f} Nm | Comp: {auth['peak_spinal_compression_n']:.0f} N\n"
        f"[bold white]Keyframe Extracted:[/bold white] [yellow]{analysis['keyframe_path']}[/yellow]\n\n"
        f"[bold cyan]Identified Posture:[/bold cyan] [bold white]{vlm['posture']}[/bold white] (Confidence: {vlm['confidence']*100:.1f}%)\n\n"
        f"[bold cyan]Biomechanical Risk Factors:[/bold cyan]\n{factors_txt}\n\n"
        f"[bold cyan]Ergonomic Observation:[/bold cyan]\n{vlm['ergonomic_observation']}\n\n"
        f"[bold yellow]Root Cause Diagnosis:[/bold yellow]\n{vlm['root_cause']}\n\n"
        f"[bold green]Recommended Ergonomic Action (NIOSH Standard):[/bold green]\n{vlm['recommended_action']}\n\n"
        f"[dim]Saved report: {out_file}[/dim]"
    )

    console.print(
        Panel(
            content,
            title=f"[bold]KineticGuard VLM Analysis | {analysis['incident_id']}[/bold]",
            border_style="red" if auth["risk_level"] == "HIGH" else "green",
        )
    )


def main():
    parser = argparse.ArgumentParser(
        description="KineticGuard - Qwen2-VL Multimodal Ergonomic Incident Analyzer"
    )
    parser.add_argument(
        "--incident",
        type=str,
        default=None,
        help="Path to an incident JSON file (e.g. output/incidents/camera_01_incidents.json)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Analyze all incident files found in output/incidents/",
    )
    parser.add_argument(
        "--lora-dir",
        type=str,
        default="models/qwen2_vl_ergonomic_lora",
        help="Path to the trained LoRA adapter directory",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output/vlm_analysis",
        help="Directory to save generated VLM reports and keyframes",
    )
    parser.add_argument(
        "--fallback",
        action="store_true",
        help="Force lightweight offline inference mode",
    )
    args = parser.parse_args()

    console = Console()
    console.print(
        Panel(
            "[bold cyan]KineticGuard: Qwen2-VL Multimodal Ergonomic Analyzer[/bold cyan]\n"
            "[dim]Phase 4: Vision-Language Root Cause Diagnosis & Ergonomic Action Engine[/dim]",
            border_style="cyan",
        )
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    engine = ErgonomicVLMEngine(
        lora_dir=args.lora_dir,
        keyframes_dir=str(output_dir / "keyframes"),
        force_fallback=args.fallback,
    )

    all_results = []

    if args.all:
        incident_files = sorted(glob.glob("output/incidents/*_incidents.json"))
        if not incident_files:
            console.print("[bold red]No incident files found in 'output/incidents/'. Run Phase 3 first:[/bold red]")
            console.print("[yellow]  python evaluate_risk.py --all[/yellow]")
            sys.exit(1)

        console.print(f"[dim]Found {len(incident_files)} incident file(s). Running VLM analysis...[/dim]")
        for fpath in incident_files:
            results = process_incident_file(Path(fpath), engine, output_dir, console)
            all_results.extend(results)

    elif args.incident:
        results = process_incident_file(Path(args.incident), engine, output_dir, console)
        all_results.extend(results)

    else:
        # Default to camera_01 if it exists
        default_file = Path("output/incidents/camera_01_incidents.json")
        if default_file.exists():
            console.print(f"[dim]No target specified. Defaulting to {default_file}...[/dim]")
            results = process_incident_file(default_file, engine, output_dir, console)
            all_results.extend(results)
        else:
            parser.print_help()
            sys.exit(1)

    # Save aggregated summary
    summary_path = output_dir / "all_incidents_vlm_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "system": "KineticGuard Qwen2-VL Multimodal Ergonomics",
                "phase": "Phase 4",
                "total_incidents_analyzed": len(all_results),
                "analyses": all_results,
            },
            f,
            indent=2,
        )

    console.print(
        Panel(
            f"[bold green]VLM Analysis Complete![/bold green]\n\n"
            f"- Total Incidents Analyzed: [bold white]{len(all_results)}[/bold white]\n"
            f"- Output Directory: [yellow]{output_dir}[/yellow]\n"
            f"- Aggregated Summary: [yellow]{summary_path}[/yellow]",
            border_style="green",
        )
    )


if __name__ == "__main__":
    main()
