"""
KineticGuard - Phase 4 CLI: Strands Agents SDK Safety Guardian Runner
Orchestrates:
Real Incident (Phase 3) -> Gemini Multimodal (Keyframe) -> Strands Agent & Tools -> Cedar Policy Gate.

Zero AWS credentials required. 100% locally runnable.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from src.config import settings
from src.safety_guardian_agent import StrandsSafetyGuardianAgent
from evaluate_risk import evaluate_camera_live

console = Console()


def process_incident_file(
    agent: StrandsSafetyGuardianAgent,
    incident_file: Path,
    output_dir: Path,
) -> List[Dict[str, Any]]:
    """Loads incidents from file and runs them through the Strands Agent + Cedar + Gemini pipeline."""
    with open(incident_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Could be standard incident dict or strands incident batch
    raw_incidents = data.get("incidents", [])
    if not raw_incidents and isinstance(data, list):
        raw_incidents = data

    results = []
    for inc_packet in raw_incidents:
        console.print(f"\n[bold cyan]-> Evaluating Incident with Strands Agent:[/bold cyan] {inc_packet.get('incident_id', 'UNKNOWN')}")
        result = agent.process_incident(inc_packet)
        results.append(result)
        _display_agent_decision(result)

    # Export Agent Decisions JSON
    out_decisions_dir = output_dir / "agent_decisions"
    out_decisions_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_decisions_dir / f"{incident_file.stem}_decisions.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump({
            "schema_version": "kineticguard.strands.v1",
            "source_file": incident_file.name,
            "total_decisions": len(results),
            "decisions": results,
        }, f, indent=2)

    console.print(f"[green][OK] Saved Agent Decisions to: {out_file}[/green]")
    return results


def _display_agent_decision(result: Dict[str, Any]):
    """Renders a formatted panel summarizing the Strands Agent + Cedar + Gemini evaluation."""
    w = result["worker_identity"]
    decision = result["decision"]
    gemini = result.get("gemini_analysis", {})
    cedar = result.get("policy_result", {})
    bio = result.get("authoritative_biomechanics", {})

    style = "bold red" if decision == "ESCALATE" else ("bold yellow" if decision == "REVIEW/NOTIFY" else "bold green")

    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_column("Field", style="bold white", width=22)
    table.add_column("Value", style="cyan")

    table.add_row("Incident ID", result["incident_id"])
    table.add_row("Operative / Station", f"{w['worker_id']} @ {w['station_id']} ({w['camera_id']})")
    table.add_row("Autonomous Decision", f"[{style}]{decision}[/{style}]")
    table.add_row("Cedar Policy Gate", f"{cedar.get('policy_gate', 'PASSED')} (Allowed: {cedar.get('allowed')})")
    table.add_row("Cumulative Risk", f"{bio.get('cumulative_risk_score')}/100 [{bio.get('risk_level')}]")
    table.add_row("Peak Spinal Comp.", f"{bio.get('peak_spinal_compression_n'):.0f} N (NIOSH AL: 3400 N)")
    table.add_row("Awkward Duty Cycle", f"{bio.get('awkward_duty_cycle_pct')}%")
    table.add_row("Gemini Model", f"{gemini.get('model')} (Source: {gemini.get('source')})")
    table.add_row("Visual Observation", gemini.get("ergonomic_observation", "N/A"))
    table.add_row("Visual Root Cause", gemini.get("root_cause", "N/A"))
    table.add_row("Recommended Action", f"[green]{result['recommended_action']}[/green]")

    panel = Panel(
        table,
        title=f"[bold white]Strands Safety Guardian - {result['incident_id']}[/bold white]",
        border_style="red" if decision == "ESCALATE" else ("yellow" if decision == "REVIEW/NOTIFY" else "green"),
        expand=False,
    )
    console.print(panel)


def main():
    parser = argparse.ArgumentParser(
        description="KineticGuard Phase 4: Strands Agents SDK + Cedar + Gemini Multimodal Safety Guardian"
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--incident",
        type=str,
        default=None,
        help="Path to Phase 3 incident JSON file to evaluate",
    )
    group.add_argument(
        "--all",
        action="store_true",
        help="Evaluate all 3 cameras (camera_01, camera_02, camera_03) through end-to-end live pipeline",
    )
    group.add_argument(
        "--camera",
        type=str,
        default="camera_01",
        help="Specific camera to evaluate (default: camera_01)",
    )

    parser.add_argument(
        "--max-frames",
        type=int,
        default=60,
        help="Limit frames for live video analysis (default: 60)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output",
        help="Directory to save agent decisions (default: output)",
    )

    args = parser.parse_args()

    header = Panel(
        "[bold cyan]KineticGuard Phase 4: Strands Agents SDK Safety Guardian[/bold cyan]\n"
        "[dim]Strands Agents SDK (5 Tools) + Cedar Policy Gate + Gemini Multimodal Vision[/dim]\n"
        "[dim yellow]Deterministic Biomechanics Authoritative | Zero AWS Dependency | Local Execution[/dim yellow]",
        border_style="cyan",
        expand=False,
    )
    console.print(header)

    output_dir = Path(args.output_dir)
    agent = StrandsSafetyGuardianAgent()

    if args.incident:
        inc_file = Path(args.incident)
        if not inc_file.exists():
            console.print(f"[bold red]Incident file not found: {inc_file}[/bold red]")
            sys.exit(1)
        process_incident_file(agent, inc_file, output_dir)
    else:
        # Live multi-camera or single camera execution
        target_cams = settings.CAMERA_IDS if args.all else [args.camera]
        all_results = []

        for cid in target_cams:
            vpath = settings.CAMERA_SOURCES.get(cid, f"data/{cid}.mp4")
            inc_file = output_dir / "incidents" / f"{cid}_incidents.json"

            # If incident file doesn't exist or live requested, generate it directly from live video
            if not inc_file.exists():
                console.print(f"\n[bold cyan]-> Live Video Processing for {cid}...[/bold cyan]")
                evaluate_camera_live(
                    video_path=vpath,
                    camera_id=cid,
                    output_dir=output_dir,
                    window_sec=20.0,
                    incident_threshold=45.0,
                    max_frames=args.max_frames,
                )

            if inc_file.exists():
                res = process_incident_file(agent, inc_file, output_dir)
                all_results.extend(res)

        console.print(f"\n[bold green]All Phase 4 Agent evaluations completed successfully! ({len(all_results)} decisions)[/bold green]")


if __name__ == "__main__":
    main()
