"""
KineticGuard - Shift Safety Reporter (Phase 6)
Aggregates enterprise-wide shift safety data into executive summaries:
- Workers and stations monitored
- Incident counts and high-risk worker tallies
- Top ergonomic risk factors
- Interventions applied and effectiveness rates
- Prioritized administrative and engineering corrective actions

Exports both structured JSON and human-readable Markdown reports.
"""

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, List

from src.dynamo_manager import DynamoDBErgonomicsManager
from src.intervention_manager import InterventionManager


class ShiftReporter:
    """Compiles shift safety intelligence from DynamoDB and Intervention ledgers."""

    def __init__(
        self,
        region_name: str = "us-east-1",
        mock_mode: bool = False,
        output_dir: str = "output/shift_reports",
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.dynamo = DynamoDBErgonomicsManager(region_name=region_name, mock_mode=mock_mode)
        self.interventions = InterventionManager(region_name=region_name, mock_mode=mock_mode)

    def generate_report(self) -> Dict[str, Any]:
        """Generates comprehensive shift safety report."""
        now_iso = datetime.now(timezone.utc).isoformat()
        out_dir = Path(self.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        # Query data
        stations = self.dynamo.list_all_stations()
        incidents = self.dynamo.get_latest_incidents(limit=50)
        all_interventions = self.interventions.list_all(limit=50)

        # Worker list from incidents
        worker_ids = set()
        for inc in incidents:
            if inc.get("worker_id"):
                worker_ids.add(inc["worker_id"])

        # Fetch passports
        passports = []
        high_risk_workers = []
        for wid in worker_ids:
            p = self.dynamo.get_worker_passport(wid)
            if p:
                passports.append(p)
                if p.get("current_ergonomic_status") in ("ACTION_REQUIRED", "AT_RISK") or p.get("fatigue_index", 0) >= 65:
                    high_risk_workers.append(wid)

        # High risk stations
        high_risk_stations = [
            s.get("station_id") for s in stations if s.get("station_risk_level") in ("HIGH", "DANGEROUS", "ACTION_REQUIRED")
        ]

        # Handle Insufficient Data
        if not incidents and not passports:
            report = {
                "shift_id": f"SHIFT-{datetime.now(timezone.utc).strftime('%Y%m%d')}-01",
                "timestamp": now_iso,
                "status": "INSUFFICIENT_DATA",
                "message": "Insufficient runtime data. No worker incidents or passports recorded in the ledger yet.",
                "executive_summary": {
                    "workers_monitored_count": 0,
                    "stations_monitored_count": 0,
                    "total_incidents_recorded": 0,
                    "high_risk_workers": [],
                    "high_risk_stations": [],
                    "shift_ergonomic_safety_score": 100.0,
                },
                "top_ergonomic_risk_factors": ["Insufficient runtime data"],
                "closed_loop_interventions": {
                    "total_interventions_recorded": 0,
                    "effective_interventions": 0,
                    "escalated_interventions": 0,
                    "effectiveness_rate_pct": 0.0,
                },
                "station_risk_breakdown": [],
                "worker_passport_highlights": [],
                "prioritized_corrective_actions": [
                    "Initiate live video surveillance pipeline to capture workstation baseline telemetry."
                ],
            }
            json_path = out_dir / "shift_report_latest.json"
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(report, f, indent=2)
            md_path = out_dir / "shift_report_latest.md"
            self._write_markdown_report(report, md_path)
            return {"json_path": str(json_path), "markdown_path": str(md_path), "report": report}

        # Interventions summary
        total_int = len(all_interventions)
        effective_int = sum(1 for i in all_interventions if i.get("effectiveness_status") == "EFFECTIVE")
        escalated_int = sum(1 for i in all_interventions if i.get("effectiveness_status") in ("INEFFECTIVE", "ESCALATED"))
        eff_rate_pct = round((effective_int / total_int * 100.0), 1) if total_int > 0 else 100.0

        # Top risk factors
        factor_counts = {}
        for inc in incidents:
            vlm = inc.get("vlm_analysis", {})
            for f in vlm.get("risk_factors", []):
                factor_counts[f] = factor_counts.get(f, 0) + 1
            for f in inc.get("contributing_factors", []):
                factor_counts[f] = factor_counts.get(f, 0) + 1
        top_factors = sorted(factor_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        top_factor_strings = [f"{k} ({v} occurrences)" for k, v in top_factors] if top_factors else [
            "No recurring ergonomic hazards recorded"
        ]

        # Dynamic Prioritized recommendations from actual affected stations and workers
        recommendations = []
        for st_id in high_risk_stations:
            recommendations.append(f"Deploy ergonomic intervention (height adjustment / lift assist) at {st_id}.")
        for wk_id in high_risk_workers:
            recommendations.append(f"Enforce mandatory task rotation for operative {wk_id}.")
        if not recommendations:
            recommendations.append("Workplace biomechanical exposure within safe baseline. Maintain continuous surveillance.")
        recommendations.append("Maintain continuous pose surveillance and verify intervention effectiveness via closed-loop feedback.")

        # Overall safety score calculation
        if passports:
            avg_worker_risk = sum(float(p.get("mean_risk_score", 0.0)) for p in passports) / len(passports)
            shift_safety_score = round(max(0.0, 100.0 - avg_worker_risk), 1)
        else:
            shift_safety_score = 100.0

        report = {
            "shift_id": f"SHIFT-{datetime.now(timezone.utc).strftime('%Y%m%d')}-01",
            "timestamp": now_iso,
            "status": "ACTIVE_DATA",
            "executive_summary": {
                "workers_monitored_count": len(worker_ids),
                "stations_monitored_count": len(stations),
                "total_incidents_recorded": len(incidents),
                "high_risk_workers": high_risk_workers,
                "high_risk_stations": high_risk_stations,
                "shift_ergonomic_safety_score": shift_safety_score,
            },
            "top_ergonomic_risk_factors": top_factor_strings,
            "closed_loop_interventions": {
                "total_interventions_recorded": total_int,
                "effective_interventions": effective_int,
                "escalated_interventions": escalated_int,
                "effectiveness_rate_pct": eff_rate_pct,
            },
            "station_risk_breakdown": stations,
            "worker_passport_highlights": passports[:5],
            "prioritized_corrective_actions": recommendations,
        }

        # Save JSON
        json_path = out_dir / "shift_report_latest.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)

        # Save Markdown
        md_path = out_dir / "shift_report_latest.md"
        self._write_markdown_report(report, md_path)

        return {
            "json_path": str(json_path),
            "markdown_path": str(md_path),
            "report": report,
        }

    def _write_markdown_report(self, data: Dict[str, Any], path: Path):
        summary = data["executive_summary"]
        interventions = data["closed_loop_interventions"]

        md = f"""# KineticGuard: Executive Shift Safety Report
**Shift Identifier**: `{data['shift_id']}`  
**Generated At**: `{data['timestamp']}`

---

## 1. Executive Shift Summary
* **Monitored Workstations**: {summary['stations_monitored_count']}
* **Operatives Monitored**: {summary['workers_monitored_count']}
* **Total Ergonomic Incidents Recorded**: {summary['total_incidents_recorded']}
* **High-Risk Operatives Flagged**: {', '.join(summary['high_risk_workers']) or 'None'}
* **High-Risk Workstations**: {', '.join(summary['high_risk_stations']) or 'None'}
* **Shift Safety Health Index**: **{summary['shift_ergonomic_safety_score']}/100**

---

## 2. Top Ergonomic Hazard Factors
"""
        for f in data["top_ergonomic_risk_factors"]:
            md += f"* {f}\n"

        md += f"""
---

## 3. Closed-Loop Intervention Efficacy
* **Total Interventions Administered**: {interventions['total_interventions_recorded']}
* **Confirmed Effective (Risk Reduced >= 20%)**: {interventions['effective_interventions']}
* **Escalated Interventions**: {interventions['escalated_interventions']}
* **Intervention Success Rate**: **{interventions['effectiveness_rate_pct']}%**

---

## 4. Prioritized Ergonomic Corrective Actions
"""
        for r in data["prioritized_corrective_actions"]:
            md += f"1. {r}\n"

        md += "\n---\n*Report generated autonomously by KineticGuard Autonomous Safety Intelligence Platform.*"

        with open(path, "w", encoding="utf-8") as f:
            f.write(md)
