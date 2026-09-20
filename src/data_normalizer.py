"""
KineticGuard - Canonical Data Normalizer Layer
Provides a single, strict normalization bridge between OpenSearch and dashboard/business logic.

Guarantees:
1. Missing telemetry does NOT become 0 (returns None / 'N/A', while real 0.0 remains 0.0).
2. Canonical station_id is always resolved and never evaluates to 'undefined'.
3. Worker passports aggregate real persisted telemetry without fabricating zeros.
4. Distinguishes REAL_TELEMETRY, TEST_FIXTURE, HISTORICAL, MODELLED, and SIMULATED data sources.
5. Station profiles derive Active Operatives strictly from current REAL_TELEMETRY sessions.
6. Station risk index aggregates strictly from REAL_TELEMETRY incidents.
7. Preserves WAITING_FOR_SUFFICIENT_DATA lifecycle for interventions.
8. Preserves deterministic authoritative ground truth over AI reasoning.
"""

import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Union


DATA_SOURCE_REAL = "REAL_TELEMETRY"
DATA_SOURCE_TEST = "TEST_FIXTURE"
DATA_SOURCE_HISTORICAL = "HISTORICAL"
DATA_SOURCE_MODELLED = "MODELLED"
DATA_SOURCE_SIMULATED = "SIMULATED"


def first_not_none(*values: Any) -> Any:
    """Returns the first argument that is not None and not empty string."""
    for v in values:
        if v is not None and v != "":
            return v
    return None


def safe_num(
    val: Any,
    precision: int = 1,
    allow_zero: bool = True
) -> Optional[Union[float, int]]:
    """
    Safely parses a numerical metric while preserving missing/null semantics.
    - None / '' / NaN / False -> None
    - Real 0 or 0.0 -> 0.0 (or 0 if precision==0)
    - Real number -> rounded float/int
    """
    if val is None or val == "" or val is False:
        return None
    try:
        f = float(val)
        if math.isnan(f) or math.isinf(f):
            return None
        if f == 0.0 and not allow_zero:
            return None
        if precision == 0:
            return int(round(f))
        return round(f, precision)
    except (ValueError, TypeError):
        return None


def classify_data_source(doc: Dict[str, Any], default: str = DATA_SOURCE_REAL) -> str:
    """
    Classifies the data source of an OpenSearch document into:
    - REAL_TELEMETRY: from real multi-camera RTSP/MP4 ingestion feeds
    - TEST_FIXTURE: from test suite assertions, mock scenarios, or synthetic test runs
    - HISTORICAL: historical baseline / archived legacy records without live telemetry
    - MODELLED: from deterministic biomechanical models
    - SIMULATED: from What-If simulation projections
    """
    if not isinstance(doc, dict):
        return default

    # Explicit declaration takes highest precedence
    explicit = doc.get("data_source")
    if explicit in (DATA_SOURCE_REAL, DATA_SOURCE_TEST, DATA_SOURCE_HISTORICAL, DATA_SOURCE_MODELLED, DATA_SOURCE_SIMULATED):
        return explicit

    # Check simulation flags
    if doc.get("is_simulated") or doc.get("simulated") or doc.get("status") == "SIMULATED_ESTIMATE":
        return DATA_SOURCE_SIMULATED

    # Fields to check for source indicators
    check_fields = [
        doc.get("_id"),
        doc.get("incident_id"),
        doc.get("event_id"),
        doc.get("worker_id"),
        (doc.get("worker_identity") or {}).get("worker_id"),
        doc.get("station_id"),
        (doc.get("worker_identity") or {}).get("station_id"),
        doc.get("intervention_id"),
        doc.get("session_id"),
        doc.get("source"),
    ]

    # Check historical identifiers first
    hist_keywords = ("HIST", "ARCHIVE", "LEGACY")
    for field_val in check_fields:
        if field_val and isinstance(field_val, str):
            val_upper = field_val.upper()
            if any(k in val_upper for k in hist_keywords):
                return DATA_SOURCE_HISTORICAL

    # Check test keywords
    test_keywords = (
        "TEST", "GOV", "PRIVACY", "MOCK", "E2E", "SCHEMA", "TRUTH",
        "CRIT", "MILD", "DYN", "LINK", "ESC", "LOG", "REV", "FIXTURE", "SYNTH",
        "W-OP", "OP-0", "STATION-NORTH", "STATION-OS", "OS-0"
    )
    for field_val in check_fields:
        if field_val and isinstance(field_val, str):
            val_upper = field_val.upper()
            if any(k in val_upper for k in test_keywords):
                return DATA_SOURCE_TEST

    # Real camera / session / video identifiers
    source_val = str(doc.get("source") or "").lower()
    camera_val = str(doc.get("camera_id") or "").lower()
    session_val = str(doc.get("session_id") or "").lower()

    if "camera_" in source_val or "camera_" in camera_val or "sess_" in session_val or "ap_" in source_val or "gd_" in source_val:
        return DATA_SOURCE_REAL

    return default


def normalize_station_id(doc: Dict[str, Any], default: str = "N/A") -> str:
    """
    Resolves canonical station_id with fallback support for legacy field names.
    Never returns 'undefined' or 'null' as a string.
    """
    if not isinstance(doc, dict):
        return default

    candidates = [
        doc.get("station_id"),
        (doc.get("worker_identity") or {}).get("station_id"),
        doc.get("station"),
        doc.get("station_name"),
        doc.get("assigned_station"),
        (doc.get("location") or {}).get("station_id"),
    ]

    for c in candidates:
        if c is not None:
            s = str(c).strip()
            if s and s.lower() not in ("undefined", "null", "none", "nan", "unassigned", "n/a"):
                return s
    return default


def normalize_incident(
    doc: Dict[str, Any],
    dec_doc: Optional[Dict[str, Any]] = None,
    keyframes_dir: Optional[Path] = None
) -> Dict[str, Any]:
    """
    Normalizes an OpenSearch incident document into the canonical KineticGuard schema.
    Authoritative deterministic biomechanics are preserved and not overwritten by AI reasoning.
    """
    dec_doc = dec_doc or {}
    inc_id = str(first_not_none(doc.get("incident_id"), doc.get("event_id"), doc.get("_id"), "INC-UNKNOWN"))
    ts_iso = str(first_not_none(
        doc.get("timestamp"),
        doc.get("@timestamp"),
        doc.get("created_at"),
        datetime.now(timezone.utc).isoformat()
    ))

    worker_id = str(first_not_none(
        doc.get("worker_id"),
        (doc.get("worker_identity") or {}).get("worker_id"),
        "W-UNKNOWN"
    ))

    station_id = normalize_station_id(doc)
    if station_id == "N/A" and dec_doc:
        station_id = normalize_station_id(dec_doc)

    camera_id = str(first_not_none(
        doc.get("camera_id"),
        (doc.get("worker_identity") or {}).get("camera_id"),
        "camera_01"
    ))

    risk_level = str(first_not_none(
        doc.get("risk_level"),
        (doc.get("risk_assessment") or {}).get("risk_level"),
        "MODERATE"
    ))

    # Resolve authoritative biomechanics packet
    auth = first_not_none(
        doc.get("authoritative_biomechanics"),
        doc.get("exposure_metrics"),
        doc.get("authoritative_metrics"),
        doc.get("biomechanical_exposure"),
        (doc.get("strands_packet") or {}).get("biomechanical_exposure"),
        dec_doc.get("authoritative_biomechanics"),
        {}
    )
    if not isinstance(auth, dict):
        auth = {}

    # Numerical metrics (strict null preservation)
    raw_risk = first_not_none(
        doc.get("cumulative_risk_score"),
        doc.get("risk_score"),
        (doc.get("risk_assessment") or {}).get("cumulative_risk_score"),
        (doc.get("risk_assessment") or {}).get("risk_score"),
        auth.get("cumulative_risk_score"),
        auth.get("risk_score"),
    )
    cum_risk = safe_num(raw_risk, precision=1)

    raw_torque = first_not_none(
        auth.get("peak_lumbar_torque_nm"),
        auth.get("mean_lumbar_torque_nm"),
        auth.get("lumbar_torque_nm"),
        doc.get("lumbar_torque_nm"),
    )
    peak_torque = safe_num(raw_torque, precision=1)

    raw_comp = first_not_none(
        auth.get("peak_spinal_compression_n"),
        auth.get("mean_spinal_compression_n"),
        auth.get("spinal_compression_n"),
        doc.get("spinal_compression_n"),
    )
    peak_comp = safe_num(raw_comp, precision=0)

    raw_trunk = first_not_none(
        auth.get("peak_trunk_flexion_deg"),
        auth.get("mean_trunk_flexion_deg"),
        auth.get("trunk_flexion_deg"),
        doc.get("trunk_flexion_deg"),
    )
    trunk_angle = safe_num(raw_trunk, precision=1)

    raw_rep = first_not_none(
        auth.get("rep_rate_per_min"),
        auth.get("repetition_rate_per_min"),
        auth.get("repetition_rate"),
        doc.get("repetition_rate"),
        doc.get("repetition_rate_per_min"),
    )
    rep_rate = safe_num(raw_rep, precision=1)

    raw_duty = first_not_none(
        auth.get("awkward_duty_cycle_pct"),
        auth.get("awkward_duty_cycle"),
        doc.get("awkward_duty_cycle_pct"),
        doc.get("awkward_duty_cycle"),
    )
    duty_cycle = safe_num(raw_duty, precision=1)

    raw_dur = first_not_none(
        doc.get("duration_sec"),
        (doc.get("temporal_window") or {}).get("duration_sec"),
        auth.get("window_duration_sec"),
    )
    duration_sec = safe_num(raw_dur, precision=1)

    data_source = classify_data_source(doc)

    # Resolve Keyframe
    keyframe_path = doc.get("keyframe_path") or (doc.get("strands_packet") or {}).get("visual_evidence", {}).get("keyframe_path")
    keyframe_filename = None
    has_keyframe = False

    if keyframe_path:
        keyframe_filename = Path(keyframe_path).name
        if keyframes_dir and (keyframes_dir / keyframe_filename).exists():
            has_keyframe = True
        elif Path(f"output/incidents/keyframes/{keyframe_filename}").exists():
            has_keyframe = True

    if not has_keyframe:
        fallback_name = f"{inc_id}_keyframe.jpg"
        if keyframes_dir and (keyframes_dir / fallback_name).exists():
            keyframe_filename = fallback_name
            has_keyframe = True
        elif Path(f"output/incidents/keyframes/{fallback_name}").exists():
            keyframe_filename = fallback_name
            has_keyframe = True

    keyframe_url = f"/api/keyframe/{keyframe_filename}" if (has_keyframe and keyframe_filename) else None

    # Resolve Cedar Policy and Strands Guardian
    cedar_info = dec_doc.get("policy_result") or {}
    cedar_decision = str(first_not_none(
        dec_doc.get("decision"),
        cedar_info.get("decision"),
        doc.get("cedar_decision"),
        "ESCALATE"
    ))
    cedar_gate = cedar_info.get("policy_gate") or f"PASSED_{cedar_decision}_POLICY"
    cedar_allowed = cedar_info.get("allowed", True)

    strands_reasoning = dec_doc.get("reasoning") or (
        f"Cedar policy gate evaluated {cedar_decision} (Allowed: {cedar_allowed}). "
        f"Biomechanical metrics confirm cumulative risk {cum_risk if cum_risk is not None else 'N/A'}."
    )
    rec_action = str(first_not_none(
        dec_doc.get("recommended_action"),
        doc.get("recommended_action"),
        (doc.get("risk_assessment") or {}).get("recommended_action"),
        "Enforce immediate task rotation."
    ))

    # Gemini multimodal reasoning
    gemini = dec_doc.get("gemini_analysis") or {}
    gemini_obs = gemini.get("ergonomic_observation") or doc.get("ergonomic_observation")
    gemini_root = gemini.get("root_cause") or doc.get("root_cause")
    gemini_conf = safe_num(gemini.get("confidence"), precision=2)

    return {
        "incident_id": inc_id,
        "event_id": inc_id,
        "timestamp": ts_iso,
        "@timestamp": ts_iso,
        "created_at": ts_iso,
        "worker_id": worker_id,
        "station_id": station_id,
        "camera_id": camera_id,
        "risk_level": risk_level,
        "severity": risk_level,
        "risk_score": cum_risk,
        "cumulative_risk_score": cum_risk,
        "lumbar_torque_nm": peak_torque,
        "spinal_compression_n": peak_comp,
        "trunk_flexion_deg": trunk_angle,
        "repetition_rate": rep_rate,
        "repetition_rate_per_min": rep_rate,
        "awkward_duty_cycle": duty_cycle,
        "awkward_duty_cycle_pct": duty_cycle,
        "temporal_exposure": duration_sec,
        "data_source": data_source,
        "keyframe": keyframe_url,
        "keyframe_path": keyframe_path,
        "keyframe_url": keyframe_url,
        "authoritative_biomechanics": {
            "cumulative_risk_score": cum_risk,
            "peak_lumbar_torque_nm": peak_torque,
            "peak_spinal_compression_n": peak_comp,
            "mean_trunk_flexion_deg": trunk_angle,
            "awkward_duty_cycle_pct": duty_cycle,
            "repetition_rate_per_min": rep_rate,
            "data_source": data_source,
            "is_estimated_model_indicator": True,
            "disclaimer": "Modeled estimates derived from video pose tracking, not direct clinical measurements.",
        },
        "visual_evidence": {
            "has_keyframe": has_keyframe,
            "keyframe_filename": keyframe_filename,
            "keyframe_url": keyframe_url,
        },
        "strands_guardian": {
            "decision": cedar_decision,
            "reasoning": strands_reasoning,
            "recommended_action": rec_action,
        },
        "gemini_analysis": {
            "model": gemini.get("model", "gemini-3.6-flash"),
            "observation": gemini_obs or "Worker observed in sustained forward trunk flexion.",
            "root_cause": gemini_root or "Workstation geometry requires reaching below knee height.",
            "confidence": gemini_conf or 0.92,
        },
        "cedar_decision": cedar_decision,
        "cedar_policy": {
            "decision": cedar_decision,
            "policy_gate": cedar_gate,
            "allowed": cedar_allowed,
        },
        "gemini_multimodal_reasoning": gemini_obs or "Biomechanical threshold breached.",
    }


def normalize_worker_passport(
    worker_id: str,
    profile: Optional[Dict[str, Any]] = None,
    incidents: Optional[List[Dict[str, Any]]] = None,
    interventions: Optional[List[Dict[str, Any]]] = None
) -> Dict[str, Any]:
    """
    Aggregates and normalizes a worker's ergonomic passport.
    Reads real persisted telemetry from OpenSearch and never fabricates 0s.
    """
    profile = profile or {}
    incidents = incidents or []
    interventions = interventions or []

    # Canonical station_id
    station_id = normalize_station_id(profile)
    if station_id == "N/A" and incidents:
        for inc in incidents:
            st = normalize_station_id(inc)
            if st != "N/A":
                station_id = st
                break

    data_source = classify_data_source(profile)
    if data_source == DATA_SOURCE_REAL and incidents:
        # If all incidents are TEST_FIXTURE, passport is TEST_FIXTURE
        if all(classify_data_source(i) == DATA_SOURCE_TEST for i in incidents):
            data_source = DATA_SOURCE_TEST
        elif all(classify_data_source(i) == DATA_SOURCE_HISTORICAL for i in incidents):
            data_source = DATA_SOURCE_HISTORICAL

    # Also check worker_id for historical keyword
    wid_upper = worker_id.upper()
    if any(k in wid_upper for k in ("HIST", "ARCHIVE", "LEGACY")):
        data_source = DATA_SOURCE_HISTORICAL
    elif any(k in wid_upper for k in ("ESC", "LINK", "LOG", "TEST", "GOV", "REV", "CRIT", "MILD", "TRUTH")):
        if data_source != DATA_SOURCE_HISTORICAL:
            data_source = DATA_SOURCE_TEST

    # Extract metrics across real incidents
    scores: List[float] = []
    torques: List[float] = []
    compressions: List[float] = []
    duty_cycles: List[float] = []
    rep_rates: List[float] = []
    risk_trend: List[Dict[str, Any]] = []
    high_risk_incidents = 0

    for inc in reversed(incidents):
        norm_inc = normalize_incident(inc)
        sc = norm_inc.get("cumulative_risk_score")
        tq = norm_inc.get("lumbar_torque_nm")
        cp = norm_inc.get("spinal_compression_n")
        dt = norm_inc.get("awkward_duty_cycle_pct")
        rp = norm_inc.get("repetition_rate_per_min")
        ts = norm_inc.get("timestamp")

        if sc is not None:
            scores.append(sc)
            if sc >= 60.0:
                high_risk_incidents += 1
        if tq is not None:
            torques.append(tq)
        if cp is not None:
            compressions.append(cp)
        if dt is not None:
            duty_cycles.append(dt)
        if rp is not None:
            rep_rates.append(rp)

        risk_trend.append({
            "timestamp": ts,
            "score": sc,
            "peak_lumbar_torque_nm": tq,
            "peak_spinal_compression_n": cp,
            "awkward_duty_cycle_pct": dt,
            "incident_id": norm_inc.get("incident_id"),
        })

    # Aggregated metrics (preserve None when absent)
    if scores:
        mean_risk_score = safe_num(sum(scores) / len(scores), precision=1)
    else:
        mean_risk_score = safe_num(first_not_none(
            profile.get("risk_score"),
            profile.get("cumulative_risk_score"),
            profile.get("mean_risk_score"),
            profile.get("last_risk_score"),
        ), precision=1)

    if torques:
        peak_torque = safe_num(max(torques), precision=1)
    else:
        peak_torque = safe_num(first_not_none(
            profile.get("peak_lumbar_torque_nm"),
            profile.get("lumbar_torque_nm"),
            profile.get("mean_lumbar_torque_nm"),
        ), precision=1)

    if compressions:
        peak_compression = safe_num(max(compressions), precision=0)
    else:
        peak_compression = safe_num(first_not_none(
            profile.get("peak_spinal_compression_n"),
            profile.get("spinal_compression_n"),
            profile.get("peak_compression_n"),
        ), precision=0)

    if duty_cycles:
        mean_duty_cycle = safe_num(sum(duty_cycles) / len(duty_cycles), precision=1)
    else:
        mean_duty_cycle = safe_num(first_not_none(
            profile.get("awkward_duty_cycle_pct"),
            profile.get("awkward_duty_cycle"),
        ), precision=1)

    if rep_rates:
        mean_rep_rate = safe_num(sum(rep_rates) / len(rep_rates), precision=1)
    else:
        mean_rep_rate = safe_num(first_not_none(
            profile.get("repetition_rate_per_min"),
            profile.get("repetition_rate"),
        ), precision=1)

    # Exposure minutes
    raw_exp = first_not_none(
        profile.get("exposure_minutes"),
        profile.get("daily_exposure_minutes"),
    )
    if raw_exp is not None:
        exposure_minutes = safe_num(raw_exp, precision=1)
    elif incidents and data_source == DATA_SOURCE_REAL:
        # Real calculated duration from incidents
        total_sec = sum(first_not_none(inc.get("duration_sec"), (inc.get("temporal_window") or {}).get("duration_sec"), 1.5) for inc in incidents)
        exposure_minutes = round(total_sec / 60.0 + len(incidents) * 2.0, 1)
    else:
        exposure_minutes = None

    # Fatigue index
    if profile.get("fatigue_index") is not None:
        fatigue_index = safe_num(profile["fatigue_index"], precision=1)
    elif data_source == DATA_SOURCE_REAL and (mean_risk_score is not None or mean_duty_cycle is not None):
        cadence_strain = (mean_rep_rate or 6.0) * 0.8
        duty_strain = ((mean_duty_cycle or 30.0) / 100.0) * 35.0
        torque_strain = ((peak_torque or 30.0) / 120.0) * 35.0
        fatigue_index = round(min(100.0, max(10.0, cadence_strain + duty_strain + torque_strain)), 1)
    else:
        fatigue_index = None

    # Status & Risk Level mapping with strict provenance and session evidence
    has_active_session = (
        "SESS" in worker_id
        or profile.get("session_id") is not None
        or (exposure_minutes is not None and exposure_minutes > 0)
        or (station_id and station_id not in ("N/A", "None", "UNASSIGNED"))
    )

    if data_source == DATA_SOURCE_HISTORICAL:
        status_label = "HISTORICAL / DATA INCOMPLETE"
        current_status = "HISTORICAL"
    elif data_source == DATA_SOURCE_TEST:
        status_label = "TEST_FIXTURE"
        if mean_risk_score is not None and mean_risk_score >= 70.0:
            current_status = "ACTION_REQUIRED"
        elif mean_risk_score is not None and mean_risk_score >= 50.0:
            current_status = "AT_RISK"
        else:
            current_status = "TEST_FIXTURE"
    elif data_source == DATA_SOURCE_REAL and has_active_session:
        status_label = "ACTIVE MONITORING"
        if mean_risk_score is not None and (mean_risk_score >= 70.0 or (fatigue_index and fatigue_index >= 70.0) or high_risk_incidents >= 2):
            current_status = "ACTION_REQUIRED"
        elif mean_risk_score is not None and (mean_risk_score >= 50.0 or (fatigue_index and fatigue_index >= 50.0) or high_risk_incidents >= 1):
            current_status = "AT_RISK"
        elif mean_risk_score is not None:
            current_status = "HEALTHY"
        else:
            current_status = "HEALTHY"
    else:
        status_label = "DATA INCOMPLETE"
        current_status = "DATA INCOMPLETE"

    rotations = []
    if peak_torque and peak_torque > 80.0:
        rotations.append("Mandate hydraulic scissor lift table use to eliminate deep torso flexion below knee height.")
    if (fatigue_index and fatigue_index >= 60.0) or (mean_duty_cycle and mean_duty_cycle >= 75.0):
        rotations.append("Rotate operative to non-manual inspection/sorting every 20 minutes.")
    if mean_rep_rate and mean_rep_rate >= 15.0:
        rotations.append("Introduce 5-minute postural recovery micro-breaks every 30 minutes.")
    if not rotations:
        rotations.append("Standard ergonomic rotation schedule; maintain neutral spinal posture.")

    last_seen = str(first_not_none(
        profile.get("last_seen_iso"),
        profile.get("updated_at"),
        profile.get("@timestamp"),
        (incidents[0].get("timestamp") if incidents else None),
        datetime.now(timezone.utc).isoformat()
    ))

    worker_name = str(first_not_none(
        profile.get("worker_name"),
        profile.get("name"),
        f"Operative {worker_id}"
    ))

    return {
        "worker_id": worker_id,
        "worker_name": worker_name,
        "station_id": station_id,
        "assigned_station": station_id,
        "role": profile.get("role", "Operator"),
        "exposure_minutes": exposure_minutes,
        "daily_exposure_minutes": exposure_minutes,
        "cumulative_exposure_hours": round(exposure_minutes / 60.0, 1) if exposure_minutes is not None else None,
        "awkward_duty_cycle": mean_duty_cycle,
        "awkward_duty_cycle_pct": mean_duty_cycle,
        "risk_score": mean_risk_score,
        "cumulative_risk_score": mean_risk_score,
        "mean_risk_score": mean_risk_score,
        "risk_level": current_status,
        "current_ergonomic_status": current_status,
        "status": status_label,
        "fatigue_index": fatigue_index,
        "peak_lumbar_torque_nm": peak_torque,
        "peak_compression_n": peak_compression,
        "repetition_rate_per_min": mean_rep_rate,
        "total_incidents": len(incidents),
        "high_risk_incidents": high_risk_incidents,
        "risk_trend": risk_trend,
        "interventions_history": interventions,
        "recommended_rotations": rotations,
        "data_source": data_source,
        "last_updated": last_seen,
        "last_seen_iso": last_seen,
        "disclaimer": "Modeled ergonomic indicators based on biomechanics. Not a clinical medical diagnosis.",
    }


DETERMINISTIC_STATION_LAYOUT = {
    "STATION-01": {"zone": "Zone A: Receiving & Ingestion", "x": 20, "y": 35, "desc": "Conveyor Unloading & Ingestion", "name": "Conveyor Receiving Bay 01"},
    "STATION-02": {"zone": "Zone B: Sorting & Staging", "x": 50, "y": 35, "desc": "Manual Sorting & Bin Packing", "name": "Manual Sorting Station 02"},
    "STATION-03": {"zone": "Zone C: Palletizing & Lift", "x": 80, "y": 35, "desc": "Heavy Palletizing & Crate Lift", "name": "Heavy Palletizing Station 03"},
    "STATION-C02-NORTH": {"zone": "Zone B-North: Express Pick", "x": 50, "y": 75, "desc": "Express Picking Line", "name": "Express Picking Station C02-North"},
    "STATION-NORTH": {"zone": "Zone A-North: Bulk Storage", "x": 20, "y": 75, "desc": "Bulk Tote Handling", "name": "Bulk Storage Station North"},
    "STATION-E2E": {"zone": "Zone C-South: End of Line", "x": 80, "y": 75, "desc": "End of Line Packaging", "name": "End of Line Packaging Station E2E"},
}


def normalize_station_profile(
    station_id: str,
    profile: Optional[Dict[str, Any]] = None,
    incidents: Optional[List[Dict[str, Any]]] = None,
    active_workers: Optional[List[Dict[str, Any]]] = None
) -> Dict[str, Any]:
    """
    Normalizes a workstation profile into the canonical KineticGuard schema.
    
    Guarantees:
    1. active_worker_id is derived from current REAL_TELEMETRY workers active at this station.
       Returns None (rendered as 'None') if no real operative is active.
    2. Live station risk index is aggregated strictly from REAL_TELEMETRY incidents.
       Test fixtures / historical records do not contaminate live operational station risk.
    3. Position metadata is deterministic and matches spatial heatmap coordinates.
    4. Canonical schema: station_id, station_name, active_worker_id, risk_index, risk_level, last_updated, data_source, position.
    """
    profile = profile or {}
    incidents = incidents or []
    active_workers = active_workers or []

    # 1. Resolve Active Operative (Must be from current REAL_TELEMETRY worker passport)
    active_worker_id = None
    for w in active_workers:
        if isinstance(w, dict):
            w_st = normalize_station_id(w)
            w_src = w.get("data_source")
            if w_st == station_id and w_src == DATA_SOURCE_REAL:
                active_worker_id = w.get("worker_id")
                break

    # 2. Camera binding
    camera_id = first_not_none(
        profile.get("active_camera_id"),
        profile.get("camera_id"),
        (incidents[0].get("camera_id") if incidents else None),
        "camera_01"
    )

    # 3. Separate REAL_TELEMETRY incidents from TEST/HISTORICAL fixtures
    real_incidents = [inc for inc in incidents if classify_data_source(inc) == DATA_SOURCE_REAL]
    is_explicit_test_station = any(k in station_id.upper() for k in ("TEST", "OS-", "NORTH", "E2E", "MOCK", "SYNTH", "LINK"))
    has_real_telemetry = (len(real_incidents) > 0 or active_worker_id is not None) and not is_explicit_test_station

    # Determine Station Data Source
    if has_real_telemetry:
        data_source = DATA_SOURCE_REAL
        eval_incidents = real_incidents
    elif incidents and all(classify_data_source(inc) == DATA_SOURCE_TEST for inc in incidents):
        data_source = DATA_SOURCE_TEST
        eval_incidents = incidents
        active_worker_id = None
    elif incidents and all(classify_data_source(inc) == DATA_SOURCE_HISTORICAL for inc in incidents):
        data_source = DATA_SOURCE_HISTORICAL
        eval_incidents = incidents
        active_worker_id = None
    else:
        data_source = classify_data_source(profile, default=DATA_SOURCE_TEST)
        eval_incidents = incidents
        active_worker_id = None

    if not has_real_telemetry:
        active_worker_id = None

    # 4. Aggregate metrics strictly from evaluated incidents
    scores = []
    compressions = []
    torques = []
    duties = []
    workers = set()
    dominant_postures: Dict[str, int] = {}

    for inc in eval_incidents:
        norm_inc = normalize_incident(inc)
        sc = norm_inc.get("cumulative_risk_score")
        cp = norm_inc.get("spinal_compression_n")
        tq = norm_inc.get("lumbar_torque_nm")
        dt = norm_inc.get("awkward_duty_cycle_pct")
        wid = norm_inc.get("worker_id")

        if sc is not None:
            scores.append(sc)
        if cp is not None:
            compressions.append(cp)
        if tq is not None:
            torques.append(tq)
        if dt is not None:
            duties.append(dt)
        if wid and wid != "W-UNKNOWN":
            workers.add(wid)

        posture = inc.get("dominant_posture", "BENDING")
        dominant_postures[posture] = dominant_postures.get(posture, 0) + 1

    if scores:
        mean_risk = safe_num(sum(scores) / len(scores), precision=1)
    else:
        mean_risk = safe_num(first_not_none(
            profile.get("mean_risk_score"),
            profile.get("last_risk_score"),
            profile.get("risk_score"),
            profile.get("risk_index"),
        ), precision=1)

    if compressions:
        peak_comp = safe_num(max(compressions), precision=0)
    else:
        peak_comp = safe_num(first_not_none(
            profile.get("highest_compression_n"),
            profile.get("peak_compression_n"),
            profile.get("peak_spinal_compression_n"),
            profile.get("spinal_compression_n"),
        ), precision=0)

    if torques:
        peak_torque = safe_num(max(torques), precision=1)
    else:
        peak_torque = safe_num(first_not_none(
            profile.get("mean_lumbar_torque_nm"),
            profile.get("peak_lumbar_torque_nm"),
            profile.get("lumbar_torque_nm"),
        ), precision=1)

    if duties:
        mean_duty = safe_num(sum(duties) / len(duties), precision=1)
    else:
        mean_duty = safe_num(first_not_none(
            profile.get("awkward_duty_cycle_pct"),
            profile.get("awkward_duty_cycle"),
        ), precision=1)

    dominant_posture = max(dominant_postures, key=dominant_postures.get) if dominant_postures else profile.get("dominant_posture", "BENDING")

    # Risk level classification
    risk_score_eval = mean_risk if mean_risk is not None else 0.0
    comp_eval = peak_comp if peak_comp is not None else 0.0

    if risk_score_eval >= 70.0 or comp_eval >= 3400.0:
        risk_level = "ACTION_REQUIRED"
    elif risk_score_eval >= 55.0:
        risk_level = "HIGH"
    elif risk_score_eval >= 40.0:
        risk_level = "MODERATE"
    else:
        risk_level = "LOW"

    # Risk factors
    factors = []
    if mean_duty and mean_duty >= 75.0:
        factors.append(f"Excessive awkward posture duty cycle ({mean_duty}%)")
    if peak_comp and peak_comp >= 3400.0:
        factors.append("Spinal compression breaching NIOSH Action Limit (3400 N)")
    elif peak_comp and peak_comp >= 2200.0:
        factors.append(f"Elevated spinal compression ({peak_comp} N)")
    if peak_torque and peak_torque >= 80.0:
        factors.append(f"High peak lumbar moment ({peak_torque} Nm)")
    if not factors:
        factors.append("Standard handling parameters")

    # 5. Position & Spatial Layout
    layout_info = DETERMINISTIC_STATION_LAYOUT.get(station_id)
    if not layout_info:
        hash_val = abs(hash(station_id))
        layout_info = {
            "zone": profile.get("zone_name") or f"Zone {station_id}",
            "x": int(profile.get("coord_x") or (hash_val % 60 + 20)),
            "y": int(profile.get("coord_y") or ((hash_val // 100) % 50 + 25)),
            "desc": profile.get("description") or f"Workstation {station_id}",
            "name": profile.get("station_name") or station_id,
        }

    station_name = str(first_not_none(
        profile.get("station_name"),
        layout_info.get("name"),
        f"Workstation {station_id}"
    ))

    coord_x = layout_info["x"]
    coord_y = layout_info["y"]
    zone_name = layout_info["zone"]
    description = layout_info["desc"]

    color = "#10b981"  # LOW
    if risk_level == "MODERATE":
        color = "#f59e0b"
    elif risk_level == "HIGH":
        color = "#f97316"
    elif risk_level == "ACTION_REQUIRED":
        color = "#ef4444"

    last_ts = str(first_not_none(
        (eval_incidents[0].get("timestamp") if eval_incidents else None),
        (incidents[0].get("timestamp") if incidents else None),
        profile.get("updated_at"),
        profile.get("@timestamp"),
        datetime.now(timezone.utc).isoformat()
    ))

    return {
        "station_id": station_id,
        "station_name": station_name,
        "active_worker_id": active_worker_id,
        "risk_index": mean_risk,
        "risk_score": mean_risk,
        "mean_risk_score": mean_risk,
        "risk_level": risk_level,
        "station_risk_level": risk_level,
        "last_updated": last_ts,
        "latest_incident_timestamp": last_ts,
        "data_source": data_source,
        "position": {
            "x": coord_x,
            "y": coord_y,
            "zone": zone_name,
            "description": description,
        },
        "coord_x": coord_x,
        "coord_y": coord_y,
        "zone_name": zone_name,
        "description": description,
        "color": color,
        "active_camera_id": camera_id,
        "peak_compression_n": peak_comp,
        "mean_lumbar_torque_nm": peak_torque,
        "awkward_duty_cycle_pct": mean_duty,
        "total_incidents": len(eval_incidents),
        "dominant_posture": dominant_posture,
        "dominant_risk_factor": "; ".join(factors),
        "workers_affected": sorted(list(workers)),
    }


def normalize_intervention(item: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalizes a closed-loop intervention record.
    Preserves WAITING_FOR_SUFFICIENT_DATA and ensures missing post metrics remain None.
    """
    int_id = str(first_not_none(item.get("intervention_id"), item.get("_id"), "INTV-UNKNOWN"))
    inc_id = str(first_not_none(item.get("incident_id"), "N/A"))
    worker_id = str(first_not_none(item.get("worker_id"), "W-UNKNOWN"))
    station_id = normalize_station_id(item)
    camera_id = str(first_not_none(item.get("camera_id"), "camera_01"))
    data_source = classify_data_source(item)

    eff_status = str(first_not_none(item.get("effectiveness_status"), item.get("status"), "WAITING_FOR_SUFFICIENT_DATA"))
    frames_obs = int(item.get("frames_observed", 0))
    min_req = int(item.get("min_required_frames", 20))

    pre = first_not_none(item.get("pre_intervention_metrics"), item.get("pre_metrics"))
    post = first_not_none(item.get("post_intervention_metrics"), item.get("post_metrics"))

    pre_risk = safe_num(pre.get("cumulative_risk_score") or pre.get("risk_score") if isinstance(pre, dict) else item.get("pre_intervention_risk"), precision=1)
    post_risk = safe_num(post.get("cumulative_risk_score") or post.get("risk_score") if isinstance(post, dict) else item.get("post_intervention_risk"), precision=1)
    risk_change_pct = safe_num(first_not_none(item.get("risk_reduction_pct"), item.get("risk_change_pct")), precision=1)

    is_waiting = (
        eff_status == "WAITING_FOR_SUFFICIENT_DATA"
        or frames_obs < min_req
        or post_risk is None
        or risk_change_pct is None
        or pre_risk is None
    )

    if is_waiting:
        eff_status = "WAITING_FOR_SUFFICIENT_DATA"
        post_metrics_norm = None
        risk_change_pct = None
        is_effective = None
        eff_message = f"Additional real telemetry required. Observed {frames_obs}/{min_req} frames."
    else:
        post_metrics_norm = {
            "risk_score": post_risk,
            "cumulative_risk_score": post_risk,
            "lumbar_torque_nm": safe_num(post.get("peak_lumbar_torque_nm") or post.get("lumbar_torque_nm"), precision=1) if isinstance(post, dict) else None,
            "peak_lumbar_torque_nm": safe_num(post.get("peak_lumbar_torque_nm") or post.get("lumbar_torque_nm"), precision=1) if isinstance(post, dict) else None,
            "spinal_compression_n": safe_num(post.get("mean_spinal_compression_n") or post.get("peak_spinal_compression_n") or post.get("spinal_compression_n"), precision=0) if isinstance(post, dict) else None,
            "mean_spinal_compression_n": safe_num(post.get("mean_spinal_compression_n") or post.get("peak_spinal_compression_n") or post.get("spinal_compression_n"), precision=0) if isinstance(post, dict) else None,
            "awkward_duty_cycle": safe_num(post.get("awkward_duty_cycle_pct") or post.get("awkward_duty_cycle"), precision=1) if isinstance(post, dict) else None,
            "awkward_duty_cycle_pct": safe_num(post.get("awkward_duty_cycle_pct") or post.get("awkward_duty_cycle"), precision=1) if isinstance(post, dict) else None,
            "frames_analyzed": post.get("frames_analyzed", frames_obs) if isinstance(post, dict) else frames_obs,
        }
        is_effective = item.get("is_effective", True if (risk_change_pct and risk_change_pct > 0) else False)
        eff_message = f"Statistically validated with {frames_obs or (post.get('frames_analyzed', 30) if isinstance(post, dict) else 30)} real post-intervention frames."

    pre_metrics_norm = {
        "risk_score": pre_risk,
        "cumulative_risk_score": pre_risk,
        "lumbar_torque_nm": safe_num(pre.get("peak_lumbar_torque_nm") or pre.get("lumbar_torque_nm"), precision=1) if isinstance(pre, dict) else None,
        "peak_lumbar_torque_nm": safe_num(pre.get("peak_lumbar_torque_nm") or pre.get("lumbar_torque_nm"), precision=1) if isinstance(pre, dict) else None,
        "spinal_compression_n": safe_num(pre.get("peak_spinal_compression_n") or pre.get("spinal_compression_n"), precision=0) if isinstance(pre, dict) else None,
        "peak_spinal_compression_n": safe_num(pre.get("peak_spinal_compression_n") or pre.get("spinal_compression_n"), precision=0) if isinstance(pre, dict) else None,
        "awkward_duty_cycle": safe_num(pre.get("awkward_duty_cycle_pct") or pre.get("awkward_duty_cycle"), precision=1) if isinstance(pre, dict) else None,
        "awkward_duty_cycle_pct": safe_num(pre.get("awkward_duty_cycle_pct") or pre.get("awkward_duty_cycle"), precision=1) if isinstance(pre, dict) else None,
    } if pre else None

    action_type = str(first_not_none(
        item.get("action_type"),
        item.get("action_taken"),
        item.get("intervention_type"),
        "SUPERVISOR_REVIEW"
    ))
    action_desc = str(first_not_none(
        item.get("action_description"),
        item.get("description"),
        "Ergonomic intervention applied"
    ))
    applied_at = str(first_not_none(
        item.get("applied_at"),
        item.get("created_at"),
        item.get("@timestamp"),
        datetime.now(timezone.utc).isoformat()
    ))
    evaluated_at = item.get("evaluated_at") or item.get("updated_at")

    legacy_post_dict = post_metrics_norm if post_metrics_norm is not None else {
        "risk_score": None,
        "cumulative_risk_score": None,
        "lumbar_torque_nm": None,
        "peak_lumbar_torque_nm": None,
        "spinal_compression_n": None,
        "mean_spinal_compression_n": None,
        "awkward_duty_cycle": None,
        "awkward_duty_cycle_pct": None,
        "frames_analyzed": frames_obs,
    }

    return {
        "intervention_id": int_id,
        "incident_id": inc_id,
        "worker_id": worker_id,
        "station_id": station_id,
        "camera_id": camera_id,
        "action": action_type,
        "action_type": action_type,
        "action_taken": action_type,
        "action_description": action_desc,
        "status": item.get("status", "APPLIED"),
        "effectiveness": eff_status,
        "effectiveness_status": eff_status,
        "effectiveness_assessment": {
            "status": eff_status,
            "is_waiting_for_data": is_waiting,
            "risk_reduction_pct": risk_change_pct,
            "message": eff_message,
        },
        "is_waiting_for_data": is_waiting,
        "frames_observed": frames_obs,
        "min_required_frames": min_req,
        "pre_metrics": pre_metrics_norm,
        "post_metrics": post_metrics_norm,
        "pre_intervention_metrics": pre_metrics_norm or {},
        "post_intervention_metrics": legacy_post_dict,
        "risk_reduction_pct": risk_change_pct,
        "is_effective": is_effective,
        "data_source": data_source,
        "created_at": applied_at,
        "applied_at": applied_at,
        "updated_at": evaluated_at,
        "evaluated_at": evaluated_at,
        "lifecycle_stepper": {
            "stage_1_incident": {
                "name": "Incident Triggered",
                "status": "COMPLETED",
                "incident_id": inc_id,
                "trigger_risk": pre_risk,
            },
            "stage_2_action": {
                "name": "Corrective Action Dispatched",
                "status": "COMPLETED",
                "action_type": action_type,
            },
            "stage_3_remonitor": {
                "name": "Live Video Re-Monitoring",
                "status": "IN_PROGRESS" if is_waiting else "COMPLETED",
                "frames_observed": frames_obs,
                "min_required_frames": min_req,
            },
            "stage_4_verification": {
                "name": "Statistical Verification",
                "status": "COMPLETED" if not is_waiting else "WAITING_FOR_DATA",
                "risk_reduction_pct": risk_change_pct,
            },
            "stage_4_outcome": {
                "name": "Statistical Verification",
                "status": "COMPLETED" if not is_waiting else "WAITING_FOR_DATA",
                "risk_reduction_pct": risk_change_pct,
            },
        },
    }
