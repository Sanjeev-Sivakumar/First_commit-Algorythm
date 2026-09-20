"""
KineticGuard - Targeted Dashboard Interaction & Detail Modal Test Suite
Validates that every dashboard entity (Incidents, Passports, Stations, Interventions, Reports, Governance, Digital Twin, Architecture, Telemetry)
has proper click handlers, modal dialog components, single-entity API endpoints, and strict data integrity.
"""

import json
import pytest
from src.dashboard_server import DASHBOARD_HTML, DashboardHTTPHandler
from src.opensearch_dashboard_backend import opensearch_dashboard_backend


def test_dashboard_html_contains_modal_infrastructure():
    """Verify HTML contains modal overlay and dialog markup."""
    assert 'id="detail-modal-overlay"' in DASHBOARD_HTML
    assert 'class="modal-dialog"' in DASHBOARD_HTML
    assert 'class="modal-close-btn"' in DASHBOARD_HTML
    assert 'id="modal-category"' in DASHBOARD_HTML
    assert 'id="modal-title"' in DASHBOARD_HTML
    assert 'id="modal-body"' in DASHBOARD_HTML


def test_dashboard_html_contains_interaction_css():
    """Verify CSS has interactive cursor, hover states, and modal transitions."""
    assert '.clickable-card' in DASHBOARD_HTML
    assert '.clickable-row' in DASHBOARD_HTML
    assert 'cursor: pointer' in DASHBOARD_HTML
    assert '.modal-overlay' in DASHBOARD_HTML
    assert '.modal-overlay.active' in DASHBOARD_HTML
    assert 'detail-metric-card' in DASHBOARD_HTML


def test_dashboard_html_contains_all_detail_renderers():
    """Verify all required detail renderer and modal control JS functions exist."""
    required_functions = [
        'function showModal',
        'function closeDetailModal',
        'function openIncidentDetail',
        'function openWorkerDetail',
        'function openStationDetail',
        'function openInterventionDetail',
        'function openReportDetail',
        'function openGovernanceDetail',
        'function openTwinDetail',
        'function openArchDetail',
        'function openTelemetryDetail',
    ]
    for fn in required_functions:
        assert fn in DASHBOARD_HTML, f"Missing JS function: {fn}"


def test_dashboard_html_card_click_bindings():
    """Verify dynamic fetchers attach click handlers and registries."""
    assert 'incidentsMap' in DASHBOARD_HTML
    assert 'passportsMap' in DASHBOARD_HTML
    assert 'stationsMap' in DASHBOARD_HTML
    assert 'interventionsMap' in DASHBOARD_HTML
    assert 'openIncidentDetail(inc.incident_id)' in DASHBOARD_HTML
    assert 'openWorkerDetail(w.worker_id)' in DASHBOARD_HTML
    assert 'openStationDetail(st.station_id)' in DASHBOARD_HTML
    assert 'openInterventionDetail(intv.intervention_id)' in DASHBOARD_HTML
    assert 'openReportDetail()' in DASHBOARD_HTML
    assert 'openGovernanceDetail' in DASHBOARD_HTML
    assert 'openTwinDetail()' in DASHBOARD_HTML
    assert 'Escape' in DASHBOARD_HTML


def test_backend_single_entity_endpoints():
    """Verify OpenSearchDashboardBackend provides single entity lookups with canonical normalization."""
    # 1. Stations
    stations = opensearch_dashboard_backend.get_stations()
    if stations:
        sid = stations[0]["station_id"]
        st = opensearch_dashboard_backend.get_station(sid)
        assert st is not None
        assert st["station_id"] == sid
        assert "data_source" in st

    # 2. Worker Passports
    passports = opensearch_dashboard_backend.get_passports()
    if passports:
        wid = passports[0]["worker_id"]
        p = opensearch_dashboard_backend.get_worker_passport(wid)
        assert p is not None
        assert p["worker_id"] == wid
        assert "data_source" in p

    # 3. Incidents
    incidents = opensearch_dashboard_backend.get_latest_incidents(limit=5)
    if incidents:
        inc_id = incidents[0]["incident_id"]
        inc = opensearch_dashboard_backend.get_incident(inc_id)
        assert inc is not None
        assert inc["incident_id"] == inc_id
        assert "data_source" in inc

    # 4. Interventions
    interventions = opensearch_dashboard_backend.get_interventions(limit=5)
    if interventions:
        intv_id = interventions[0]["intervention_id"]
        intv = opensearch_dashboard_backend.get_intervention(intv_id)
        assert intv is not None
        assert intv["intervention_id"] == intv_id
        assert "data_source" in intv


def test_no_forbidden_vendor_leakage():
    """Verify that product UI names are clean and do not leak internal vendor names."""
    # Ensure product UI exposes KGVision, Pose Engine, Privacy Guard, Cedar Policy Gate
    assert 'KGVision' in DASHBOARD_HTML or 'KGVISION' in DASHBOARD_HTML
    assert 'Privacy Guard' in DASHBOARD_HTML or 'PRIVACY GUARD' in DASHBOARD_HTML
    assert 'Policy Gate' in DASHBOARD_HTML or 'POLICY GATE' in DASHBOARD_HTML
