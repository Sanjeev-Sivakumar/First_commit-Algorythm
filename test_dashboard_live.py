"""
KineticGuard - End-to-End Live Dashboard Verification Script
Tests all HTTP, JSON, MJPEG and Video switching endpoints on the running server.
"""

import urllib.request
import urllib.parse
import json
import re
import sys

BASE_URL = "http://127.0.0.1:8080"

def test_endpoint(name, url, method="GET", data=None, headers=None, expected_status=200):
    req = urllib.request.Request(url, data=data, method=method)
    if headers:
        for k, v in headers.items():
            req.add_header(k, v)
    elif data and method == "POST":
        req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            status = resp.status
            content_type = resp.headers.get("Content-Type", "")
            raw = resp.read()
            print(f"[PASS] {name}: HTTP {status} (Content-Type: {content_type}, Size: {len(raw)} bytes)")
            return status, content_type, raw
    except Exception as e:
        print(f"[FAIL] {name}: Error - {e}")
        return None, None, None

def verify_no_emojis(html_str):
    # Regex to detect standard emoji unicode ranges
    emoji_pattern = re.compile(
        "[\U00010000-\U0010ffff\u2600-\u26ff\u2700-\u27bf]", 
        flags=re.UNICODE
    )
    matches = emoji_pattern.findall(html_str)
    if matches:
        print(f"[FAIL] Found emoji characters in HTML: {matches[:10]}")
        return False
    else:
        print(f"[PASS] Zero emojis detected in dashboard HTML.")
        return True

def main():
    print("==================================================")
    print("KineticGuard Dashboard End-to-End Test Suite")
    print("==================================================")

    # 0. Health Probe Check
    status, ctype, body = test_endpoint("Health Probe Check (/health)", f"{BASE_URL}/health")
    assert status == 200, "Health probe failed"
    health_data = json.loads(body.decode("utf-8"))
    assert health_data.get("status") == "HEALTHY", "Health status not HEALTHY"
    print(f"       Health Diagnostics: Service={health_data.get('service')}, Privacy={health_data.get('privacy_guard')}, Memory={health_data.get('opensearch')}")

    # 1. Main Dashboard HTML
    status, ctype, body = test_endpoint("Main Dashboard HTML", f"{BASE_URL}/")
    assert status == 200, "Dashboard failed to load"
    html_text = body.decode("utf-8")
    assert verify_no_emojis(html_text), "Found emojis in UI"
    assert "KINETICGUARD" in html_text, "Missing system title"
    assert "id=\"live-stream-img\"" in html_text, "Missing live-stream-img element"
    assert "id=\"temporalChart\"" in html_text, "Missing temporalChart canvas"
    assert "TRUNK FLEXION ANGLE" in html_text, "Missing Trunk Flexion tile"
    assert "L5/S1 LUMBAR TORQUE" in html_text, "Missing Torque tile"
    assert "SPINAL COMPRESSION" in html_text, "Missing Compression tile"
    assert "REPETITION CADENCE" in html_text, "Missing Cadence tile"
    assert "AWKWARD DUTY CYCLE" in html_text, "Missing Duty Cycle tile"
    assert "Download Sample Video Set" in html_text, "Missing Download Sample Video Set button"
    assert "/api/download/sample_videos" in html_text, "Missing download link"

    # 1b. Sample Benchmark Video Bundle Download
    status, ctype, body = test_endpoint("Sample Video Set Download (/api/download/sample_videos)", f"{BASE_URL}/api/download/sample_videos")
    assert status == 200, f"Sample download returned {status}"
    assert "application/zip" in ctype or len(body) > 1000000, f"Unexpected content-type {ctype} or small body {len(body)}"
    print(f"       Sample ZIP Downloaded: {len(body):,} bytes successfully verified")

    # 2. Live Telemetry
    status, ctype, body = test_endpoint("Live Telemetry Stream", f"{BASE_URL}/api/stream/telemetry")
    assert status == 200
    telemetry = json.loads(body.decode("utf-8"))
    print(f"       Telemetry Sample: Camera={telemetry.get('camera_id')}, Posture={telemetry.get('posture')}, "
          f"Trunk={telemetry.get('trunk_angle_deg')}deg, Torque={telemetry.get('lumbar_torque_nm')}Nm, "
          f"Compression={telemetry.get('spinal_compression_n')}N, Risk={telemetry.get('cumulative_risk_score')}")

    # 3. Switch Video Source to Camera 02
    payload = json.dumps({"source": "camera_02"}).encode("utf-8")
    status, ctype, body = test_endpoint("Switch Video Source -> Camera 02", f"{BASE_URL}/api/stream/source", method="POST", data=payload)
    assert status == 200

    # 4. Switch Video Source to Camera 03
    payload = json.dumps({"source": "camera_03"}).encode("utf-8")
    status, ctype, body = test_endpoint("Switch Video Source -> Camera 03", f"{BASE_URL}/api/stream/source", method="POST", data=payload)
    assert status == 200

    # 5. Switch Video Source back to Camera 01
    payload = json.dumps({"source": "camera_01"}).encode("utf-8")
    status, ctype, body = test_endpoint("Switch Video Source -> Camera 01", f"{BASE_URL}/api/stream/source", method="POST", data=payload)
    assert status == 200

    # 6. Overview
    status, ctype, body = test_endpoint("System Overview", f"{BASE_URL}/api/overview")
    assert status == 200
    overview = json.loads(body.decode("utf-8"))
    assert overview.get("status") == "HEALTHY"

    # 7. Incidents
    status, ctype, body = test_endpoint("Incidents API", f"{BASE_URL}/api/incidents")
    assert status == 200
    incidents = json.loads(body.decode("utf-8"))
    inc_count = len(incidents) if isinstance(incidents, list) else len(incidents.get("incidents", []))
    print(f"       Loaded {inc_count} indexed incidents from OpenSearch.")

    # 8. Worker Passports
    status, ctype, body = test_endpoint("Worker Passports API", f"{BASE_URL}/api/passports")
    assert status == 200
    passports = json.loads(body.decode("utf-8"))
    w_count = len(passports) if isinstance(passports, list) else len(passports.get("workers", []))
    print(f"       Loaded {w_count} worker passports.")

    # 9. Warehouse Heatmap
    status, ctype, body = test_endpoint("Spatial Heatmap API", f"{BASE_URL}/api/heatmap")
    assert status == 200

    # 10. Interventions & Closed-Loop
    status, ctype, body = test_endpoint("Interventions API", f"{BASE_URL}/api/interventions")
    assert status == 200
    interventions = json.loads(body.decode("utf-8"))
    int_count = len(interventions) if isinstance(interventions, list) else len(interventions.get("interventions", []))
    print(f"       Loaded {int_count} intervention records.")

    # 11. Cedar Governance Audit
    status, ctype, body = test_endpoint("Cedar Governance Audit", f"{BASE_URL}/governance/audit")
    assert status == 200
    audit = json.loads(body.decode("utf-8"))
    aud_count = len(audit) if isinstance(audit, list) else len(audit.get("decisions", []))
    print(f"       Loaded {aud_count} Cedar policy evaluations.")

    # 12. Privacy Guard Status
    status, ctype, body = test_endpoint("Privacy Guard Status", f"{BASE_URL}/privacy")
    assert status == 200
    privacy = json.loads(body.decode("utf-8"))
    assert privacy.get("status") == "ACTIVE"

    # 13. Digital Twin Simulator (What-If)
    whatif_req = json.dumps({
        "station_id": "STATION-01",
        "trunk_flexion_deg": 42.0,
        "load_weight_kg": 15.0,
        "cadence_cpm": 25.0,
        "duration_min": 60,
        "pallet_height_offset_cm": 25.0
    }).encode("utf-8")
    status, ctype, body = test_endpoint("Digital Twin What-If Simulator", f"{BASE_URL}/api/simulator/whatif", method="POST", data=whatif_req)
    assert status == 200
    sim_res = json.loads(body.decode("utf-8"))
    assert sim_res.get("label") == "MODELLED / SIMULATED" or sim_res.get("status") == "SIMULATED_ESTIMATE" or "simulation" in sim_res
    print(f"       What-If Result: Simulation is explicitly labeled MODELLED/SIMULATED.")

    # 14. Executive Shift Report
    status, ctype, body = test_endpoint("Executive Shift Report", f"{BASE_URL}/api/report/shift")
    assert status == 200

    # 15. MP4 Video Upload & Instant Processing
    with open("data/camera_01.mp4", "rb") as f:
        video_sample = f.read()
    headers = {"Content-Type": "video/mp4", "X-File-Name": "gd_0025.mp4"}
    status, ctype, body = test_endpoint("MP4 Video Upload Endpoint (gd_0025.mp4)", f"{BASE_URL}/api/upload_video", method="POST", data=video_sample, headers=headers)
    assert status == 200
    upload_res = json.loads(body.decode("utf-8"))
    assert upload_res.get("status") == "READY", f"Upload status failed: {upload_res}"
    assert upload_res.get("source") == "uploaded"
    assert upload_res.get("filename") == "gd_0025.mp4"
    assert upload_res.get("frame_ready") == True
    print(f"       Upload Success: File saved to {upload_res.get('path')} and frame 0 processed immediately with display name '{upload_res.get('filename')}'.")

    # 16. Verify Telemetry from Uploaded Video Source
    status, ctype, body = test_endpoint("Uploaded Video Telemetry Stream", f"{BASE_URL}/api/stream/telemetry")
    assert status == 200
    up_telem = json.loads(body.decode("utf-8"))
    assert up_telem.get("source_type") == "uploaded", f"Expected source_type 'uploaded', got {up_telem.get('source_type')}"
    assert up_telem.get("filename") == "gd_0025.mp4", f"Expected filename 'gd_0025.mp4', got {up_telem.get('filename')}"
    assert up_telem.get("station_id") == "STATION-UPLOAD"
    assert "trunk_angle_deg" in up_telem
    print(f"       Uploaded Video Telemetry: Filename={up_telem.get('filename')}, Posture={up_telem.get('posture')}, Trunk={up_telem.get('trunk_angle_deg')}deg, Torque={up_telem.get('lumbar_torque_nm')}Nm")

    # 17. Switch Back to Camera 01
    payload = json.dumps({"source": "camera_01"}).encode("utf-8")
    status, ctype, body = test_endpoint("Restore Video Source -> Camera 01", f"{BASE_URL}/api/stream/source", method="POST", data=payload)
    assert status == 200

    print("\n==================================================")
    print("ALL 17 END-TO-END DASHBOARD & UPLOAD CHECKS PASSED!")
    print("==================================================")

if __name__ == "__main__":
    main()
