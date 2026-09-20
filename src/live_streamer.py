"""
KineticGuard - Real-Time Live Video Streamer & Processing Engine
Manages live video ingestion from Camera feeds (01, 02, 03), Local Webcam, or Uploaded MP4 files.
Processes each frame through:
  Privacy Guard (Always-On Face De-identification)
  -> MediaPipe Pose (33 3D Keypoints)
  -> Biomechanics Analyzer (Trunk angle, L5/S1 torque, spinal compression, posture)
  -> Temporal Sliding Window (Duty cycle, cadence, cumulative risk)
  -> Biomechanical Visualizer (Skeleton overlay, joint callouts, technical CCTV HUD)
"""

import os
import time
import threading
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, Generator
import cv2
import numpy as np

from .config import settings
from .privacy_guard import privacy_guard
from .pose_detector import PoseDetector, PoseDetectionResult
from .biomechanics import BiomechanicsAnalyzer, BiomechanicsMetrics
from .temporal_analyzer import TemporalSlidingWindow, RepetitionCycleDetector
from .cumulative_risk import CumulativeRiskModel
from .visualizer import BiomechanicalVisualizer
from .incident_engine import resolve_station, generate_worker_identity


class LiveStreamManager:
    """Manages real-time video processing and streaming with MediaPipe overlays and telemetry."""

    def __init__(self):
        self.lock = threading.Lock()
        self.running = False
        self.thread: Optional[threading.Thread] = None

        self.current_source_type = "camera_01"  # camera_01, camera_02, camera_03, webcam, uploaded
        self.custom_video_path: Optional[str] = None
        self.uploaded_display_name: str = "uploaded_video.mp4"
        self.cap: Optional[cv2.VideoCapture] = None

        self.pose_detector = PoseDetector()
        self.biomech_analyzer = BiomechanicsAnalyzer(body_weight_kg=75.0)
        self.temporal_window = TemporalSlidingWindow(window_seconds=20.0)
        self.risk_model = CumulativeRiskModel()
        self.visualizer = BiomechanicalVisualizer(camera_id="camera_01")

        self.latest_frame_jpeg: Optional[bytes] = None
        self.latest_telemetry: Dict[str, Any] = self._default_telemetry()
        self.frame_index = 0
        self.last_frame_ts = time.time()
        self.is_webcam = False

        # Start streaming thread
        self.start()

    def _default_telemetry(self) -> Dict[str, Any]:
        return {
            "camera_id": "camera_01",
            "filename": None,
            "station_id": "STATION-01",
            "worker_id": "W-C01-SESS",
            "posture": "UPRIGHT",
            "trunk_angle_deg": 12.0,
            "lumbar_torque_nm": 45.0,
            "spinal_compression_n": 1450.0,
            "reps_per_minute": 0.0,
            "awkward_duty_cycle_pct": 20.0,
            "cumulative_risk_score": 25.0,
            "risk_level": "LOW",
            "pose_detected": False,
            "privacy_guard": "ACTIVE",
            "cedar_decision": "ALLOW",
            "agent_status": "READY",
            "fps": 25.0,
            "frame_index": 0,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "is_incident": False,
        }

    def set_source(self, source_type: str, custom_path: Optional[str] = None, display_name: Optional[str] = None) -> Dict[str, Any]:
        """Switches the active video source dynamically and immediately processes frame 0."""
        with self.lock:
            self.current_source_type = source_type
            if custom_path is not None:
                self.custom_video_path = custom_path
            if display_name is not None:
                self.uploaded_display_name = display_name
            elif self.custom_video_path:
                self.uploaded_display_name = Path(self.custom_video_path).name

            self.frame_index = 0
            self.temporal_window.buffer.clear()
            self.temporal_window.cycle_timestamps.clear()
            self.temporal_window.cycle_detector = RepetitionCycleDetector()

            if source_type == "uploaded":
                camera_id = "uploaded"
                self.visualizer.camera_id = f"UPLOADED / {self.uploaded_display_name.upper()}"
            else:
                camera_id = source_type if source_type in ("camera_01", "camera_02", "camera_03", "webcam") else "camera_01"
                self.visualizer.camera_id = camera_id.upper()

            if self.cap is not None:
                self.cap.release()
                self.cap = None

            self._open_capture()

            # Immediately process the first frame so streaming output is available right away
            if self.cap is not None and self.cap.isOpened():
                ret, frame = self.cap.read()
                if ret and frame is not None:
                    self._process_frame(frame, camera_id)

        return {
            "status": "SUCCESS",
            "source": source_type,
            "filename": self.uploaded_display_name if source_type == "uploaded" else None,
            "path": self.custom_video_path,
            "frame_ready": self.latest_frame_jpeg is not None,
        }

    def _open_capture(self):
        """Initializes VideoCapture for current source."""
        self.is_webcam = False

        if self.current_source_type == "webcam":
            self.cap = cv2.VideoCapture(0)
            if self.cap and self.cap.isOpened():
                self.is_webcam = True
                return
            # Fallback to camera_01 if webcam not accessible
            self.current_source_type = "camera_01"

        if self.current_source_type == "uploaded":
            if self.custom_video_path:
                vpath = Path(self.custom_video_path)
                if vpath.exists():
                    self.cap = cv2.VideoCapture(str(vpath))
                    if self.cap and self.cap.isOpened():
                        return
            # If uploaded video file could not be opened, do not fall back to camera_01
            return

        # Default camera video sources
        cam_key = self.current_source_type if self.current_source_type in settings.CAMERA_SOURCES else "camera_01"
        vpath = Path(settings.CAMERA_SOURCES.get(cam_key, "data/camera_01.mp4"))
        if vpath.exists():
            self.cap = cv2.VideoCapture(str(vpath))

    def start(self):
        """Starts background frame ingestion and processing."""
        if not self.running:
            self.running = True
            self.thread = threading.Thread(target=self._run_loop, daemon=True, name="LiveStreamManager-Worker")
            self.thread.start()

    def stop(self):
        """Stops background thread."""
        self.running = False
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    def _process_frame(self, frame: np.ndarray, camera_id: str) -> bool:
        """Processes a single raw video frame through the full CV + biomechanics + privacy pipeline."""
        if frame is None or frame.size == 0:
            return False

        # Scale high-resolution frames (e.g. 1080p / 4K) to max width 960 for smooth real-time MediaPipe inference
        h, w = frame.shape[:2]
        if w > 960:
            target_w = 960
            target_h = int(h * (960.0 / w))
            frame = cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_AREA)

        self.frame_index += 1
        station_id = "STATION-UPLOAD" if camera_id == "uploaded" else resolve_station(camera_id)
        worker_id = f"W-{camera_id.upper()}-SESS"

        # 1. Privacy Guard: Always-On Face De-identification
        deidentified = privacy_guard.deidentify_frame(frame)

        # 2. MediaPipe Pose Landmark Detection
        pose_res = self.pose_detector.detect_pose(deidentified)

        # 3. Biomechanical Calculations
        biomech = self.biomech_analyzer.analyze(
            angles=pose_res.angles,
            landmarks_3d=pose_res.landmarks_3d,
        )

        # 4. Temporal Sliding Window Aggregation
        frame_data = {
            "timestamp_sec": time.time(),
            "posture": biomech.posture,
            "trunk_flexion_deg": biomech.trunk_flexion_deg,
            "lumbar_torque_nm": biomech.lumbar_torque_nm,
            "compression_force_n": biomech.compression_force_n,
            "risk_score": biomech.risk_score,
        }
        self.temporal_window.add_frame(frame_data)
        temporal_metrics = self.temporal_window.get_metrics()
        risk_eval = self.risk_model.evaluate(temporal_metrics)

        cum_score = float(risk_eval.score)
        duty_cycle = float(temporal_metrics.get("awkward_duty_cycle_pct", 0.0))
        reps_rate = float(temporal_metrics.get("rep_rate_per_min", 0.0))
        risk_lvl = risk_eval.level

        # 5. Render Biomechanical Skeleton Overlay & CCTV HUD directly onto original video frame
        annotated = self.visualizer.draw_frame(
            frame_bgr=deidentified,
            pose_res=pose_res,
            biomech=biomech,
            frame_idx=self.frame_index,
            fps=25.0,
        )

        # Add Privacy Badge to upper-right corner of video
        cv2.putText(
            annotated,
            "[PRIVACY: ACTIVE]",
            (annotated.shape[1] - 170, annotated.shape[0] - 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (0, 255, 255),
            1,
            cv2.LINE_AA,
        )

        # Encode to JPEG
        _, jpeg_buf = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 85])
        self.latest_frame_jpeg = jpeg_buf.tobytes()

        # Update Telemetry Snapshot with sustained cumulative metrics
        is_incident = bool(cum_score >= 60.0 and duty_cycle >= 35.0)
        cedar_dec = "ESCALATE" if cum_score >= 70.0 else ("REVIEW/NOTIFY" if cum_score >= 50.0 else "LOG")

        self.latest_telemetry = {
            "camera_id": str(camera_id),
            "filename": str(self.uploaded_display_name) if str(camera_id) == "uploaded" else None,
            "display_source": f"uploaded / {self.uploaded_display_name}" if str(camera_id) == "uploaded" else str(camera_id),
            "station_id": str(station_id),
            "worker_id": str(worker_id),
            "source_type": str(self.current_source_type),
            "posture": str(biomech.posture),
            "trunk_angle_deg": round(float(biomech.trunk_flexion_deg), 1),
            "lumbar_torque_nm": round(float(biomech.lumbar_torque_nm), 1),
            "spinal_compression_n": round(float(biomech.compression_force_n), 1),
            "reps_per_minute": round(float(reps_rate), 1),
            "awkward_duty_cycle_pct": round(float(duty_cycle), 1),
            "cumulative_risk_score": round(float(cum_score), 1),
            "risk_level": str(risk_lvl),
            "pose_detected": bool(pose_res.detected),
            "privacy_guard": "ACTIVE",
            "cedar_decision": str(cedar_dec),
            "agent_status": "READY" if not is_incident else "ANALYZING",
            "fps": 25.0,
            "frame_index": int(self.frame_index),
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "is_incident": bool(is_incident),
        }
        return True

    def _run_loop(self):
        """Continuous frame processing loop with frame rate regulation."""
        while self.running:
            start_t = time.time()

            try:
                with self.lock:
                    if self.cap is None or not self.cap.isOpened():
                        self._open_capture()

                    if self.cap is None or not self.cap.isOpened():
                        time.sleep(0.1)
                        continue

                    ret, frame = self.cap.read()

                    if not ret or frame is None:
                        if not self.is_webcam:
                            # Loop video file seamlessly
                            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                            ret, frame = self.cap.read()
                        if not ret or frame is None:
                            time.sleep(0.04)
                            continue

                    camera_id = self.current_source_type if self.current_source_type in ("camera_01", "camera_02", "camera_03", "uploaded", "webcam") else "camera_01"
                    self._process_frame(frame, camera_id)

            except Exception as e:
                # Keep thread resilient
                pass

            # Frame rate pacing (~25 FPS = 40ms)
            elapsed = time.time() - start_t
            sleep_t = max(0.005, 0.040 - elapsed)
            time.sleep(sleep_t)

    def get_latest_jpeg(self) -> Optional[bytes]:
        """Returns the most recent annotated JPEG frame bytes."""
        with self.lock:
            return self.latest_frame_jpeg

    def get_latest_telemetry(self) -> Dict[str, Any]:
        """Returns the most recent live biomechanical telemetry."""
        with self.lock:
            return dict(self.latest_telemetry)

    def generate_mjpeg_stream(self) -> Generator[bytes, None, None]:
        """Yields continuous MJPEG boundary chunks."""
        while self.running:
            jpeg = self.get_latest_jpeg()
            if jpeg:
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n"
                    b"Content-Length: " + str(len(jpeg)).encode() + b"\r\n\r\n"
                    + jpeg + b"\r\n"
                )
            time.sleep(0.04)


# Global LiveStreamManager Singleton
live_stream_manager = LiveStreamManager()
