"""
KineticGuard Video Source Module
Handles video file reading with looping, frame pacing, CCTV HUD overlays,
and automatic synthetic surveillance video generation.
"""

import math
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Generator, Optional, Tuple
import cv2
import numpy as np

from .config import settings
from .logger import log_event, logger
from .privacy_guard import privacy_guard


class VideoSource:
    """Reads frames from an MP4 file with automatic looping and frame rate pacing."""

    def __init__(
        self,
        video_path: str,
        camera_id: str,
        target_fps: Optional[int] = None,
        sample_stride: int = 1,
        loop: bool = True,
    ):
        self.video_path = Path(video_path)
        self.camera_id = camera_id
        self.target_fps = target_fps or settings.FPS_TARGET
        self.sample_stride = max(1, sample_stride or settings.SAMPLE_STRIDE)
        self.effective_fps = self.target_fps / self.sample_stride
        self.frame_interval = 1.0 / self.effective_fps
        self.loop = loop
        self.last_frame_time = 0.0

        self.cap: Optional[cv2.VideoCapture] = None
        self.width = 640
        self.height = 480
        self.total_frames = 0
        self.current_frame_idx = 0

        self._init_capture()

    def _init_capture(self) -> None:
        """Initialize or reopen the OpenCV video capture."""
        if not self.video_path.exists():
            raise FileNotFoundError(f"Video file not found: {self.video_path}")

        if self.cap is not None:
            self.cap.release()

        self.cap = cv2.VideoCapture(str(self.video_path))
        if not self.cap.isOpened():
            raise IOError(f"Could not open video file: {self.video_path}")

        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 100
        self.current_frame_idx = 0

    def get_status(self) -> str:
        """Computes stream freshness status: LIVE, STALE, or NO DATA."""
        if self.cap is None or not self.cap.isOpened():
            return "NO DATA"
        if self.last_frame_time == 0.0:
            return "INITIALIZING"
        elapsed = time.time() - self.last_frame_time
        if elapsed <= max(2.0, self.frame_interval * 3.0):
            return "LIVE"
        else:
            return "STALE"

    def get_frame_generator(self) -> Generator[Tuple[np.ndarray, float, int], None, None]:
        """
        Yields (frame, timestamp, frame_number) at the target frame rate.
        Loops seamlessly when reaching the end of the video.
        """
        raw_frame_counter = 0
        frame_num = 0
        next_frame_time = time.time()

        while True:
            if self.cap is None or not self.cap.isOpened():
                self._init_capture()

            ret, frame = self.cap.read()

            if not ret or frame is None:
                if self.loop:
                    # Rewind to start
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ret, frame = self.cap.read()
                    if not ret or frame is None:
                        self._init_capture()
                        ret, frame = self.cap.read()
                        if not ret or frame is None:
                            log_event(self.camera_id, "ERROR", "Failed to read frame after loop rewind.")
                            break
                else:
                    break

            raw_frame_counter += 1
            if (raw_frame_counter - 1) % self.sample_stride != 0:
                continue

            frame_num += 1
            timestamp = time.time()
            self.last_frame_time = timestamp

            # Mandatory Privacy De-identification (ALWAYS ON)
            deidentified_frame = privacy_guard.deidentify_frame(frame)

            # Apply CCTV overlay
            annotated_frame = self._overlay_cctv_hud(deidentified_frame, frame_num)

            # Frame rate pacing
            now = time.time()
            sleep_time = next_frame_time - now
            if sleep_time > 0:
                time.sleep(sleep_time)
            next_frame_time = max(time.time(), next_frame_time + self.frame_interval)

            yield annotated_frame, timestamp, frame_num

    def get_frame_packet_generator(
        self, session_id: str
    ) -> Generator[dict, None, None]:
        """
        Yields structured frame packets suitable for local Build It pipelines
        (Strands Agents SDK, OpenSearch, Cedar, LocalStack).
        """
        for frame, timestamp, frame_num in self.get_frame_generator():
            packet = {
                "session_id": session_id,
                "camera_id": self.camera_id,
                "frame_index": frame_num,
                "timestamp": timestamp,
                "timestamp_iso": datetime.fromtimestamp(timestamp).isoformat(),
                "fps": self.effective_fps,
                "resolution": (self.width, self.height),
                "frame": frame,
                "privacy_guard": "ACTIVE",
                "status": self.get_status(),
            }
            yield packet

    def _overlay_cctv_hud(self, frame: np.ndarray, frame_num: int) -> np.ndarray:
        """Overlays subtle CCTV timestamp and camera indicator with Privacy Guard status."""
        annotated = frame.copy()
        h, w = annotated.shape[:2]

        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        hud_text = f"KG-CCTV [{self.camera_id.upper()}]  {now_str}  FRM:{frame_num:06d}  [PRIVACY: ACTIVE]"

        # Top banner background
        cv2.rectangle(annotated, (10, 10), (w - 10, 36), (20, 20, 20), -1)
        # Red REC indicator dot
        cv2.circle(annotated, (25, 23), 5, (0, 0, 255), -1)
        # Text
        cv2.putText(
            annotated,
            hud_text,
            (40, 26),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (220, 220, 220),
            1,
            cv2.LINE_AA,
        )

        return annotated

    def close(self) -> None:
        """Release video resources."""
        if self.cap is not None:
            self.cap.release()
            self.cap = None


class SyntheticVideoGenerator:
    """Creates realistic simulated surveillance videos for Phase 1 demo testing."""

    @classmethod
    def generate_camera_video(
        cls,
        output_path: str,
        camera_id: str,
        title: str,
        duration_sec: int = 8,
        fps: int = 25,
        width: int = 640,
        height: int = 480,
    ) -> str:
        """Generates a styled surveillance feed with simulated motion."""
        out_file = Path(output_path)
        out_file.parent.mkdir(parents=True, exist_ok=True)

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(out_file), fourcc, fps, (width, height))

        total_frames = duration_sec * fps

        theme_configs = {
            "camera_01": {
                "base_bg": (45, 50, 48),      # Front Entrance (Day / Neutral)
                "accent": (0, 200, 255),       # Amber motion
                "zone_name": "GATE A - ENTRANCE",
                "object_label": "PERSON [ID: 1042]",
            },
            "camera_02": {
                "base_bg": (38, 38, 48),      # Warehouse Corridor (Industrial)
                "accent": (50, 255, 120),      # Neon green vehicle
                "zone_name": "AISLE 4 - INVENTORY",
                "object_label": "FORKLIFT [ID: 021]",
            },
            "camera_03": {
                "base_bg": (30, 25, 25),      # Perimeter Fence (Low light / Night)
                "accent": (255, 180, 50),      # Cyan patrol
                "zone_name": "ZONE NORTH - PERIMETER",
                "object_label": "PATROL [ID: 007]",
            },
        }

        cfg = theme_configs.get(
            camera_id,
            {
                "base_bg": (40, 40, 40),
                "accent": (0, 255, 0),
                "zone_name": title,
                "object_label": "TARGET",
            },
        )

        for i in range(total_frames):
            frame = np.full((height, width, 3), cfg["base_bg"], dtype=np.uint8)

            # Draw background perspective grid
            for y in range(80, height - 40, 60):
                cv2.line(frame, (40, y), (width - 40, y), (55, 55, 60), 1)
            for x in range(60, width - 40, 80):
                cv2.line(frame, (x, 80), (x, height - 40), (50, 50, 55), 1)

            # Draw floor plane
            floor_y = int(height * 0.82)
            cv2.line(frame, (20, floor_y), (width - 20, floor_y), (80, 80, 85), 2)

            # 4-stage posture cycle over the video duration:
            # 0.0 - 0.25: Upright Standing / Walking
            # 0.25 - 0.50: Forward Bending at Waist (Stoop)
            # 0.50 - 0.75: Deep Squatting
            # 0.75 - 1.00: Manual Box Lifting & Carry
            progress = (i % total_frames) / float(total_frames)

            # Base anchor on floor
            worker_x = int(width * 0.48 + 40 * math.sin(progress * 2 * math.pi))
            scale = 1.0

            # Compute articulated joint coordinates based on posture phase
            if progress < 0.25:
                # Stage 1: UPRIGHT
                cycle_t = progress / 0.25
                phase_label = "STAGE 1: UPRIGHT POSTURE"
                trunk_angle = 5.0 + 3.0 * math.sin(cycle_t * math.pi)
                knee_angle = 172.0
                elbow_angle = 150.0
                hip_y = floor_y - int(140 * scale)
                has_box = False

            elif progress < 0.50:
                # Stage 2: BENDING (Stoop flexion > 45 deg)
                cycle_t = (progress - 0.25) / 0.25
                phase_label = "STAGE 2: TRUNK BENDING (HIGH TORQUE)"
                # Interpolate trunk flexion up to 55 degrees
                trunk_angle = 15.0 + 40.0 * math.sin(cycle_t * math.pi)
                knee_angle = 165.0 - 10.0 * math.sin(cycle_t * math.pi)
                elbow_angle = 160.0
                hip_y = floor_y - int(135 * scale)
                has_box = False

            elif progress < 0.75:
                # Stage 3: SQUATTING
                cycle_t = (progress - 0.50) / 0.25
                phase_label = "STAGE 3: DEEP SQUATTING"
                trunk_angle = 20.0 * math.sin(cycle_t * math.pi)
                knee_angle = 170.0 - 85.0 * math.sin(cycle_t * math.pi)  # down to 85 deg
                elbow_angle = 140.0
                # Hips drop downward significantly during squat
                hip_drop = int(60 * math.sin(cycle_t * math.pi) * scale)
                hip_y = floor_y - int(135 * scale) + hip_drop
                has_box = False

            else:
                # Stage 4: LIFTING with Package
                cycle_t = (progress - 0.75) / 0.25
                phase_label = "STAGE 4: MANUAL MATERIAL LIFTING"
                trunk_angle = 35.0 * (1.0 - cycle_t) + 8.0
                knee_angle = 120.0 + 45.0 * cycle_t
                elbow_angle = 80.0 + 20.0 * math.sin(cycle_t * math.pi)
                hip_y = floor_y - int((110 + 30 * cycle_t) * scale)
                has_box = True

            # Draw the articulated human worker figure
            cls._draw_humanoid_figure(
                frame,
                worker_x,
                hip_y,
                floor_y,
                trunk_angle,
                knee_angle,
                elbow_angle,
                has_box=has_box,
                accent_color=cfg["accent"],
            )

            # Target detection bounding box around worker
            bbox_x = worker_x - 70
            bbox_y = max(50, hip_y - 140)
            bbox_w = 140
            bbox_h = (floor_y + 10) - bbox_y

            cv2.rectangle(frame, (bbox_x, bbox_y), (bbox_x + bbox_w, bbox_y + bbox_h), cfg["accent"], 1)
            cv2.rectangle(frame, (bbox_x, bbox_y - 18), (bbox_x + 130, bbox_y), cfg["accent"], -1)
            cv2.putText(
                frame,
                cfg["object_label"],
                (bbox_x + 4, bbox_y - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.35,
                (0, 0, 0),
                1,
                cv2.LINE_AA,
            )

            # Bottom status banner
            cv2.rectangle(frame, (10, height - 32), (width - 10, height - 10), (15, 15, 15), -1)
            status_text = f"{cfg['zone_name']} | {phase_label} | 25.0 FPS"
            cv2.putText(
                frame,
                status_text,
                (20, height - 17),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.36,
                (180, 220, 255),
                1,
                cv2.LINE_AA,
            )

            writer.write(frame)

        writer.release()
        return str(out_file)

    @classmethod
    def _draw_humanoid_figure(
        cls,
        frame: np.ndarray,
        root_x: int,
        hip_y: int,
        floor_y: int,
        trunk_angle_deg: float,
        knee_angle_deg: float,
        elbow_angle_deg: float,
        has_box: bool = False,
        accent_color: Tuple[int, int, int] = (0, 255, 200),
    ) -> None:
        """Renders an anatomically articulated human figure with head, torso, arms, and legs."""
        skin_color = (190, 210, 235)
        shirt_color = (160, 90, 45)      # Safety blue/work shirt
        pants_color = (55, 60, 75)        # Work trousers
        box_color = (30, 160, 240)       # Orange parcel crate

        torso_len = 65
        thigh_len = 48
        shin_len = 48
        upper_arm_len = 36
        forearm_len = 34

        # 1. Hips (root)
        hip_x = root_x
        hip_pt = (hip_x, hip_y)

        # 2. Torso & Shoulders (tilted by trunk_angle)
        rad_trunk = math.radians(trunk_angle_deg)
        shoulder_x = int(hip_x + torso_len * math.sin(rad_trunk))
        shoulder_y = int(hip_y - torso_len * math.cos(rad_trunk))
        shoulder_pt = (shoulder_x, shoulder_y)

        # 3. Head & Helmet
        head_x = int(shoulder_x + 22 * math.sin(rad_trunk))
        head_y = int(shoulder_y - 22 * math.cos(rad_trunk))
        cv2.circle(frame, (head_x, head_y), 13, skin_color, -1)
        # Safety hardhat
        cv2.ellipse(frame, (head_x, head_y - 5), (15, 8), 0, 180, 360, (0, 215, 255), -1)

        # Torso body
        cv2.line(frame, hip_pt, shoulder_pt, shirt_color, 18, cv2.LINE_AA)

        # 4. Arms (Shoulder -> Elbow -> Wrist)
        arm_rad = rad_trunk + math.radians(elbow_angle_deg - 70)
        elbow_x = int(shoulder_x + upper_arm_len * math.sin(arm_rad))
        elbow_y = int(shoulder_y + upper_arm_len * math.cos(arm_rad))
        wrist_x = int(elbow_x + forearm_len * math.sin(rad_trunk + 0.3))
        wrist_y = int(elbow_y + forearm_len * math.cos(rad_trunk + 0.3))

        cv2.line(frame, shoulder_pt, (elbow_x, elbow_y), shirt_color, 9, cv2.LINE_AA)
        cv2.line(frame, (elbow_x, elbow_y), (wrist_x, wrist_y), skin_color, 8, cv2.LINE_AA)
        cv2.circle(frame, (wrist_x, wrist_y), 5, skin_color, -1)

        # Draw lifted box crate if active
        if has_box:
            bx, by = wrist_x - 18, wrist_y - 18
            cv2.rectangle(frame, (bx, by), (bx + 38, by + 36), box_color, -1)
            cv2.rectangle(frame, (bx, by), (bx + 38, by + 36), (20, 20, 20), 2)
            cv2.putText(frame, "10KG", (bx + 4, by + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 1)

        # 5. Legs (Hip -> Knee -> Ankle)
        # Bending knee forward
        knee_bend_rad = math.radians(180.0 - knee_angle_deg)
        knee_x = int(hip_x + thigh_len * math.sin(knee_bend_rad * 0.7))
        knee_y = int(hip_y + thigh_len * math.cos(knee_bend_rad * 0.5))

        ankle_x = int(knee_x - shin_len * math.sin(knee_bend_rad * 0.5))
        ankle_y = floor_y

        cv2.line(frame, hip_pt, (knee_x, knee_y), pants_color, 12, cv2.LINE_AA)
        cv2.line(frame, (knee_x, knee_y), (ankle_x, ankle_y), pants_color, 10, cv2.LINE_AA)
        # Work boot
        cv2.rectangle(frame, (ankle_x - 4, ankle_y - 6), (ankle_x + 18, ankle_y + 2), (25, 25, 25), -1)

    @classmethod
    def ensure_all_sample_videos(cls) -> None:
        """Checks and creates sample videos for all 3 cameras if missing."""
        cams = [
            ("camera_01", settings.CAMERA_SOURCES["camera_01"], "Front Entrance"),
            ("camera_02", settings.CAMERA_SOURCES["camera_02"], "Warehouse Corridor"),
            ("camera_03", settings.CAMERA_SOURCES["camera_03"], "Perimeter Gate"),
        ]

        for cam_id, file_path, label in cams:
            p = Path(file_path)
            if not p.exists():
                logger.info(f"[cyan]Generating simulated video for {cam_id} ({p.name})...[/cyan]")
                cls.generate_camera_video(str(p), cam_id, label)
                logger.info(f"[green][OK] Generated {p.name}[/green]")
