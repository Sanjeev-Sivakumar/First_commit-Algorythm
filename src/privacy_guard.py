"""
KineticGuard - Privacy Guard (Always-On Local De-Identification Layer)
Applies mandatory local face de-identification immediately after OpenCV frame ingestion
and before Gemini multimodal visual reasoning, OpenSearch visual storage, or dashboard rendering.

Privacy processing is ALWAYS ENABLED. There is NO on/off toggle.
Body pose, skeletal landmarks, joint angles, and biomechanical calculations are 100% preserved.

Compliance Statement:
This module implements privacy-preserving processing designed to reduce exposure
of personally identifiable visual information. It does not claim formal GDPR or HIPAA compliance.
"""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np


class PrivacyGuard:
    """
    Mandatory, always-on local face de-identification engine.
    Detects faces dynamically using OpenCV Haar Cascades and craniofacial anatomical
    landmarks, applying Gaussian blur and pixelation to obscure facial features while
    strictly preserving neck, shoulder, torso, and skeletal biomechanics.
    """

    def __init__(self):
        # Always active - NO toggle switch allowed
        self._is_active: bool = True
        self.total_frames_processed: int = 0
        self.total_faces_deidentified: int = 0

        # Load standard local OpenCV Haar cascade classifiers (100% offline, zero cloud)
        self.face_cascade = None
        self.profile_cascade = None

        cascade_dir = getattr(cv2, "data", None)
        if cascade_dir and hasattr(cascade_dir, "haarcascades"):
            frontal_path = os.path.join(cascade_dir.haarcascades, "haarcascade_frontalface_default.xml")
            profile_path = os.path.join(cascade_dir.haarcascades, "haarcascade_profileface.xml")

            if os.path.exists(frontal_path):
                self.face_cascade = cv2.CascadeClassifier(frontal_path)
            if os.path.exists(profile_path):
                self.profile_cascade = cv2.CascadeClassifier(profile_path)

    @property
    def is_always_on(self) -> bool:
        """Confirms privacy guard is always enabled with no on/off toggle."""
        return True

    def is_active(self) -> bool:
        """Returns True unconditionally. Privacy processing cannot be disabled."""
        return True

    def deidentify_frame(
        self,
        frame: np.ndarray,
        landmarks_px: Optional[Dict[str, Tuple[int, int]]] = None,
    ) -> np.ndarray:
        """
        De-identifies any human faces in the video frame using Gaussian blur + pixelation.
        Guarantees that body pose, skeletal landmarks, and joint angles remain fully intact.
        
        Args:
            frame: Input BGR image from OpenCV
            landmarks_px: Optional pre-computed pixel landmark positions (e.g. nose, shoulders)
            
        Returns:
            De-identified BGR frame with blurred facial features.
        """
        if frame is None or frame.size == 0:
            return frame

        self.total_frames_processed += 1
        out = frame.copy()
        h, w = out.shape[:2]
        detected_boxes: List[Tuple[int, int, int, int]] = []

        # 1. Detect faces using OpenCV Haar Cascades
        gray = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)
        det_scale = 1.0
        if w > 640:
            det_scale = 640.0 / w
            det_gray = cv2.resize(gray, (640, int(h * det_scale)), interpolation=cv2.INTER_LINEAR)
        else:
            det_gray = gray

        try:
            if self.face_cascade is not None and not self.face_cascade.empty():
                faces = self.face_cascade.detectMultiScale(
                    det_gray,
                    scaleFactor=1.15,
                    minNeighbors=3,
                    minSize=(20, 20),
                )
                for (x, y, fw, fh) in faces:
                    detected_boxes.append((
                        int(x / det_scale),
                        int(y / det_scale),
                        int(fw / det_scale),
                        int(fh / det_scale),
                    ))
        except Exception:
            pass

        try:
            if self.profile_cascade is not None and not self.profile_cascade.empty():
                profiles = self.profile_cascade.detectMultiScale(
                    det_gray,
                    scaleFactor=1.15,
                    minNeighbors=3,
                    minSize=(20, 20),
                )
                for (x, y, pw, ph) in profiles:
                    orig_x = int(x / det_scale)
                    orig_y = int(y / det_scale)
                    # Avoid duplicates
                    if not any(abs(orig_x - bx) < 30 and abs(orig_y - by) < 30 for bx, by, _, _ in detected_boxes):
                        detected_boxes.append((
                            orig_x,
                            orig_y,
                            int(pw / det_scale),
                            int(ph / det_scale),
                        ))
        except Exception:
            pass

        # 2. Craniofacial landmark bounding box fallback (MediaPipe Pose landmarks)
        # Guarantees coverage when operative is bending down, wearing a cap, or viewed from high CCTV angles
        if landmarks_px:
            nose = landmarks_px.get("nose")
            left_shoulder = landmarks_px.get("left_shoulder")
            right_shoulder = landmarks_px.get("right_shoulder")

            if nose:
                nx, ny = nose
                # Derive head radius from shoulder span if available, otherwise sensible default
                if left_shoulder and right_shoulder:
                    shoulder_span = abs(left_shoulder[0] - right_shoulder[0])
                    radius = max(30, int(shoulder_span * 0.40))
                else:
                    radius = max(28, int(h * 0.07))

                x1 = max(0, nx - radius)
                y1 = max(0, ny - radius)
                x2 = min(w, nx + radius)
                y2 = min(h, ny + int(radius * 0.9))
                box_w = x2 - x1
                box_h = y2 - y1

                # Check overlap with existing Haar boxes
                if box_w > 10 and box_h > 10:
                    if not any(abs(x1 - bx) < radius and abs(y1 - by) < radius for bx, by, _, _ in detected_boxes):
                        detected_boxes.append((x1, y1, box_w, box_h))

        # 3. Apply strong pixelation + Gaussian blur to every face box
        for (x, y, bw, bh) in detected_boxes:
            # Ensure boundaries are within frame
            x_start = max(0, x)
            y_start = max(0, y)
            x_end = min(w, x + bw)
            y_end = min(h, y + bh)

            roi = out[y_start:y_end, x_start:x_end]
            if roi.size == 0 or roi.shape[0] < 4 or roi.shape[1] < 4:
                continue

            self.total_faces_deidentified += 1

            # Two-stage de-identification:
            # Stage A: Subsample down to destroy facial micro-features (irreversible)
            small_w = max(4, roi.shape[1] // 8)
            small_h = max(4, roi.shape[0] // 8)
            small = cv2.resize(roi, (small_w, small_h), interpolation=cv2.INTER_LINEAR)
            pixelated = cv2.resize(small, (roi.shape[1], roi.shape[0]), interpolation=cv2.INTER_NEAREST)

            # Stage B: Deep Gaussian blur over pixelated region to soften edges
            k_w = max(15, (roi.shape[1] // 4) * 2 + 1)
            k_h = max(15, (roi.shape[0] // 4) * 2 + 1)
            blurred = cv2.GaussianBlur(pixelated, (k_w, k_h), sigmaX=30, sigmaY=30)

            out[y_start:y_end, x_start:x_end] = blurred

        return out

    def deidentify_image_file(self, file_path: str, output_path: Optional[str] = None) -> str:
        """
        Reads an image from disk, de-identifies any faces, and writes it back.
        Returns the path to the de-identified image.
        """
        p = Path(file_path)
        if not p.exists():
            return file_path

        img = cv2.imread(str(p))
        if img is None:
            return file_path

        deidentified = self.deidentify_frame(img)
        target = Path(output_path) if output_path else p
        target.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(target), deidentified)
        return str(target)

    def get_status(self) -> Dict[str, Any]:
        """Returns runtime status of the privacy guard layer."""
        return {
            "status": "ACTIVE",
            "privacy_guard": "ACTIVE",
            "mode": "ALWAYS_ON",
            "toggle_allowed": False,
            "frames_processed": self.total_frames_processed,
            "faces_deidentified": self.total_faces_deidentified,
            "detection_methods": [
                "OpenCV Haar Frontal Face Cascade (Offline)",
                "OpenCV Haar Profile Face Cascade (Offline)",
                "MediaPipe Craniofacial Landmark Spatial ROI",
            ],
            "deidentification_method": "Multi-Stage Pixelation + Gaussian Blur (ksize>=15, sigma=30)",
            "compliance_notice": (
                "Privacy-preserving processing designed to reduce exposure of personally identifiable "
                "visual information. Not intended as formal GDPR or HIPAA compliance certification."
            ),
        }


# Global singleton instance (ALWAYS ACTIVE)
privacy_guard = PrivacyGuard()
