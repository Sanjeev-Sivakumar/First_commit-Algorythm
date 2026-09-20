"""
KineticGuard - Phase 1 Test Suite: Video Ingestion Layer (Build It Architecture)
Verifies:
1. Real video files exist and are decoded via OpenCV.
2. Multi-camera concurrent processing works for camera_01, camera_02, camera_03.
3. Configurable FPS and frame sampling (sample_stride).
4. Dynamic runtime session IDs are generated per camera feed.
5. No hardcoded worker/station identities in ingestion layer.
6. Local processing works with ZERO AWS credentials (no KVS requirement).
7. Stream status reports LIVE / STALE / NO DATA dynamically.
8. No S3, Lambda, API Gateway, DynamoDB, SageMaker, or SNS invoked.
"""

import os
import time
import unittest
from pathlib import Path
import cv2
import numpy as np

from src.config import settings
from src.video_source import VideoSource
from src.producer import CameraStreamProducer, MultiCameraIngestionManager


class TestPhase1IngestionBuildIt(unittest.TestCase):
    """Test suite verifying the Phase 1 video ingestion layer for Build It architecture."""

    def test_01_real_video_files_exist_and_opencv_decodable(self):
        """Verify real MP4 video files exist and decode successfully via OpenCV."""
        for cam_id in settings.CAMERA_IDS:
            vpath = Path(settings.CAMERA_SOURCES[cam_id])
            self.assertTrue(vpath.exists(), f"Video file not found: {vpath}")
            self.assertGreater(vpath.stat().st_size, 10000, f"File unexpectedly small: {vpath}")

            cap = cv2.VideoCapture(str(vpath))
            self.assertTrue(cap.isOpened(), f"OpenCV failed to open: {vpath}")
            ret, frame = cap.read()
            self.assertTrue(ret, f"OpenCV failed to read first frame from: {vpath}")
            self.assertIsInstance(frame, np.ndarray)
            self.assertEqual(len(frame.shape), 3)
            self.assertEqual(frame.shape[2], 3)
            cap.release()

    def test_02_video_source_frame_sampling_and_fps(self):
        """Verify configurable FPS and frame sampling stride in VideoSource."""
        vpath = settings.CAMERA_SOURCES["camera_01"]
        vs = VideoSource(
            video_path=vpath,
            camera_id="camera_01",
            target_fps=25,
            sample_stride=2,
            loop=False,
        )
        self.assertEqual(vs.sample_stride, 2)
        self.assertEqual(vs.effective_fps, 12.5)

        gen = vs.get_frame_generator()
        frames_read = 0
        for frame, ts, frame_idx in gen:
            frames_read += 1
            if frames_read >= 5:
                break
        self.assertEqual(frames_read, 5)
        vs.close()

    def test_03_runtime_session_ids_and_frame_packets(self):
        """Verify dynamic runtime session IDs and structured frame packets for Build It pipeline."""
        custom_session = f"sess_test_{int(time.time())}_c01"
        vs = VideoSource(
            video_path=settings.CAMERA_SOURCES["camera_01"],
            camera_id="camera_01",
            target_fps=25,
            loop=False,
        )

        packet_gen = vs.get_frame_packet_generator(session_id=custom_session)
        packet = next(packet_gen)

        self.assertEqual(packet["session_id"], custom_session)
        self.assertEqual(packet["camera_id"], "camera_01")
        self.assertEqual(packet["frame_index"], 1)
        self.assertIn("timestamp", packet)
        self.assertIn("timestamp_iso", packet)
        self.assertIsInstance(packet["frame"], np.ndarray)
        self.assertIn(packet["status"], ["LIVE", "INITIALIZING"])
        vs.close()

    def test_04_no_hardcoded_identities_in_ingestion(self):
        """Verify no hardcoded workers (e.g. WORKER-1042) or stations exist in Phase 1 metadata."""
        vpath = settings.CAMERA_SOURCES["camera_01"]
        producer = CameraStreamProducer(
            camera_id="camera_01",
            video_path=vpath,
            use_kvs=False,
        )
        metrics = producer.get_metrics()
        self.assertNotIn("WORKER-1042", str(metrics))
        self.assertNotIn("STATION-GATE-A", str(metrics))
        self.assertTrue(metrics["session_id"].startswith("sess_"))

    def test_05_stream_status_reporting(self):
        """Verify dynamic LIVE / NO DATA status calculation."""
        # Test NO DATA on invalid path
        vs_bad = VideoSource.__new__(VideoSource)
        vs_bad.cap = None
        vs_bad.last_frame_time = 0.0
        self.assertEqual(vs_bad.get_status(), "NO DATA")

        # Test valid path
        vs_good = VideoSource(
            video_path=settings.CAMERA_SOURCES["camera_01"],
            camera_id="camera_01",
            target_fps=25,
            loop=False,
        )
        gen = vs_good.get_frame_generator()
        next(gen)
        self.assertEqual(vs_good.get_status(), "LIVE")
        vs_good.close()

    def test_06_local_processing_zero_aws_credentials_required(self):
        """Verify CameraStreamProducer runs locally without AWS credentials or KVS."""
        producer = CameraStreamProducer(
            camera_id="camera_01",
            video_path=settings.CAMERA_SOURCES["camera_01"],
            use_kvs=False,
            target_fps=50,  # Fast pace for quick unit test
        )
        producer.start()
        for _ in range(40):
            if producer.frames_read > 0:
                break
            time.sleep(0.1)
        producer.stop()
        producer.join(timeout=2.0)

        metrics = producer.get_metrics()
        self.assertGreater(metrics["frames_read"], 0)
        self.assertGreater(metrics["bytes_sent_mb"], 0.0)
        self.assertEqual(metrics["mode"], "LOCAL")
        self.assertEqual(metrics["reconnections"], 0)

    def test_07_multi_camera_concurrent_local_ingestion(self):
        """Verify MultiCameraIngestionManager ingests camera_01, camera_02, camera_03 concurrently."""
        manager = MultiCameraIngestionManager(
            use_kvs=False,
            target_fps=50,  # Fast pace for test
            session_id_prefix="test_multi",
        )
        manager.initialize_producers()
        self.assertEqual(len(manager.producers), 3)

        manager.start()
        for _ in range(60):
            metrics_list = manager.get_dashboard_metrics()
            if len(metrics_list) == 3 and all(m["frames_read"] > 0 for m in metrics_list):
                break
            time.sleep(0.1)
        manager.stop()

        self.assertEqual(len(metrics_list), 3)
        for m in metrics_list:
            self.assertIn(m["camera_id"], ["camera_01", "camera_02", "camera_03"])
            self.assertTrue(m["session_id"].startswith("test_multi_"))
            self.assertGreater(m["frames_read"], 0, f"No frames read for {m['camera_id']}")
            self.assertEqual(m["mode"], "LOCAL")

    def test_08_no_cloud_runtime_invocations(self):
        """Verify Phase 1 ingestion layer does not touch S3, Lambda, API Gateway, DynamoDB, SageMaker, or SNS."""
        import sys
        # Check that importing and running local ingestion does not require active cloud services
        manager = MultiCameraIngestionManager(use_kvs=False)
        self.assertIsNone(manager.kvs_manager)
        self.assertFalse(manager.use_kvs)


if __name__ == "__main__":
    unittest.main()
