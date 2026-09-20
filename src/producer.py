"""
KineticGuard Multi-Camera Stream Producer
Orchestrates multi-camera video ingestion with real OpenCV video decoding,
configurable FPS/sampling, dynamic runtime session IDs, live status tracking,
and optional AWS Kinesis Video Streams integration.
"""

import queue
import subprocess
import threading
import time
import uuid
from typing import Dict, Generator, List, Optional
import imageio_ffmpeg

from .config import settings
from .kvs_client import KVSPutMediaClient
from .kvs_manager import KVSManager
from .logger import log_event, logger
from .video_source import VideoSource


class CameraStreamProducer(threading.Thread):
    """Worker thread that ingests a single camera feed with real-time metrics and live status."""

    def __init__(
        self,
        camera_id: str,
        video_path: str,
        kvs_manager: Optional[KVSManager] = None,
        mock_mode: bool = False,
        use_kvs: bool = False,
        sample_stride: int = 1,
        target_fps: Optional[int] = None,
        session_id: Optional[str] = None,
        frame_queue: Optional[queue.Queue] = None,
    ):
        super().__init__(name=f"Producer-{camera_id}", daemon=True)
        self.camera_id = camera_id
        self.video_path = video_path
        self.kvs_manager = kvs_manager
        self.mock_mode = mock_mode
        self.use_kvs = use_kvs
        self.sample_stride = max(1, sample_stride)
        self.target_fps = target_fps or settings.FPS_TARGET

        # Dynamic runtime session identity
        self.session_id = session_id or f"sess_{int(time.time())}_{camera_id}"
        self.stream_name = settings.get_stream_name(camera_id)
        self.status = "INITIALIZING"

        # Frame queue for downstream local Build It consumers (Strands Agents SDK, OpenSearch)
        self.frame_queue = frame_queue
        self.latest_packet: Optional[dict] = None

        # Metrics
        self.frames_read = 0
        self.frames_sent = 0
        self.bytes_sent = 0
        self.fragments_persisted = 0
        self.reconnections = 0
        self.start_time = 0.0
        self.last_stat_time = 0.0
        self.last_frame_time = 0.0
        self.current_fps = 0.0

        # State control
        self._stop_event = threading.Event()
        self.kvs_client: Optional[KVSPutMediaClient] = None
        self.video_source: Optional[VideoSource] = None
        self.ffmpeg_proc: Optional[subprocess.Popen] = None

    def run(self) -> None:
        """Main producer loop. Runs local OpenCV decoding by default, or KVS if requested."""
        self.start_time = time.time()
        self.last_stat_time = self.start_time

        if self.use_kvs:
            self._run_kvs_pipeline()
        else:
            self._run_local_pipeline()

    def _run_local_pipeline(self) -> None:
        """
        Executes local OpenCV video decoding without requiring AWS credentials or Kinesis Video Streams.
        Decodes real video frames, samples at configured stride, tracks throughput, and produces frame packets.
        """
        log_event(
            self.camera_id,
            "LOCAL_INGEST_INIT",
            f"Starting local OpenCV video ingestion for session '{self.session_id}' from {self.video_path} (Stride: {self.sample_stride})",
        )

        try:
            self.video_source = VideoSource(
                video_path=self.video_path,
                camera_id=self.camera_id,
                target_fps=self.target_fps,
                sample_stride=self.sample_stride,
                loop=True,
            )
            self.status = "LIVE"
        except Exception as e:
            self.status = "NO DATA"
            log_event(self.camera_id, "ERROR", f"Failed to initialize video source: {e}", level="error")
            return

        try:
            for packet in self.video_source.get_frame_packet_generator(self.session_id):
                if self._stop_event.is_set():
                    break

                self.frames_read += 1
                self.frames_sent += 1
                frame_bytes = packet["frame"].nbytes
                self.bytes_sent += frame_bytes
                self.last_frame_time = packet["timestamp"]
                self.latest_packet = packet
                self.status = "LIVE"

                if self.frame_queue is not None:
                    try:
                        self.frame_queue.put_nowait(packet)
                    except queue.Full:
                        try:
                            self.frame_queue.get_nowait()
                            self.frame_queue.put_nowait(packet)
                        except Exception:
                            pass

                now = time.time()
                elapsed = now - self.last_stat_time
                if elapsed >= 3.0:
                    self.current_fps = self.frames_sent / max(0.001, (now - self.start_time))
                    log_event(
                        self.camera_id,
                        "INGEST",
                        f"[LOCAL LIVE] Session {self.session_id} | {self.current_fps:.1f} FPS | "
                        f"{self.frames_sent} frames read | {self.bytes_sent / (1024*1024):.2f} MB decoded",
                    )
                    self.last_stat_time = now

        except Exception as exc:
            self.status = "ERROR"
            log_event(self.camera_id, "ERROR", f"Local decoding error: {exc}", level="error")
        finally:
            self.status = "STOPPED"
            self._cleanup_resources()

    def _run_kvs_pipeline(self) -> None:
        """KVS pipeline with automated reconnection and exponential backoff."""
        backoff_sec = 2.0
        max_backoff = 30.0

        log_event(
            self.camera_id,
            "STREAM_INIT",
            f"Starting KVS ingestion worker for '{self.stream_name}' from {self.video_path}",
        )

        while not self._stop_event.is_set():
            try:
                self.status = "CONNECTING"

                # 1. Ensure stream is active and discover PUT_MEDIA endpoint
                if self.kvs_manager:
                    stream_info = self.kvs_manager.ensure_stream(self.stream_name)
                    data_endpoint = self.kvs_manager.get_data_endpoint(
                        self.stream_name, api_name="PUT_MEDIA"
                    )
                else:
                    data_endpoint = f"https://s-simulated.{settings.AWS_REGION}.kinesisvideo.amazonaws.com"

                # 2. Initialize video source
                self.video_source = VideoSource(
                    video_path=self.video_path,
                    camera_id=self.camera_id,
                    target_fps=self.target_fps,
                    sample_stride=self.sample_stride,
                    loop=True,
                )

                # 3. Create KVS client
                self.kvs_client = KVSPutMediaClient(
                    stream_name=self.stream_name,
                    data_endpoint=data_endpoint,
                    camera_id=self.camera_id,
                    mock_mode=self.mock_mode,
                    on_ack=self._on_fragment_ack,
                    on_error=self._on_kvs_error,
                )

                self.status = "STREAMING"
                backoff_sec = 2.0

                # 4. Stream media
                media_generator = self._create_mkv_stream_generator()
                self.kvs_client.stream_media(media_generator)

            except Exception as exc:
                if self._stop_event.is_set():
                    break

                self.reconnections += 1
                self.status = "RECONNECTING"
                log_event(
                    self.camera_id,
                    "RECONNECT",
                    f"Connection lost ({exc}). Reconnecting in {backoff_sec:.1f}s (Attempt #{self.reconnections})...",
                    level="warning",
                )

                time.sleep(backoff_sec)
                backoff_sec = min(backoff_sec * 1.8, max_backoff)

            finally:
                self._cleanup_resources()

        self.status = "STOPPED"
        log_event(self.camera_id, "COMPLETED", "Producer thread stopped cleanly.")

    def _create_mkv_stream_generator(self) -> Generator[bytes, None, None]:
        """Pipes raw frames from VideoSource into an FFmpeg H.264/MKV encoder."""
        if self.mock_mode:
            yield from self._simulated_mkv_generator()
            return

        w = self.video_source.width
        h = self.video_source.height
        fps = self.target_fps
        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()

        cmd = [
            ffmpeg_exe,
            "-y",
            "-f", "rawvideo",
            "-vcodec", "rawvideo",
            "-s", f"{w}x{h}",
            "-pix_fmt", "bgr24",
            "-r", str(fps),
            "-i", "-",
            "-c:v", "libx264",
            "-preset", "ultrafast",
            "-tune", "zerolatency",
            "-g", str(fps * 2),
            "-f", "matroska",
            "-",
        ]

        self.ffmpeg_proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            bufsize=64 * 1024,
        )

        feed_error = []

        def feed_frames():
            try:
                for frame, timestamp, frame_idx in self.video_source.get_frame_generator():
                    if self._stop_event.is_set() or self.ffmpeg_proc.poll() is not None:
                        break
                    raw_bytes = frame.tobytes()
                    self.ffmpeg_proc.stdin.write(raw_bytes)
                    self.frames_read += 1
                    self.frames_sent += 1
                    self.last_frame_time = timestamp

                    now = time.time()
                    elapsed = now - self.last_stat_time
                    if elapsed >= 3.0:
                        self.current_fps = self.frames_sent / max(0.001, (now - self.start_time))
                        log_event(
                            self.camera_id,
                            "INGEST",
                            f"Streaming active: {self.current_fps:.1f} FPS | "
                            f"{self.frames_sent} frames | {self.bytes_sent / (1024*1024):.2f} MB uploaded",
                        )
                        self.last_stat_time = now

            except Exception as e:
                feed_error.append(e)
            finally:
                try:
                    if self.ffmpeg_proc and self.ffmpeg_proc.stdin:
                        self.ffmpeg_proc.stdin.close()
                except Exception:
                    pass

        feed_thread = threading.Thread(target=feed_frames, daemon=True)
        feed_thread.start()

        try:
            while not self._stop_event.is_set():
                chunk = self.ffmpeg_proc.stdout.read(16384)
                if not chunk:
                    break
                self.bytes_sent += len(chunk)
                yield chunk
        finally:
            try:
                if feed_thread.is_alive():
                    feed_thread.join(timeout=1.0)
            except Exception:
                pass

    def _simulated_mkv_generator(self) -> Generator[bytes, None, None]:
        """Paces and yields simulated byte chunks matching real-time video."""
        fps = self.target_fps
        chunk_size = 8192

        dummy_header = b"\x1a\x45\xdf\xa3\x9f\x42\x86\x81\x01\x42\xf7\x81\x01" + b"\x00" * 200
        self.bytes_sent += len(dummy_header)
        yield dummy_header

        for frame, timestamp, frame_idx in self.video_source.get_frame_generator():
            if self._stop_event.is_set():
                break

            self.frames_read += 1
            self.frames_sent += 1
            self.last_frame_time = timestamp

            frame_chunk = b"\x00\x00\x00\x01" + bytes([frame_idx % 256]) * (chunk_size // 2)
            self.bytes_sent += len(frame_chunk)

            now = time.time()
            elapsed = now - self.last_stat_time
            if elapsed >= 3.0:
                self.current_fps = self.frames_sent / max(0.001, (now - self.start_time))
                log_event(
                    self.camera_id,
                    "INGEST",
                    f"[SIMULATION] Ingesting: {self.current_fps:.1f} FPS | "
                    f"{self.frames_sent} frames | {self.bytes_sent / (1024*1024):.2f} MB",
                )
                self.last_stat_time = now

            yield frame_chunk

    def _on_fragment_ack(self, ack_data: dict) -> None:
        """Callback on fragment acknowledgment."""
        if ack_data.get("EventType") == "PERSISTED":
            self.fragments_persisted += 1

    def _on_kvs_error(self, error: Exception) -> None:
        """Callback on KVS error."""
        log_event(self.camera_id, "ERROR", f"Stream error notified: {error}", level="error")

    def _cleanup_resources(self) -> None:
        """Close sub-processes and video file handles."""
        if self.video_source:
            self.video_source.close()
            self.video_source = None

        if self.ffmpeg_proc:
            try:
                self.ffmpeg_proc.terminate()
                self.ffmpeg_proc.wait(timeout=1.0)
            except Exception:
                pass
            self.ffmpeg_proc = None

    def stop(self) -> None:
        """Signal worker to stop gracefully."""
        self._stop_event.set()
        if self.kvs_client:
            self.kvs_client.stop()
        self._cleanup_resources()

    def get_metrics(self) -> dict:
        """Return real-time performance and status metrics."""
        # Calculate dynamic freshness: LIVE, STALE, NO DATA, or STOPPED
        if self._stop_event.is_set() or self.status == "STOPPED":
            display_status = "STOPPED"
        elif self.frames_read == 0:
            display_status = "NO DATA"
        elif self.last_frame_time > 0 and (time.time() - self.last_frame_time) > 2.0:
            display_status = "STALE"
        else:
            display_status = "LIVE"

        return {
            "camera_id": self.camera_id,
            "session_id": self.session_id,
            "stream_name": self.stream_name,
            "status": display_status,
            "frames_read": self.frames_read,
            "frames_sent": self.frames_sent,
            "bytes_sent_mb": self.bytes_sent / (1024 * 1024),
            "fragments_persisted": self.fragments_persisted,
            "fps": round(self.current_fps, 1),
            "reconnections": self.reconnections,
            "uptime_sec": int(time.time() - self.start_time) if self.start_time else 0,
            "mode": "KVS" if self.use_kvs else "LOCAL",
        }


class MultiCameraIngestionManager:
    """Manages concurrent ingestion for camera_01, camera_02, camera_03."""

    def __init__(
        self,
        mock_mode: Optional[bool] = None,
        use_kvs: bool = False,
        sample_stride: int = 1,
        target_fps: Optional[int] = None,
        session_id_prefix: Optional[str] = None,
    ):
        self.use_kvs = use_kvs
        self.mock_mode = settings.MOCK_AWS if mock_mode is None else mock_mode
        self.sample_stride = sample_stride
        self.target_fps = target_fps or settings.FPS_TARGET
        self.session_id_prefix = session_id_prefix or f"sess_{int(time.time())}"

        self.kvs_manager: Optional[KVSManager] = None
        if self.use_kvs:
            self.kvs_manager = KVSManager(mock_mode=self.mock_mode)

        self.producers: Dict[str, CameraStreamProducer] = {}
        self._stop_event = threading.Event()

    def initialize_producers(self) -> None:
        """Initialize worker threads for all configured cameras."""
        for camera_id in settings.CAMERA_IDS:
            video_path = settings.CAMERA_SOURCES[camera_id]
            session_id = f"{self.session_id_prefix}_{camera_id}"
            producer = CameraStreamProducer(
                camera_id=camera_id,
                video_path=video_path,
                kvs_manager=self.kvs_manager,
                mock_mode=self.mock_mode,
                use_kvs=self.use_kvs,
                sample_stride=self.sample_stride,
                target_fps=self.target_fps,
                session_id=session_id,
                frame_queue=queue.Queue(maxsize=30),
            )
            self.producers[camera_id] = producer

    def start(self) -> None:
        """Start all camera producer threads concurrently."""
        mode_label = "KVS Ingestion" if self.use_kvs else "Local Build It Ingestion (Zero AWS Required)"
        log_event("", "STREAM_INIT", f"Launching multi-camera {mode_label} pipeline...")
        for camera_id, producer in self.producers.items():
            producer.start()

    def stop(self) -> None:
        """Signal all camera producer threads to stop and wait for termination."""
        self._stop_event.set()
        log_event("", "STREAM_INIT", "Stopping all camera producer threads...")
        for camera_id, producer in self.producers.items():
            producer.stop()

        for camera_id, producer in self.producers.items():
            producer.join(timeout=3.0)
        log_event("", "COMPLETED", "All camera streams shut down successfully.")

    def get_dashboard_metrics(self) -> List[dict]:
        """Return metrics list for all active camera streams."""
        return [p.get_metrics() for p in self.producers.values()]

    def get_latest_frame_packet(self, camera_id: str) -> Optional[dict]:
        """Returns the most recently decoded frame packet for a given camera."""
        producer = self.producers.get(camera_id)
        return producer.latest_packet if producer else None

    def get_all_latest_packets(self) -> Dict[str, dict]:
        """Returns the latest frame packets from all active camera streams."""
        return {
            cam_id: p.latest_packet
            for cam_id, p in self.producers.items()
            if p.latest_packet is not None
        }
