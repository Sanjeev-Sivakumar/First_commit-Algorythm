"""
KineticGuard - Cloud Storage Persistence Layer
Manages persistent state across ephemeral container environments (e.g. Google Cloud Run).
Mirrors keyframes, incident telemetry, OpenSearch data, reports, and uploaded videos to Google Cloud Storage (GCS)
when GCS_BUCKET_NAME is configured, while transparently operating 100% locally when unconfigured.
"""

import os
import shutil
import threading
from pathlib import Path
from typing import Any, Optional

try:
    from google.cloud import storage
    HAS_GCS = True
except ImportError:
    HAS_GCS = False


class CloudStorageManager:
    """Manages local storage and optional bidirectional GCS persistence."""

    def __init__(self, bucket_name: Optional[str] = None):
        self.bucket_name = bucket_name or os.getenv("GCS_BUCKET_NAME")
        self.client: Optional[Any] = None
        self.bucket: Optional[Any] = None
        self._enabled = False

        if HAS_GCS and self.bucket_name:
            try:
                self.client = storage.Client()
                self.bucket = self.client.bucket(self.bucket_name)
                self._enabled = True
            except Exception:
                self._enabled = False

    @property
    def is_enabled(self) -> bool:
        return self._enabled

    def sync_to_cloud_async(self, local_path: Path, cloud_prefix: str = ""):
        """Asynchronously uploads a local file to GCS."""
        if not self._enabled or not local_path.exists():
            return
        thread = threading.Thread(
            target=self._upload_file_sync,
            args=(local_path, cloud_prefix),
            daemon=True,
            name="GCS-Sync-Worker",
        )
        thread.start()

    def _upload_file_sync(self, local_path: Path, cloud_prefix: str = ""):
        """Internal synchronous file upload to GCS."""
        try:
            rel_path = local_path.as_posix()
            blob_name = f"{cloud_prefix}/{rel_path}".replace("//", "/").lstrip("/")
            blob = self.bucket.blob(blob_name)
            blob.upload_from_filename(str(local_path))
        except Exception:
            pass

    def sync_from_cloud_on_startup(self, local_dir: Path, cloud_prefix: str = ""):
        """Downloads persistent snapshots from GCS upon cold container startup."""
        if not self._enabled:
            return
        try:
            blobs = self.client.list_blobs(self.bucket_name, prefix=cloud_prefix)
            for blob in blobs:
                dest_path = Path(blob.name)
                dest_path.parent.mkdir(parents=True, exist_ok=True)
                if not dest_path.exists():
                    blob.download_to_filename(str(dest_path))
        except Exception:
            pass


# Global singleton instance
cloud_storage_manager = CloudStorageManager()
