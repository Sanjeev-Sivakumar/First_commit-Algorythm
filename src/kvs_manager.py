"""
KineticGuard KVS Manager
Handles AWS Kinesis Video Streams control plane: stream creation, description,
data endpoint discovery, and resource cleanup.
"""

import time
from typing import Any, Dict, List, Optional
from botocore.exceptions import ClientError, NoCredentialsError

from .config import settings
from .logger import log_event, logger


class KVSManager:
    """Manages AWS Kinesis Video Streams control plane resources."""

    def __init__(self, mock_mode: Optional[bool] = None):
        self.mock_mode = settings.MOCK_AWS if mock_mode is None else mock_mode

        if not self.mock_mode:
            if not settings.has_aws_credentials():
                logger.warning(
                    "[yellow]No valid AWS credentials found. Automatically enabling MOCK_AWS mode for KVSManager.[/yellow]"
                )
                self.mock_mode = True

        if not self.mock_mode:
            try:
                session = settings.get_boto3_session()
                self.client = session.client("kinesisvideo", region_name=settings.AWS_REGION)
            except Exception as exc:
                logger.warning(
                    f"[yellow]Failed to initialize boto3 client ({exc}). Falling back to MOCK mode.[/yellow]"
                )
                self.mock_mode = True
                self.client = None
        else:
            self.client = None

        self._mock_streams: Dict[str, Dict[str, Any]] = {}

    def ensure_stream(
        self,
        stream_name: str,
        retention_hours: Optional[int] = None,
        media_type: str = "video/h264",
    ) -> Dict[str, Any]:
        """
        Check if stream exists; if not, create it and wait until it is ACTIVE.
        """
        retention = retention_hours or settings.KVS_DATA_RETENTION_HOURS

        if self.mock_mode:
            log_event(
                "",
                "STREAM_INIT",
                f"[SIMULATION] Stream '{stream_name}' provisioned (Retention: {retention}h)",
            )
            mock_info = {
                "StreamName": stream_name,
                "StreamARN": f"arn:aws:kinesisvideo:{settings.AWS_REGION}:123456789012:stream/{stream_name}/1700000000000",
                "Status": "ACTIVE",
                "DataRetentionInHours": retention,
                "MediaType": media_type,
            }
            self._mock_streams[stream_name] = mock_info
            return mock_info

        log_event("", "STREAM_INIT", f"Checking AWS KVS stream '{stream_name}'...")

        try:
            desc = self.client.describe_stream(StreamName=stream_name)
            stream_info = desc["StreamDescription"]
            status = stream_info.get("Status", "ACTIVE")
            log_event(
                "",
                "STREAM_INIT",
                f"Stream '{stream_name}' exists with status: [green]{status}[/green]",
            )
            return stream_info

        except ClientError as e:
            error_code = e.response.get("Error", {}).get("Code")
            if error_code != "ResourceNotFoundException":
                log_event("", "ERROR", f"Error describing stream '{stream_name}': {e}", level="error")
                raise

            # Stream does not exist -> create it
            log_event(
                "",
                "STREAM_INIT",
                f"Stream '{stream_name}' not found. Creating stream (Retention: {retention}h)...",
            )
            self.client.create_stream(
                StreamName=stream_name,
                DataRetentionInHours=retention,
                MediaType=media_type,
                Tags={"Project": "KineticGuard", "Phase": "1", "ManagedBy": "KineticGuard"},
            )

            # Wait for stream to become ACTIVE
            return self._wait_for_stream_active(stream_name)

    def _wait_for_stream_active(self, stream_name: str, timeout_sec: int = 60) -> Dict[str, Any]:
        """Poll until stream transitions to ACTIVE."""
        start_time = time.time()
        while time.time() - start_time < timeout_sec:
            desc = self.client.describe_stream(StreamName=stream_name)
            stream_info = desc["StreamDescription"]
            status = stream_info.get("Status")
            if status == "ACTIVE":
                log_event("", "STREAM_INIT", f"Stream '{stream_name}' is now [green]ACTIVE[/green].")
                return stream_info
            elif status in ("CREATING", "UPDATING"):
                time.sleep(2)
            else:
                log_event("", "WARNING", f"Stream '{stream_name}' status: {status}")
                time.sleep(2)

        raise TimeoutError(f"Stream '{stream_name}' did not become ACTIVE within {timeout_sec}s.")

    def get_data_endpoint(self, stream_name: str, api_name: str = "PUT_MEDIA") -> str:
        """
        Get the specific data endpoint for PUT_MEDIA or GET_MEDIA.
        """
        if self.mock_mode:
            return f"https://s-simulated.{settings.AWS_REGION}.kinesisvideo.amazonaws.com"

        try:
            response = self.client.get_data_endpoint(StreamName=stream_name, APIName=api_name)
            endpoint = response["DataEndpoint"]
            return endpoint
        except Exception as exc:
            log_event(
                "",
                "ERROR",
                f"Failed to get {api_name} endpoint for '{stream_name}': {exc}",
                level="error",
            )
            raise

    def describe_stream(self, stream_name: str) -> Optional[Dict[str, Any]]:
        """Fetch full description for a stream."""
        if self.mock_mode:
            return self._mock_streams.get(
                stream_name,
                {
                    "StreamName": stream_name,
                    "StreamARN": f"arn:aws:kinesisvideo:{settings.AWS_REGION}:123456789012:stream/{stream_name}/1700000000000",
                    "Status": "ACTIVE",
                    "DataRetentionInHours": settings.KVS_DATA_RETENTION_HOURS,
                },
            )

        try:
            res = self.client.describe_stream(StreamName=stream_name)
            return res["StreamDescription"]
        except ClientError as e:
            if e.response.get("Error", {}).get("Code") == "ResourceNotFoundException":
                return None
            raise

    def list_kineticguard_streams(self) -> List[Dict[str, Any]]:
        """List all streams matching KineticGuard stream prefix."""
        if self.mock_mode:
            return [
                {
                    "StreamName": settings.get_stream_name(cam_id),
                    "StreamARN": f"arn:aws:kinesisvideo:{settings.AWS_REGION}:123456789012:stream/{settings.get_stream_name(cam_id)}/mock",
                    "Status": "ACTIVE",
                    "DataRetentionInHours": settings.KVS_DATA_RETENTION_HOURS,
                }
                for cam_id in settings.CAMERA_IDS
            ]

        try:
            streams = []
            paginator = self.client.get_paginator("list_streams")
            for page in paginator.paginate():
                for s in page.get("StreamInfoList", []):
                    if s["StreamName"].startswith(settings.KVS_STREAM_PREFIX):
                        streams.append(s)
            return streams
        except Exception as exc:
            log_event("", "ERROR", f"Failed to list streams: {exc}", level="error")
            raise

    def delete_stream(self, stream_name: str) -> bool:
        """Delete a stream to prevent unwanted AWS resource accumulation."""
        if self.mock_mode:
            self._mock_streams.pop(stream_name, None)
            log_event("", "STREAM_INIT", f"[SIMULATION] Stream '{stream_name}' deleted.")
            return True

        try:
            desc = self.describe_stream(stream_name)
            if not desc:
                log_event("", "WARNING", f"Stream '{stream_name}' not found for deletion.")
                return False

            arn = desc["StreamARN"]
            self.client.delete_stream(StreamARN=arn)
            log_event("", "STREAM_INIT", f"Stream '{stream_name}' successfully deleted.")
            return True
        except Exception as exc:
            log_event("", "ERROR", f"Failed to delete stream '{stream_name}': {exc}", level="error")
            raise
