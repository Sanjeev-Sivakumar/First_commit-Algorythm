"""
KineticGuard KVS PutMedia Client
Handles streaming MKV fragments to the AWS Kinesis Video Streams PutMedia API
with AWS SigV4 authentication, chunked transfer encoding, and ACK parsing.
"""

import json
import threading
import time
from typing import Callable, Generator, Iterator, Optional
import requests
from urllib.parse import urlparse

from botocore.auth import SigV4Auth
from botocore.awsrequest import AWSRequest
from botocore.credentials import Credentials

from .config import settings
from .logger import log_event, logger


class KVSPutMediaClient:
    """Streams MKV byte chunks to an AWS KVS PutMedia endpoint."""

    def __init__(
        self,
        stream_name: str,
        data_endpoint: str,
        camera_id: str,
        mock_mode: bool = False,
        on_ack: Optional[Callable[[dict], None]] = None,
        on_error: Optional[Callable[[Exception], None]] = None,
    ):
        self.stream_name = stream_name
        self.data_endpoint = data_endpoint.rstrip("/")
        self.camera_id = camera_id
        self.mock_mode = mock_mode
        self.on_ack = on_ack
        self.on_error = on_error
        self.is_connected = False
        self._stop_event = threading.Event()

    def _generate_signed_headers(self) -> dict:
        """Construct AWS SigV4 signed headers for PutMedia."""
        url = f"{self.data_endpoint}/putMedia"
        parsed_url = urlparse(url)
        host = parsed_url.netloc

        session = settings.get_boto3_session()
        creds = session.get_credentials()
        if not creds:
            raise ValueError("No AWS credentials available to sign PutMedia request.")

        frozen_creds = creds.get_frozen_credentials()

        headers = {
            "Host": host,
            "x-amzn-stream-name": self.stream_name,
            "x-amzn-fragment-timecode-type": "RELATIVE",
            "x-amzn-producer-start-timestamp": f"{time.time():.3f}",
            "x-amz-content-sha256": "UNSIGNED-PAYLOAD",
            "Content-Type": "video/x-matroska",
            "User-Agent": "KineticGuard-Phase1-Producer/1.0",
        }

        aws_request = AWSRequest(method="POST", url=url, headers=headers)
        auth = SigV4Auth(frozen_creds, "kinesisvideo", settings.AWS_REGION)
        auth.add_auth(aws_request)

        return dict(aws_request.headers)

    def stream_media(self, chunk_generator: Generator[bytes, None, None]) -> None:
        """
        Send chunked MKV stream to KVS PutMedia endpoint.
        Maintains persistent connection until generator ends or stop requested.
        """
        if self.mock_mode:
            self._stream_simulated(chunk_generator)
            return

        self._stream_live(chunk_generator)

    def _stream_live(self, chunk_generator: Generator[bytes, None, None]) -> None:
        url = f"{self.data_endpoint}/putMedia"
        log_event(
            self.camera_id,
            "STREAM_INIT",
            f"Preparing SigV4-signed connection to {url}...",
        )

        signed_headers = self._generate_signed_headers()
        session = requests.Session()

        try:
            log_event(self.camera_id, "CONNECTED", f"Connecting to stream '{self.stream_name}'...")
            self.is_connected = True

            # We use a wrapper generator so we can abort on _stop_event
            def streaming_body():
                for chunk in chunk_generator:
                    if self._stop_event.is_set():
                        break
                    yield chunk

            response = session.post(
                url,
                data=streaming_body(),
                headers=signed_headers,
                stream=True,
                timeout=(10, None),  # 10s connect timeout, infinite read timeout
            )

            if response.status_code != 200:
                error_body = response.text[:300]
                raise ConnectionError(
                    f"KVS PutMedia returned HTTP {response.status_code}: {error_body}"
                )

            log_event(
                self.camera_id,
                "CONNECTED",
                f"KVS stream established (HTTP 200 OK). Listening for fragment ACKs...",
            )

            # Read and parse ACK events from the response stream
            self._parse_ack_stream(response)

        except Exception as exc:
            self.is_connected = False
            log_event(self.camera_id, "ERROR", f"Streaming error: {exc}", level="error")
            if self.on_error:
                self.on_error(exc)
            raise
        finally:
            self.is_connected = False
            session.close()

    def _parse_ack_stream(self, response: requests.Response) -> None:
        """Reads JSON ACK fragments streamed back by AWS PutMedia."""
        buffer = ""
        for line in response.iter_lines(decode_unicode=True):
            if self._stop_event.is_set():
                break
            if not line:
                continue
            try:
                ack_data = json.loads(line)
                event_type = ack_data.get("EventType", "UNKNOWN")
                frag_num = ack_data.get("FragmentNumber", "N/A")
                timecode = ack_data.get("FragmentTimecode", 0)

                log_event(
                    self.camera_id,
                    "ACK",
                    f"Fragment ACK: {event_type} | Frag: ...{str(frag_num)[-8:]} | Time: {timecode}ms",
                )
                if self.on_ack:
                    self.on_ack(ack_data)

                if event_type == "ERROR":
                    err_code = ack_data.get("ErrorCode")
                    err_id = ack_data.get("ErrorId")
                    log_event(
                        self.camera_id,
                        "ERROR",
                        f"AWS KVS Error ACK: Code {err_code} (Id {err_id})",
                        level="error",
                    )
            except json.JSONDecodeError:
                pass

    def _stream_simulated(self, chunk_generator: Generator[bytes, None, None]) -> None:
        """Simulate KVS ingestion locally when offline or in mock mode."""
        log_event(
            self.camera_id,
            "CONNECTED",
            f"[SIMULATION] Stream '{self.stream_name}' connected to mock endpoint.",
        )
        self.is_connected = True

        fragment_counter = 1000
        bytes_accumulator = 0
        last_ack_time = time.time()

        try:
            for chunk in chunk_generator:
                if self._stop_event.is_set():
                    break

                bytes_accumulator += len(chunk)
                now = time.time()

                # Simulate an ACK every ~2 seconds or ~100KB of video
                if now - last_ack_time >= 2.0 or bytes_accumulator >= 100 * 1024:
                    fragment_counter += 1
                    timecode = int((now - last_ack_time) * 1000)
                    ack_event = {
                        "EventType": "PERSISTED",
                        "FragmentTimecode": timecode,
                        "FragmentNumber": f"MOCK-FRAG-{fragment_counter:06d}",
                        "PayloadSize": bytes_accumulator,
                    }

                    log_event(
                        self.camera_id,
                        "ACK",
                        f"[SIMULATION] Frag ACK: PERSISTED | Frag #{fragment_counter} | {bytes_accumulator/1024:.1f} KB",
                    )

                    if self.on_ack:
                        self.on_ack(ack_event)

                    bytes_accumulator = 0
                    last_ack_time = now

        finally:
            self.is_connected = False
            log_event(
                self.camera_id,
                "STREAM_INIT",
                f"[SIMULATION] Stream '{self.stream_name}' simulation ended.",
            )

    def stop(self) -> None:
        """Signal the stream to terminate."""
        self._stop_event.set()
