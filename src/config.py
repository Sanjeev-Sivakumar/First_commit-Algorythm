"""
KineticGuard Configuration Module
Loads environment variables from .env and provides application settings.
"""

import os
from pathlib import Path
from typing import Dict, List, Optional
from dotenv import load_dotenv

# Base project directory
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

# Ensure data directory exists
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Load .env file
ENV_FILE = BASE_DIR / ".env"
if ENV_FILE.exists():
    load_dotenv(dotenv_path=ENV_FILE, override=False)
else:
    load_dotenv(override=False)


class Settings:
    """Application configuration settings."""

    # AWS Settings
    AWS_REGION: str = os.getenv("AWS_REGION", "us-east-1")
    AWS_ACCESS_KEY_ID: Optional[str] = os.getenv("AWS_ACCESS_KEY_ID")
    AWS_SECRET_ACCESS_KEY: Optional[str] = os.getenv("AWS_SECRET_ACCESS_KEY")
    AWS_SESSION_TOKEN: Optional[str] = os.getenv("AWS_SESSION_TOKEN")

    # Kinesis Video Streams Settings
    KVS_STREAM_PREFIX: str = os.getenv("KVS_STREAM_PREFIX", "kineticguard-")
    KVS_DATA_RETENTION_HOURS: int = int(os.getenv("KVS_DATA_RETENTION_HOURS", "24"))

    # Cameras Configuration
    CAMERA_IDS: List[str] = ["camera_01", "camera_02", "camera_03"]

    CAMERA_SOURCES: Dict[str, str] = {
        "camera_01": os.getenv("CAMERA_01_SOURCE", str(DATA_DIR / "camera_01.mp4")),
        "camera_02": os.getenv("CAMERA_02_SOURCE", str(DATA_DIR / "camera_02.mp4")),
        "camera_03": os.getenv("CAMERA_03_SOURCE", str(DATA_DIR / "camera_03.mp4")),
    }

    # Ingestion Parameters
    FPS_TARGET: int = int(os.getenv("FPS_TARGET", "25"))
    SAMPLE_STRIDE: int = int(os.getenv("SAMPLE_STRIDE", "1"))
    GEMINI_API_KEY: Optional[str] = os.getenv("GEMINI_API_KEY")
    MOCK_AWS: bool = os.getenv("MOCK_AWS", "false").strip().lower() in ("true", "1", "yes")
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO").upper()

    @classmethod
    def get_stream_name(cls, camera_id: str) -> str:
        """Generate standard KVS stream name for a given camera ID."""
        return f"{cls.KVS_STREAM_PREFIX}{camera_id}"

    @classmethod
    def has_aws_credentials(cls) -> bool:
        """
        Check if valid AWS credentials are configured.
        Returns False if credentials are empty or contain placeholder values.
        """
        key = cls.AWS_ACCESS_KEY_ID or ""
        secret = cls.AWS_SECRET_ACCESS_KEY or ""

        placeholders = (
            "your_aws_access_key_id_here",
            "your_aws_secret_access_key_here",
            "your_access_key",
            "your_secret_key",
            "xxx",
            "",
        )

        if not key or not secret:
            return False

        if key.lower() in placeholders or secret.lower() in placeholders:
            return False

        return True

    @classmethod
    def get_boto3_session(cls):
        """
        Return a configured boto3.Session object.
        Uses credentials if explicitly provided, else relies on standard AWS credential chain.
        """
        import boto3

        kwargs = {"region_name": cls.AWS_REGION}

        if cls.has_aws_credentials():
            kwargs["aws_access_key_id"] = cls.AWS_ACCESS_KEY_ID
            kwargs["aws_secret_access_key"] = cls.AWS_SECRET_ACCESS_KEY
            if cls.AWS_SESSION_TOKEN:
                kwargs["aws_session_token"] = cls.AWS_SESSION_TOKEN

        return boto3.Session(**kwargs)


settings = Settings()
