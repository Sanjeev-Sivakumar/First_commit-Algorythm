"""
KineticGuard - Local AWS Serverless Cloud Emulator (Phase 5)
Provides an in-process and standalone local HTTP server that perfectly emulates
API Gateway, Lambda, DynamoDB, S3, and SNS for 100% offline verification.

Usage:
  python -m src.mock_aws --port 8080
"""

import argparse
import json
import os
import sys
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
from typing import Optional

from src.aws_orchestrator import KineticGuardOrchestrator


class MockAWSHTTPHandler(BaseHTTPRequestHandler):
    """Handles HTTP requests matching AWS API Gateway routing."""

    orchestrator: Optional[KineticGuardOrchestrator] = None

    def log_message(self, format, *args):
        # Silence default stderr spam; keep output clean
        pass

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")
        self.end_headers()

    def do_GET(self):
        self._dispatch("GET")

    def do_POST(self):
        self._dispatch("POST")

    def _dispatch(self, method: str):
        parsed = urlparse(self.path)
        path = parsed.path
        raw_query = parse_qs(parsed.query)
        # Flatten query parameters
        query_params = {k: v[0] for k, v in raw_query.items()}

        body = None
        if method == "POST":
            content_length = int(self.headers.get("Content-Length", 0))
            if content_length > 0:
                body = self.rfile.read(content_length).decode("utf-8")

        response = self.orchestrator.route_request(
            method=method,
            path=path,
            query_params=query_params,
            body=body,
        )

        status_code = response.get("statusCode", 200)
        headers = response.get("headers", {})
        resp_body = response.get("body", "{}").encode("utf-8")

        self.send_response(status_code)
        for h_k, h_v in headers.items():
            self.send_header(h_k, h_v)
        self.end_headers()
        self.wfile.write(resp_body)


class MockAWSCloudServer:
    """Manages local mock AWS cloud server lifecycle."""

    def __init__(self, host: str = "127.0.0.1", port: int = 8080):
        self.host = host
        self.port = port
        self.orchestrator = KineticGuardOrchestrator(mock_mode=True)
        MockAWSHTTPHandler.orchestrator = self.orchestrator
        self.server = None
        self.thread = None

    def start_background(self):
        """Starts the mock server in a background daemon thread."""
        self.server = HTTPServer((self.host, self.port), MockAWSHTTPHandler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        print(f"[bold green][MOCK CLOUD][/bold green] Local AWS Cloud Server running at http://{self.host}:{self.port}")
        return self

    def stop(self):
        """Stops the mock server."""
        if self.server:
            self.server.shutdown()
            self.server.server_close()
            print("[bold yellow][MOCK CLOUD][/bold yellow] Local AWS Cloud Server stopped.")


def main():
    parser = argparse.ArgumentParser(description="KineticGuard Local AWS Cloud Server (Mock Mode)")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host address (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8080, help="Port (default: 8080)")
    args = parser.parse_args()

    orchestrator = KineticGuardOrchestrator(mock_mode=True)
    MockAWSHTTPHandler.orchestrator = orchestrator

    server = HTTPServer((args.host, args.port), MockAWSHTTPHandler)
    print(f"\n===========================================================")
    print(f"   KineticGuard Local AWS Serverless Cloud Server (MOCK)   ")
    print(f"===========================================================")
    print(f"API Gateway URL:  http://{args.host}:{args.port}")
    print(f"DynamoDB State:   output/mock_aws_state/dynamodb/")
    print(f"S3 Artifacts:     output/mock_aws_state/s3/")
    print(f"SNS Safety Log:   output/mock_aws_state/sns_alerts.json")
    print(f"\nEndpoints Active:")
    print(f"  GET  http://{args.host}:{args.port}/health")
    print(f"  POST http://{args.host}:{args.port}/incidents")
    print(f"  GET  http://{args.host}:{args.port}/incidents/latest")
    print(f"  GET  http://{args.host}:{args.port}/incidents?worker_id=WORKER-1042")
    print(f"  GET  http://{args.host}:{args.port}/passport/WORKER-1042")
    print(f"  GET  http://{args.host}:{args.port}/stations")
    print(f"\nPress Ctrl+C to stop.")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down mock server...")
        server.shutdown()
        server.server_close()


if __name__ == "__main__":
    main()
