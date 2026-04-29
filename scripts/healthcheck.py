#!/usr/bin/env python3
"""
RAGOps — Service Health Check Script
=====================================
Verifies all four services are reachable and healthy before
running evaluations or CI pipelines.

Usage:
    python scripts/healthcheck.py
    python scripts/healthcheck.py --timeout 120  # wait up to 120s for services

Exit codes:
    0 — all services healthy
    1 — one or more services unhealthy
"""

import argparse
import logging
import sys
import time
from dataclasses import dataclass

import httpx

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger("ragops.healthcheck")


@dataclass
class ServiceCheck:
    """Configuration for a single service health check."""

    name: str
    url: str
    expected_status: int = 200
    json_key: str | None = None  # optional key to assert exists in response JSON


SERVICES: list[ServiceCheck] = [
    ServiceCheck(
        name="ChromaDB",
        url="http://localhost:8000/api/v1/heartbeat",
        json_key="nanosecond heartbeat",
    ),
    ServiceCheck(
        name="MLflow",
        url="http://localhost:5000/health",
    ),
    ServiceCheck(
        name="Ollama",
        url="http://localhost:11434/api/tags",
        json_key="models",
    ),
    ServiceCheck(
        name="FastAPI (RAGOps)",
        url="http://localhost:8080/health",
        json_key="status",
    ),
]


def check_service(service: ServiceCheck, timeout: float = 10.0) -> tuple[bool, str]:
    """
    Perform a single HTTP health check against a service.

    Parameters
    ----------
    service : ServiceCheck
        Service configuration including URL and expected response shape.
    timeout : float
        HTTP request timeout in seconds.

    Returns
    -------
    tuple[bool, str]
        (is_healthy, detail_message)
    """
    try:
        response = httpx.get(service.url, timeout=timeout)
        if response.status_code != service.expected_status:
            return (
                False,
                f"HTTP {response.status_code} (expected {service.expected_status})",
            )

        if service.json_key is not None:
            body = response.json()
            if service.json_key not in body:
                return False, f"Response JSON missing key '{service.json_key}'"

        return True, f"HTTP {response.status_code} OK"

    except httpx.ConnectError:
        return False, "Connection refused — service may not be running"
    except httpx.TimeoutException:
        return False, f"Timed out after {timeout}s"
    except Exception as exc:  # noqa: BLE001
        return False, f"Unexpected error: {exc}"


def wait_for_all_healthy(
    services: list[ServiceCheck],
    timeout: int = 60,
    poll_interval: int = 5,
) -> bool:
    """
    Poll all services until they are all healthy or the timeout is reached.

    Parameters
    ----------
    services : list[ServiceCheck]
        List of services to check.
    timeout : int
        Maximum seconds to wait for all services to become healthy.
    poll_interval : int
        Seconds between polling rounds.

    Returns
    -------
    bool
        True if all services became healthy within the timeout.
    """
    deadline = time.monotonic() + timeout
    unhealthy = set(s.name for s in services)

    while time.monotonic() < deadline and unhealthy:
        remaining_secs = int(deadline - time.monotonic())
        logger.info(
            "Waiting for services: %s (%ds remaining)",
            ", ".join(sorted(unhealthy)),
            remaining_secs,
        )

        for service in services:
            if service.name not in unhealthy:
                continue
            is_healthy, detail = check_service(service)
            if is_healthy:
                logger.info("✓ %-25s %s", service.name, detail)
                unhealthy.discard(service.name)
            else:
                logger.debug("✗ %-25s %s", service.name, detail)

        if unhealthy:
            time.sleep(poll_interval)

    return len(unhealthy) == 0


def run_once(services: list[ServiceCheck]) -> bool:
    """
    Run a single health check pass across all services and log results.

    Parameters
    ----------
    services : list[ServiceCheck]
        List of services to check.

    Returns
    -------
    bool
        True if all services are healthy.
    """
    all_healthy = True
    for service in services:
        is_healthy, detail = check_service(service)
        symbol = "✓" if is_healthy else "✗"
        level = logging.INFO if is_healthy else logging.ERROR
        logger.log(level, "%s %-25s %s", symbol, service.name, detail)
        if not is_healthy:
            all_healthy = False
    return all_healthy


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="RAGOps service health checker")
    parser.add_argument(
        "--timeout",
        type=int,
        default=0,
        help="Seconds to wait for services (0 = single pass, no waiting)",
    )
    parser.add_argument(
        "--poll-interval",
        type=int,
        default=5,
        help="Seconds between polling rounds when --timeout > 0",
    )
    return parser.parse_args()


def main() -> None:
    """Entry point."""
    args = parse_args()
    logger.info("RAGOps health check starting — %d service(s)", len(SERVICES))

    if args.timeout > 0:
        logger.info(
            "Waiting mode: up to %ds for all services to become healthy", args.timeout
        )
        success = wait_for_all_healthy(
            SERVICES,
            timeout=args.timeout,
            poll_interval=args.poll_interval,
        )
    else:
        success = run_once(SERVICES)

    if success:
        logger.info("All services healthy ✓")
        sys.exit(0)
    else:
        logger.error("One or more services are unhealthy ✗")
        sys.exit(1)


if __name__ == "__main__":
    main()
