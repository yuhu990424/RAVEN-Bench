#!/usr/bin/env python3
"""Compatibility facade for the RAVEN-Bench benchmark commands."""

from raven_benchmark.benchmark import *  # noqa: F401,F403
from raven_benchmark.benchmark import main


if __name__ == "__main__":
    main()
