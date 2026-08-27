#!/usr/bin/env bash
set -e
mkdir -p outputs/logs
python -m src.run_experiments --config configs/full_optional.yaml 2>&1 | tee outputs/logs/full_optional.log
