#!/usr/bin/env bash
set -e
mkdir -p outputs/logs
python -m src.run_experiments --config configs/smoke.yaml 2>&1 | tee outputs/logs/smoke.log
