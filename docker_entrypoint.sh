#!/usr/bin/env bash
set -euo pipefail

CONFIG_PATH="/tmp/data_agent_config.yaml"

mkdir -p /logs /output

cat > "${CONFIG_PATH}" <<EOF
dataset:
  root_path: /input

agent:
  model: ${MODEL_NAME:-}
  api_base: ${MODEL_API_URL:-}
  api_key: ${MODEL_API_KEY:-}
  max_steps: ${AGENT_MAX_STEPS:-32}
  temperature: ${AGENT_TEMPERATURE:-0.0}

run:
  output_dir: /output
  run_id: .
  max_workers: ${RUN_MAX_WORKERS:-4}
  task_timeout_seconds: ${TASK_TIMEOUT_SECONDS:-600}
EOF

find /output -mindepth 1 -maxdepth 1 -type d -name 'task_*' -exec rm -rf {} +

dabench run-benchmark --config "${CONFIG_PATH}" 2>&1 | tee /logs/runtime.log
