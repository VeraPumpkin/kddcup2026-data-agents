#!/usr/bin/env bash
set -euo pipefail

CONFIG_PATH="/tmp/data_agent_config.yaml"
RUN_ID="official_eval_run"
RUN_OUTPUT_DIR="/output/${RUN_ID}"

mkdir -p /logs

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
  run_id: ${RUN_ID}
  max_workers: ${RUN_MAX_WORKERS:-4}
  task_timeout_seconds: ${TASK_TIMEOUT_SECONDS:-600}
EOF

rm -rf "${RUN_OUTPUT_DIR}"

dabench run-benchmark --config "${CONFIG_PATH}" 2>&1 | tee /logs/runtime.log

if [ -d "${RUN_OUTPUT_DIR}" ]; then
    find "${RUN_OUTPUT_DIR}" -mindepth 1 -maxdepth 1 -type d -name 'task_*' -exec cp -R {} /output/ \;
fi
