#!/usr/bin/env bash
set -euo pipefail

# 在 Linux 服务器上续跑阶段 7：严格使用六实例清单，并复用本地已完成的 12 条标签。
PROJECT_ROOT="${PROJECT_ROOT:-/data/yongzhi/yangziheng/MA-FSTSP-reproduce}"
PYTHON_BIN="${PYTHON_BIN:-python}"
RESULT_ROOT="${RESULT_ROOT:-${PROJECT_ROOT}/results/服务器运行结果下载/260826_55k_master_results}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/results/partition_learning_stage7_active_server}"
MANIFEST="${PROJECT_ROOT}/results/服务器运行结果下载/260930supervised_8stage/stage7_active_data/selected_instances.json"
BOOTSTRAP="${PROJECT_ROOT}/results/服务器运行结果下载/260930supervised_8stage/stage7_active_data/bootstrap_candidate_records.jsonl"
LOG_PATH="${OUTPUT_DIR}/stage7_server.log"

mkdir -p "${OUTPUT_DIR}"
cd "${PROJECT_ROOT}"

# 清单和 bootstrap 缺失时直接停止，避免意外退回默认 60 实例实验。
test -f "${MANIFEST}"
test -f "${BOOTSTRAP}"

nohup "${PYTHON_BIN}" scripts/run_partition_learning_rounds.py \
  --round algorithms \
  --result-root "${RESULT_ROOT}" \
  --output-dir "${OUTPUT_DIR}" \
  --solver-time-limit 600 \
  --algorithm-candidates-per-instance 12 \
  --instance-indices-manifest "${MANIFEST}" \
  --bootstrap-records "${BOOTSTRAP}" \
  --customer-counts 50 100 150 \
  >"${LOG_PATH}" 2>&1 </dev/null &

experiment_pid=$!
echo "PID=${experiment_pid}"
echo "LOG=${LOG_PATH}"

# 启动后核对日志必须出现六实例分母，否则停止错误范围的任务。
sleep 15
if ! kill -0 "${experiment_pid}" 2>/dev/null; then
  tail -n 30 "${LOG_PATH}"
  echo "阶段 7 进程启动失败。" >&2
  exit 1
fi
if ! grep -Eq '[0-6]/6' "${LOG_PATH}"; then
  kill "${experiment_pid}"
  wait "${experiment_pid}" || true
  tail -n 30 "${LOG_PATH}"
  echo "实例范围核对失败，任务已停止。" >&2
  exit 1
fi

tail -n 12 "${LOG_PATH}"
