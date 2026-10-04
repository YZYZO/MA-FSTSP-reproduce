#!/usr/bin/env bash
set -euo pipefail

# 在Linux服务器上真实复核RL导出的五个候选；默认每个仓库组最多求解600秒。
PROJECT_ROOT="${PROJECT_ROOT:-/data/yongzhi/yangziheng/MA-FSTSP-reproduce}"
PYTHON_BIN="${PYTHON_BIN:-python}"
RESULT_ROOT="${RESULT_ROOT:-${PROJECT_ROOT}/results/服务器运行结果下载/260826_55k_master_results}"
CANDIDATES="${CANDIDATES:-${PROJECT_ROOT}/results/服务器运行结果下载/261004partition_rl_stage1_local/boston_11k_100_016/validation_candidates.json}"
SOLVER_TIME_LIMIT="${SOLVER_TIME_LIMIT:-600}"
LIMIT_TAG="${SOLVER_TIME_LIMIT//./p}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/results/partition_rl_stage1_true_validation/tl${LIMIT_TAG}}"
LOG_PATH="${OUTPUT_DIR}/rl_true_validation.log"

mkdir -p "${OUTPUT_DIR}"
cd "${PROJECT_ROOT}"

# 启动前明确检查本轮新增代码和候选文件，避免 set -e 静默退出。
required_files=(
  "${PROJECT_ROOT}/scripts/run_partition_rl_validation.py"
  "${PROJECT_ROOT}/src/partition_learning/rl_validation.py"
  "${CANDIDATES}"
)
missing_files=()
for required_file in "${required_files[@]}"; do
  if [[ ! -f "${required_file}" ]]; then
    missing_files+=("${required_file}")
  fi
done
if (( ${#missing_files[@]} > 0 )); then
  echo "缺少RL真实复核所需文件：" >&2
  printf '  - %s\n' "${missing_files[@]}" >&2
  echo "请把本地 partition_rl_validation_server_bundle.tar.gz 上传并在项目根目录解压。" >&2
  exit 2
fi

echo "候选文件：${CANDIDATES}"
echo "求解上限：${SOLVER_TIME_LIMIT} 秒/仓库组"
echo "准备启动RL真实复核……"

nohup "${PYTHON_BIN}" scripts/run_partition_rl_validation.py \
  --candidates "${CANDIDATES}" \
  --result-root "${RESULT_ROOT}" \
  --output-dir "${OUTPUT_DIR}" \
  --solver-time-limit "${SOLVER_TIME_LIMIT}" \
  --solver-threads 1 \
  --only-graph boston_11k \
  --customer-counts 100 \
  >"${LOG_PATH}" 2>&1 </dev/null &

experiment_pid=$!
echo "PID=${experiment_pid}"
echo "LOG=${LOG_PATH}"
echo "OUTPUT=${OUTPUT_DIR}"

# Python会先输出候选总数再加载路网；据此检查任务范围，而不等待昂贵求解完成。
sleep 15
if ! kill -0 "${experiment_pid}" 2>/dev/null; then
  if wait "${experiment_pid}"; then
    tail -n 40 "${LOG_PATH}"
    echo "RL真实复核已快速完成。"
    exit 0
  fi
  tail -n 40 "${LOG_PATH}"
  echo "RL真实复核进程启动失败。" >&2
  exit 1
fi
if ! grep -q '\[RL validation\] 候选总数=5' "${LOG_PATH}"; then
  kill "${experiment_pid}"
  wait "${experiment_pid}" || true
  tail -n 40 "${LOG_PATH}"
  echo "候选范围核对失败，任务已停止。" >&2
  exit 1
fi

tail -n 20 "${LOG_PATH}"
