#!/usr/bin/env bash
set -euo pipefail

# 在Linux服务器上真实复核PPO迁移子集，并复用第一轮五候选的仓库组缓存。
PROJECT_ROOT="${PROJECT_ROOT:-/data/yongzhi/yangziheng/MA-FSTSP-reproduce}"
PYTHON_BIN="${PYTHON_BIN:-python}"
RESULT_ROOT="${RESULT_ROOT:-${PROJECT_ROOT}/results/服务器运行结果下载/260826_55k_master_results}"
CANDIDATES="${CANDIDATES:-${PROJECT_ROOT}/results/服务器运行结果下载/261004partition_rl_stage2_subsets/subset_candidates_for_true_validation.json}"
PRIMARY_GROUP_CACHE="${PROJECT_ROOT}/results/partition_rl_stage1_true_validation/tl600/group_evaluations.sqlite3"
BUNDLED_GROUP_CACHE="${PROJECT_ROOT}/results/服务器运行结果下载/261004partition_rl_stage1_true_validation/tl600/group_evaluations.sqlite3"
if [[ -z "${GROUP_CACHE:-}" ]]; then
  if [[ -f "${PRIMARY_GROUP_CACHE}" ]]; then
    GROUP_CACHE="${PRIMARY_GROUP_CACHE}"
  else
    GROUP_CACHE="${BUNDLED_GROUP_CACHE}"
  fi
fi
SOLVER_TIME_LIMIT="${SOLVER_TIME_LIMIT:-600}"
EXPECTED_CANDIDATE_COUNT="${EXPECTED_CANDIDATE_COUNT:-6}"
LIMIT_TAG="${SOLVER_TIME_LIMIT//./p}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/results/partition_rl_stage2_subset_validation/tl${LIMIT_TAG}}"
LOG_PATH="${OUTPUT_DIR}/rl_subset_validation.log"

mkdir -p "${OUTPUT_DIR}"
cd "${PROJECT_ROOT}"

# 新代码、候选和上一轮SQLite缓存缺一不可；明确报错而不是静默退出。
required_files=(
  "${PROJECT_ROOT}/scripts/run_partition_rl_validation.py"
  "${PROJECT_ROOT}/src/partition_learning/rl_validation.py"
  "${CANDIDATES}"
  "${GROUP_CACHE}"
)
missing_files=()
for required_file in "${required_files[@]}"; do
  if [[ ! -f "${required_file}" ]]; then
    missing_files+=("${required_file}")
  fi
done
if (( ${#missing_files[@]} > 0 )); then
  echo "缺少PPO子集真实复核所需文件：" >&2
  printf '  - %s\n' "${missing_files[@]}" >&2
  exit 2
fi

echo "候选文件：${CANDIDATES}"
echo "复用缓存：${GROUP_CACHE}"
echo "求解上限：${SOLVER_TIME_LIMIT} 秒/仓库组"

nohup "${PYTHON_BIN}" scripts/run_partition_rl_validation.py \
  --candidates "${CANDIDATES}" \
  --result-root "${RESULT_ROOT}" \
  --output-dir "${OUTPUT_DIR}" \
  --group-cache "${GROUP_CACHE}" \
  --solver-time-limit "${SOLVER_TIME_LIMIT}" \
  --solver-threads 1 \
  --only-graph boston_11k \
  --customer-counts 100 \
  >"${LOG_PATH}" 2>&1 </dev/null &

experiment_pid=$!
echo "PID=${experiment_pid}"
echo "LOG=${LOG_PATH}"
echo "OUTPUT=${OUTPUT_DIR}"

sleep 15
if ! kill -0 "${experiment_pid}" 2>/dev/null; then
  if wait "${experiment_pid}"; then
    tail -n 40 "${LOG_PATH}"
    echo "PPO子集真实复核已快速完成。"
    exit 0
  fi
  tail -n 40 "${LOG_PATH}"
  echo "PPO子集真实复核进程启动失败。" >&2
  exit 1
fi
if ! grep -q "\[RL validation\] 候选总数=${EXPECTED_CANDIDATE_COUNT}" "${LOG_PATH}"; then
  kill "${experiment_pid}"
  wait "${experiment_pid}" || true
  tail -n 40 "${LOG_PATH}"
  echo "候选范围核对失败，任务已停止。" >&2
  exit 1
fi

tail -n 20 "${LOG_PATH}"
