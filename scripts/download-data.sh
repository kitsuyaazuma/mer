#!/usr/bin/env bash
# Downloads the function-calling data (BFCL v3, ToolACE, When2Call) into ./data.
set -euo pipefail

readonly DATA_DIR="${DATA_DIR:-data}"
readonly BFCL_SHA="916260dfc116bf06793a1af79b4ec8195b0453b6" # the BFCL version of Jev Decision Index
readonly BFCL_URL="https://raw.githubusercontent.com/ShishirPatil/gorilla/${BFCL_SHA}/berkeley-function-call-leaderboard/data"
readonly HF_URL="https://huggingface.co/datasets"

# Downloads url to file, unless file is already there.
fetch() {
  local url="$1" file="$2"
  if [[ -s "${file}" ]]; then
    return 0
  fi
  mkdir -p "$(dirname "${file}")"
  echo "downloading ${url}"
  curl -fsSL --retry 3 "${url}" -o "${file}.part"
  mv "${file}.part" "${file}"
}

main() {
  local c f
  for c in simple multiple parallel parallel_multiple live_simple live_multiple live_parallel live_parallel_multiple; do
    fetch "${BFCL_URL}/BFCL_v3_${c}.json" "${DATA_DIR}/bfcl/BFCL_v3_${c}.json"
    fetch "${BFCL_URL}/possible_answer/BFCL_v3_${c}.json" "${DATA_DIR}/bfcl/possible_answer_BFCL_v3_${c}.json"
  done
  for c in irrelevance live_irrelevance; do # only their tool names are read
    fetch "${BFCL_URL}/BFCL_v3_${c}.json" "${DATA_DIR}/bfcl/BFCL_v3_${c}.json"
  done
  fetch "${HF_URL}/Team-ACE/ToolACE/resolve/main/data.json" "${DATA_DIR}/toolace/data.json"
  for f in test/when2call_test_mcq.jsonl train/when2call_train_sft.jsonl train/when2call_train_pref.jsonl; do
    fetch "${HF_URL}/nvidia/When2Call/resolve/main/${f}" "${DATA_DIR}/when2call/${f}"
  done
  echo "done: ${DATA_DIR}/bfcl, ${DATA_DIR}/toolace, ${DATA_DIR}/when2call"
}

main "$@"
