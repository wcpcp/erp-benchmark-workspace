#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
THIRD_PARTY_DIR="${ROOT_DIR}/_third_party"

mkdir -p "${THIRD_PARTY_DIR}"

clone_repo() {
  local target="$1"
  shift
  local url

  if [[ -d "${target}/.git" || -f "${target}/.downloaded" ]]; then
    echo "[skip] ${target} already exists"
    return 0
  fi

  for url in "$@"; do
    echo "[try] ${url}"
    if git clone "${url}" "${target}"; then
      touch "${target}/.downloaded"
      echo "[ok] cloned ${url}"
      return 0
    fi
  done

  echo "[warn] failed to clone into ${target}"
  return 1
}

download_archive() {
  local target="$1"
  local archive_url="$2"
  local tmp_zip
  local tmp_dir

  if [[ -d "${target}" && ! -z "$(ls -A "${target}" 2>/dev/null)" ]]; then
    echo "[skip] ${target} already has content"
    return 0
  fi

  tmp_zip="$(mktemp -t erp-bench.XXXXXX.zip)"
  tmp_dir="$(mktemp -d -t erp-bench.XXXXXX)"

  echo "[try] ${archive_url}"
  if curl -fL "${archive_url}" -o "${tmp_zip}"; then
    unzip -q "${tmp_zip}" -d "${tmp_dir}"
    mkdir -p "${target}"
    find "${tmp_dir}" -mindepth 1 -maxdepth 1 -type d -exec cp -R {}/* "${target}/" \;
    touch "${target}/.downloaded"
    echo "[ok] unpacked ${archive_url}"
  else
    echo "[warn] archive download failed for ${archive_url}"
  fi

  rm -f "${tmp_zip}"
  rm -rf "${tmp_dir}"
}

echo "==> Fetching public benchmark sources into ${THIRD_PARTY_DIR}"

clone_repo \
  "${THIRD_PARTY_DIR}/360Loc" \
  "https://github.com/HuajianUP/360Loc.git" \
  "git@github.com:HuajianUP/360Loc.git" \
|| download_archive \
  "${THIRD_PARTY_DIR}/360Loc" \
  "https://codeload.github.com/HuajianUP/360Loc/zip/refs/heads/main"

clone_repo \
  "${THIRD_PARTY_DIR}/habitat-lab" \
  "https://github.com/facebookresearch/habitat-lab.git"

clone_repo \
  "${THIRD_PARTY_DIR}/RxR" \
  "https://github.com/google-research-datasets/RxR.git"

echo
echo "==> Hugging Face datasets are not mirrored by default."
echo "Use benchmark/scripts/download_hf_dataset.py for OSR-Bench or PanoEnv."
