#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_FILE="${ROOT_DIR}/.env.platform"
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export MODELSCOPE_CACHE="${MODELSCOPE_CACHE:-$HOME/.cache/modelscope}"

if [[ -f "${ENV_FILE}" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "${ENV_FILE}"
  set +a
fi

hf_has_token() {
  uv run --project "${ROOT_DIR}" python - <<'PY'
from huggingface_hub import get_token
raise SystemExit(0 if get_token() else 1)
PY
}

echo "==> GitHub"
git config --global url."git@github.com:".insteadOf https://github.com/
git config --global credential.helper osxkeychain
gh config set git_protocol ssh >/dev/null

if gh auth status >/dev/null 2>&1; then
  echo "[ok] gh already logged in"
elif [[ -n "${GITHUB_TOKEN:-}" ]]; then
  printf '%s' "${GITHUB_TOKEN}" | gh auth login --hostname github.com --git-protocol ssh --with-token
  echo "[ok] gh logged in with GITHUB_TOKEN"
else
  echo "[skip] set GITHUB_TOKEN in benchmark/.env.platform or run: gh auth login"
fi

echo
echo "==> Hugging Face"
if hf_has_token >/dev/null 2>&1; then
  echo "[ok] Hugging Face already logged in"
elif [[ -n "${HF_TOKEN:-}" ]]; then
  HF_TOKEN_VALUE="${HF_TOKEN}" uv run --project "${ROOT_DIR}" python - <<'PY'
import os
from huggingface_hub import login

token = os.environ["HF_TOKEN_VALUE"]
login(token=token, add_to_git_credential=False)
PY
  echo "[ok] Hugging Face logged in with HF_TOKEN"
else
  echo "[skip] set HF_TOKEN in benchmark/.env.platform or run: uv run --project benchmark hf auth login"
fi

echo
echo "==> ModelScope"
if [[ -n "${MODELSCOPE_API_TOKEN:-}" ]]; then
  uv run --project "${ROOT_DIR}" modelscope --token "${MODELSCOPE_API_TOKEN}" login
  echo "[ok] ModelScope logged in with MODELSCOPE_API_TOKEN"
else
  echo "[skip] set MODELSCOPE_API_TOKEN in benchmark/.env.platform or run: uv run --project benchmark modelscope --token <token> login"
fi
