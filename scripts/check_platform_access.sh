#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "==> Git version"
git --version

echo
echo "==> GitHub SSH"
ssh -T git@github.com || true

echo
echo "==> GitHub repo access smoke test"
git ls-remote https://github.com/git/git.git HEAD

echo
echo "==> gh auth status"
gh auth status || true

echo
echo "==> Hugging Face public access"
uv run --project "${ROOT_DIR}" hf download gpt2 README.md --repo-type model --local-dir /tmp/hf-gpt2-readme >/dev/null
ls -l /tmp/hf-gpt2-readme/README.md
uv run --project "${ROOT_DIR}" python - <<'PY'
from huggingface_hub import get_token
print("HF token:", "present" if get_token() else "not logged in")
PY

echo
echo "==> ModelScope public access"
uv run --project "${ROOT_DIR}" python - <<'PY'
from modelscope.hub.snapshot_download import snapshot_download
path = snapshot_download('Qwen/Qwen2.5-0.5B', allow_file_pattern='README.md', cache_dir='/tmp/modelscope-cache')
print(path)
PY
