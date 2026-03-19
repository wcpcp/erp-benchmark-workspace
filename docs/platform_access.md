# Platform Access Setup

This workspace now has a minimal access toolchain for GitHub, Hugging Face, and ModelScope.

## What is already configured

- GitHub `https://github.com/...` is rewritten to SSH via global Git config.
- Git credentials are stored with `osxkeychain`.
- `gh` default Git protocol is set to `ssh`.
- `~/.ssh/config` now has a `github.com` host entry using `~/.ssh/id_ed25519`.
- Python-side tools are now meant to run through [`uv`](https://docs.astral.sh/uv/) from [`benchmark/pyproject.toml`](/Users/wcp/code/erp_data_pipeline/benchmark/pyproject.toml).

## One-time login

Copy the template and fill in the tokens you want to use:

```bash
cp benchmark/.env.platform.example benchmark/.env.platform
```

Then run:

```bash
uv sync --project benchmark
bash benchmark/scripts/login_platforms.sh
```

If you prefer interactive login instead of token files:

```bash
gh auth login
uv run --project benchmark hf auth login
uv run --project benchmark modelscope --token <your_token> login
```

## Proxy

If you need the local proxy:

```bash
source benchmark/scripts/enable_proxy.sh
```

Or specify a custom host and port:

```bash
source benchmark/scripts/enable_proxy.sh 127.0.0.1 7890
```

## Verification

Run:

```bash
uv sync --project benchmark
bash benchmark/scripts/check_platform_access.sh
```

This checks:

- GitHub SSH auth
- GitHub public repo clone access
- `gh` auth status
- Hugging Face public download and token state
- ModelScope public download
