#!/usr/bin/env bash

PROXY_HOST="${1:-127.0.0.1}"
PROXY_PORT="${2:-7890}"

export HTTP_PROXY="http://${PROXY_HOST}:${PROXY_PORT}"
export HTTPS_PROXY="http://${PROXY_HOST}:${PROXY_PORT}"
export ALL_PROXY="socks5://${PROXY_HOST}:${PROXY_PORT}"
export http_proxy="${HTTP_PROXY}"
export https_proxy="${HTTPS_PROXY}"
export all_proxy="${ALL_PROXY}"

git config --global http.proxy "${HTTP_PROXY}"
git config --global https.proxy "${HTTPS_PROXY}"

echo "Proxy enabled:"
echo "  HTTP_PROXY=${HTTP_PROXY}"
echo "  HTTPS_PROXY=${HTTPS_PROXY}"
echo "  ALL_PROXY=${ALL_PROXY}"
echo
echo "To apply in the current shell:"
echo "  source benchmark/scripts/enable_proxy.sh"
