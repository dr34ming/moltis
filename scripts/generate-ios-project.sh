#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
IOS_APP_DIR="${REPO_ROOT}/apps/ios"

if ! command -v xcodegen >/dev/null 2>&1; then
  echo "error: xcodegen is required (install with: brew install xcodegen)" >&2
  exit 1
fi

cd "${IOS_APP_DIR}"

if [ ! -f local.xcconfig ]; then
  echo "error: apps/ios/local.xcconfig not found." >&2
  echo "" >&2
  echo "  cp apps/ios/local.xcconfig.example apps/ios/local.xcconfig" >&2
  echo "" >&2
  echo "Then edit it and set DEVELOPMENT_TEAM to your Apple team ID." >&2
  echo "Find your team ID: Xcode > Settings > Accounts > select team > Team ID" >&2
  exit 1
fi

xcodegen generate --spec project.yml

echo "Generated ${IOS_APP_DIR}/Moltis.xcodeproj"
