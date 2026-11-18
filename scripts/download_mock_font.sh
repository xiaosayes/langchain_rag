#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FONT_DIR="$ROOT_DIR/assets/fonts"
BASE_URL="https://raw.githubusercontent.com/notofonts/noto-cjk/main/Sans"
TEMP_DIR="$(mktemp -d)"
trap 'rm -rf "$TEMP_DIR"' EXIT

curl --fail --location \
  "$BASE_URL/OTF/SimplifiedChinese/NotoSansCJKsc-Regular.otf" \
  --output "$TEMP_DIR/NotoSansCJKsc-Regular.otf"
curl --fail --location "$BASE_URL/LICENSE" --output "$TEMP_DIR/OFL.txt"
cp "$FONT_DIR/SHA256SUMS" "$TEMP_DIR/SHA256SUMS"
(
  cd "$TEMP_DIR"
  sha256sum --check SHA256SUMS
)

mkdir -p "$FONT_DIR"
mv "$TEMP_DIR/NotoSansCJKsc-Regular.otf" "$FONT_DIR/NotoSansCJKsc-Regular.otf"
mv "$TEMP_DIR/OFL.txt" "$FONT_DIR/OFL.txt"
