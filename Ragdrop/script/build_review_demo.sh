#!/usr/bin/env bash
# Builds a separate synthetic-data app. Never stops or replaces Ragdrop.
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
SCRATCH="$ROOT_DIR/.build/review-demo"
swift build --scratch-path "$SCRATCH" -Xswiftc -DRAGDROP_REVIEW_DEMO
BINARY_DIR="$(swift build --scratch-path "$SCRATCH" --show-bin-path)"
APP="$ROOT_DIR/dist/Ragdrop Review Demo.app"
mkdir -p "$APP/Contents/MacOS" "$APP/Contents/Resources"
cp "$BINARY_DIR/Ragdrop" "$APP/Contents/MacOS/RagdropReviewDemo"
cp "$ROOT_DIR/Assets/RagdropIcon.icns" "$APP/Contents/Resources/RagdropIcon.icns"
cat > "$APP/Contents/Info.plist" <<'PLIST'
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
<key>CFBundleExecutable</key><string>RagdropReviewDemo</string>
<key>CFBundleIdentifier</key><string>com.tofunori.ragdrop.review-demo</string>
<key>CFBundleName</key><string>Ragdrop Review Demo</string>
<key>CFBundlePackageType</key><string>APPL</string>
<key>CFBundleIconFile</key><string>RagdropIcon</string>
<key>LSMinimumSystemVersion</key><string>14.0</string>
<key>NSPrincipalClass</key><string>NSApplication</string>
</dict></plist>
PLIST
codesign --force --deep --sign - "$APP"
if [[ "${1:-build}" == "--open" ]]; then /usr/bin/open -n "$APP"; fi
printf 'Built %s\n' "$APP"
