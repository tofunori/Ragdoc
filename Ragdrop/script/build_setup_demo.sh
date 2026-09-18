#!/usr/bin/env bash
# Builds a separate synthetic-data app. Never stops or replaces Ragdrop.
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
VALIDATION="${1:-preview}"
EXTRA_FLAGS=()
if [[ "$VALIDATION" == "validation" ]]; then EXTRA_FLAGS=(-Xswiftc -DRAGDROP_SETUP_VALIDATION); fi
SCRATCH="$ROOT_DIR/.build/setup-$VALIDATION"
swift build --scratch-path "$SCRATCH" -Xswiftc -DRAGDROP_SETUP_DEMO "${EXTRA_FLAGS[@]}"
BINARY_DIR="$(swift build --scratch-path "$SCRATCH" --show-bin-path)"
APP="$ROOT_DIR/dist/Ragdrop Setup Preview.app"
if [[ "$VALIDATION" == "validation" ]]; then APP="$ROOT_DIR/dist/Ragdrop Setup Validation.app"; fi
mkdir -p "$APP/Contents/MacOS" "$APP/Contents/Resources"
cp "$BINARY_DIR/Ragdrop" "$APP/Contents/MacOS/RagdropSetupPreview"
cp "$ROOT_DIR/Assets/RagdropIcon.icns" "$APP/Contents/Resources/RagdropIcon.icns"
cat > "$APP/Contents/Info.plist" <<'PLIST'
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
<key>CFBundleExecutable</key><string>RagdropSetupPreview</string>
<key>CFBundleIdentifier</key><string>com.tofunori.ragdrop.setup-preview</string>
<key>CFBundleName</key><string>Ragdrop Setup Preview</string>
<key>CFBundleDevelopmentRegion</key><string>en</string>
<key>CFBundleLocalizations</key><array><string>en</string></array>
<key>CFBundlePackageType</key><string>APPL</string>
<key>CFBundleIconFile</key><string>RagdropIcon</string>
<key>LSMinimumSystemVersion</key><string>14.0</string>
<key>NSPrincipalClass</key><string>NSApplication</string>
</dict></plist>
PLIST
if [[ "$VALIDATION" == "validation" ]]; then
  python3 "$ROOT_DIR/script/package_local_engine.py" "$APP/Contents/Resources"
fi
if [[ "$VALIDATION" == "validation" ]]; then
  /usr/libexec/PlistBuddy -c "Set :CFBundleIdentifier com.tofunori.ragdrop.setup-validation" "$APP/Contents/Info.plist"
  /usr/libexec/PlistBuddy -c "Set :CFBundleName Ragdrop Setup Validation" "$APP/Contents/Info.plist"
fi
codesign --force --deep --sign - "$APP"
if [[ "${1:-build}" == "--open" ]]; then /usr/bin/open -n "$APP"; fi
printf 'Built %s\n' "$APP"
