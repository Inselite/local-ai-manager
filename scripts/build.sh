#!/bin/sh
# Builds build/Local AI Manager.app (ad-hoc signed, Apple silicon, macOS 26+).
set -eu
cd "$(dirname "$0")/.."
APP="build/Local AI Manager.app"
rm -rf "$APP"
mkdir -p "$APP/Contents/MacOS" "$APP/Contents/Resources"
swiftc -O -suppress-warnings -target arm64-apple-macos26.0 Sources/LocalAIManager.swift -o "$APP/Contents/MacOS/LocalAIManager"
cp Resources/Info.plist "$APP/Contents/Info.plist"
cp LICENSE "$APP/Contents/Resources/LICENSE"
codesign --force --sign - "$APP" 2>/dev/null
echo "Built $APP"
