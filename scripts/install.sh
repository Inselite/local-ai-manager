#!/bin/sh
# Builds the app, copies it to ~/Applications and starts it at login (restarted if it crashes).
set -eu
cd "$(dirname "$0")/.."
LABEL=io.github.inselite.LocalAIManager
DEST="$HOME/Applications/Local AI Manager.app"
PLIST="$HOME/Library/LaunchAgents/$LABEL.plist"

./scripts/build.sh
launchctl bootout "gui/$(id -u)/$LABEL" 2>/dev/null || true
mkdir -p "$HOME/Applications" "$HOME/Library/LaunchAgents"
rm -rf "$DEST"
ditto "build/Local AI Manager.app" "$DEST"

cat > "$PLIST" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
	<key>Label</key>
	<string>$LABEL</string>
	<key>ProgramArguments</key>
	<array>
		<string>$DEST/Contents/MacOS/LocalAIManager</string>
	</array>
	<key>RunAtLoad</key>
	<true/>
	<key>KeepAlive</key>
	<dict>
		<key>SuccessfulExit</key>
		<false/>
	</dict>
	<key>ProcessType</key>
	<string>Interactive</string>
	<key>StandardOutPath</key>
	<string>/tmp/local-ai-manager.log</string>
	<key>StandardErrorPath</key>
	<string>/tmp/local-ai-manager.log</string>
</dict>
</plist>
PLIST

# The old copy can take a moment to stop; retry until launchd accepts the new one.
for _ in 1 2 3 4 5; do
    launchctl bootstrap "gui/$(id -u)" "$PLIST" 2>/dev/null && break
    sleep 1
done
launchctl print "gui/$(id -u)/$LABEL" >/dev/null
echo "Installed $DEST — look for the memory-chip icon in the menu bar."
