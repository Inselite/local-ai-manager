#!/bin/sh
# Removes the app and its login agents. Your models and the app's data folder are left alone.
set -eu
LABEL=io.github.inselite.LocalAIManager
for l in "$LABEL" "$LABEL.splash"; do
    launchctl bootout "gui/$(id -u)/$l" 2>/dev/null || true
    rm -f "$HOME/Library/LaunchAgents/$l.plist"
done
rm -rf "$HOME/Applications/Local AI Manager.app"
echo "Removed. Favorites and history are still in ~/Library/Application Support/Local AI Manager"
echo "(delete that folder too if you don't want them)."
