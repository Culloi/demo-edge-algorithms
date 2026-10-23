#!/bin/sh
# Run the desktop gesture demo.
# Usage: ./run.sh [camera-device]
# Window controls: Open Camera, Local Image, Full Screen, Quit
# Shortcuts: Esc/Q to quit, F to toggle full screen
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

# Prefer project-bundled libraries over system libraries.
export LD_LIBRARY_PATH="$ROOT/lib:$ROOT/third_party/modelinfer:$ROOT/third_party/rknn:$ROOT/third_party/opencv/lib:$LD_LIBRARY_PATH"
# Set a valid code here or export GESTURE_AUTH_CODE before launching.
export GESTURE_AUTH_CODE="${GESTURE_AUTH_CODE:-XXXX}"
export GESTURE_LICENSE_FILE="${GESTURE_LICENSE_FILE:-$ROOT/authorization.lic}"

# Prefer the active Wayland session; fall back to XWayland.
export XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-/run/user/$(id -u)}"
WLPLUG=/usr/lib/aarch64-linux-gnu/qt5/plugins/platforms/libqwayland-generic.so
if [ -z "$QT_QPA_PLATFORM" ]; then
    if [ -S "$XDG_RUNTIME_DIR/wayland-0" ] && [ -e "$WLPLUG" ]; then
        export QT_QPA_PLATFORM=wayland
        export WAYLAND_DISPLAY="${WAYLAND_DISPLAY:-wayland-0}"
    else
        export QT_QPA_PLATFORM=xcb
        export DISPLAY="${DISPLAY:-:0}"
        XA=$(ls "$XDG_RUNTIME_DIR"/.mutter-Xwaylandauth.* 2>/dev/null | head -1)
        [ -n "$XA" ] && export XAUTHORITY="$XA"
    fi
fi

CAM="${1:-auto}"

echo "========================================================"
echo " Gesture Recognition Demo"
echo " Platform: QT_QPA_PLATFORM=$QT_QPA_PLATFORM   Camera: $CAM"
echo " Controls: Open Camera | Local Image | Full Screen | Quit"
echo " Shortcuts: Esc/Q to quit, F to toggle full screen"
echo "========================================================"
./gesture_cam_demo "$CAM"
