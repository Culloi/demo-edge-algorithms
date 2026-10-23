#!/bin/sh
# Run the Face Recognition desktop demo.
# Usage: ./run.sh [camera-device]
# Controls: Open Camera, Local Image, Register Face, Landmarks, Full Screen, Quit
# Shortcuts: Esc/Q to quit, F to toggle full screen
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

# Prefer project libraries over system libraries.
export LD_LIBRARY_PATH="$ROOT/lib:$LD_LIBRARY_PATH"
# Set a valid code here or export FACE_AUTH_CODE before launching.
export FACE_AUTH_CODE="${FACE_AUTH_CODE:-XXXX}"
# RK3576 uses module 3, as defined in authorization.h.
export FACE_MODULE="${FACE_MODULE:-3}"
export FACE_APP_ID="${FACE_APP_ID:-145}"
export FACE_LICENSE_FILE="${FACE_LICENSE_FILE:-$ROOT/face_license.ini}"

# Prefer Wayland; fall back to XWayland (xcb).
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
echo " Face Recognition Demo"
echo " Platform: QT_QPA_PLATFORM=$QT_QPA_PLATFORM   Camera: $CAM   module=$FACE_MODULE"
echo " Controls: Open Camera | Local Image | Register Face | Landmarks | Full Screen | Quit"
echo " Shortcuts: Esc/Q to quit, F to toggle full screen"
echo "========================================================"
./face_cam_demo "$CAM"
