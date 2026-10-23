#!/bin/sh
# Build the project after copying SDK dependencies with setup_from_sdk.sh.
set -e
ROOT="$(cd "$(dirname "$0")" && pwd)"; cd "$ROOT"
if [ ! -f lib/libgesture.so ] || [ ! -f third_party/opencv/lib/libopencv_world.so ]; then
    echo "Dependencies are missing. Run: sh setup_from_sdk.sh [SDK_ROOT]"
    exit 1
fi
make -j
echo "build OK -> $ROOT/gesture_cam_demo"
