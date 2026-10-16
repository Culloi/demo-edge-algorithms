#!/bin/sh
# Build the demo after copying its SDK dependencies.
set -e
ROOT="$(cd "$(dirname "$0")" && pwd)"; cd "$ROOT"
if [ ! -f lib/libfacerecognition.so ] || [ ! -f lib/libopencv_world.so ]; then
    echo "Dependencies are missing. Run: sh setup_from_sdk.sh [SDK_ROOT]"
    exit 1
fi
make -j
echo "build OK -> $ROOT/face_cam_demo"
