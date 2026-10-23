#!/bin/sh
# Copy the libraries, headers, models, and sample image required by this demo.
# Usage: sh setup_from_sdk.sh [SDK_ROOT]
# SDK_ROOT must contain include/, lib/, 3rd_party/, and models/.
# Default: ~/gesture_RK3576
set -e
PROJ="$(cd "$(dirname "$0")" && pwd)"
HOME_DIR="${HOME:-$(getent passwd "$(id -un)" | cut -d: -f6)}"
SDK="${1:-$HOME_DIR/gesture_RK3576}"

[ -f "$SDK/lib/libgesture.so" ] || { echo "Error: SDK library not found at $SDK/lib/libgesture.so. Pass SDK_ROOT: sh setup_from_sdk.sh <SDK_ROOT>"; exit 1; }
echo "Copying SDK dependencies: $SDK -> $PROJ"

mkdir -p "$PROJ/include" "$PROJ/lib" "$PROJ/models" "$PROJ/images"
mkdir -p "$PROJ/third_party/opencv/include" "$PROJ/third_party/opencv/lib"
mkdir -p "$PROJ/third_party/modelinfer" "$PROJ/third_party/rknn"

# 1) Gesture SDK public header and library.
cp -f  "$SDK/include/gesture.h"                        "$PROJ/include/"
cp -f  "$SDK/lib/libgesture.so"                        "$PROJ/lib/"

# 2) Third-party runtime libraries required by libgesture.so.
cp -f  "$SDK/3rd_party/modelinfer/libinference.so"     "$PROJ/third_party/modelinfer/"
cp -f  "$SDK/3rd_party/modelinfer/libauthorization.so" "$PROJ/third_party/modelinfer/"
cp -f  "$SDK/3rd_party/rknn/librknnrt.so"              "$PROJ/third_party/rknn/"

# 3) Bundled OpenCV headers and world library.
cp -rf "$SDK/3rd_party/opencv/include/." "$PROJ/third_party/opencv/include/"
cp -f  "$SDK/3rd_party/opencv/lib/libopencv_world.so"  "$PROJ/third_party/opencv/lib/"

# 4) Models and sample images.
cp -f  "$SDK/models/board6_RK3576_Ubuntu_hand_det.engine"    "$PROJ/models/"
cp -f  "$SDK/models/board6_RK3576_Ubuntu_gesture_cls.engine" "$PROJ/models/"
cp -f  "$SDK"/images/*.jpg "$PROJ/images/" 2>/dev/null || true

echo "Dependencies copied. Project files:"
ls -1 "$PROJ/lib" "$PROJ/third_party/modelinfer" "$PROJ/third_party/rknn" \
       "$PROJ/third_party/opencv/lib" "$PROJ/models"
