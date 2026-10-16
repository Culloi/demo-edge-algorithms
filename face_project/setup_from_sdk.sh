#!/bin/sh
# Copy SDK headers, libraries, models, sample images, and reference faces.
# Usage: sh setup_from_sdk.sh [SDK_ROOT]
# SDK_ROOT is the Linux SDK directory containing include/, lib/, model/, facedb/, and image/.
# Default: ~/rknn-2.3.2-linux (the original SDK directory name).
set -e
PROJ="$(cd "$(dirname "$0")" && pwd)"
HOME_DIR="${HOME:-$(getent passwd "$(id -un)" | cut -d: -f6)}"
SDK="${1:-$HOME_DIR/rknn-2.3.2-linux}"

[ -f "$SDK/lib/libfacerecognition.so" ] || { echo "Error: SDK library not found at $SDK/lib/libfacerecognition.so. Run: sh setup_from_sdk.sh <SDK_ROOT>"; exit 1; }
echo "Copying SDK dependencies: $SDK -> $PROJ"

mkdir -p "$PROJ/include" "$PROJ/lib" "$PROJ/models" "$PROJ/images" "$PROJ/facedb"

# 1) Public headers, bundled OpenCV headers, and RKNN headers.
cp -rf "$SDK/include/." "$PROJ/include/"

# 2) Face SDK, its dependencies, and bundled OpenCV world.
cp -f  "$SDK"/lib/*.so  "$PROJ/lib/"

# 3) RK3576 detection, landmark, and recognition models.
cp -f  "$SDK/model/rknn/RK3576/facedet_512_q_3576_enc.rknn"   "$PROJ/models/"
cp -f  "$SDK/model/rknn/RK3576/facelandmark_q_3576_enc.rknn"  "$PROJ/models/"
cp -f  "$SDK/model/rknn/RK3576/facereg_l_q_3576_enc.rknn"     "$PROJ/models/"

# 4) Reference faces and sample images; reference filenames provide identities.
cp -f  "$SDK"/facedb/* "$PROJ/facedb/" 2>/dev/null || true
cp -f  "$SDK"/image/*  "$PROJ/images/" 2>/dev/null || true

echo "Dependencies copied. Project files:"
ls -1 "$PROJ/lib" "$PROJ/models"
echo "Reference face images:"
ls -1 "$PROJ/facedb" 2>/dev/null || true
