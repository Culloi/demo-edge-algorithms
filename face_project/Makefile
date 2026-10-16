# Build face_cam_demo for RK3576 Debian aarch64.
# setup_from_sdk.sh copies SDK dependencies into the project.
# Use system Qt5 for the UI and bundled OpenCV world for image processing.

CXX      ?= g++
CXXFLAGS := -std=c++17 -O2 -fPIC
QT_CFLAGS:= $(shell pkg-config --cflags Qt5Widgets Qt5Gui Qt5Core)
QT_LIBS  := $(shell pkg-config --libs Qt5Widgets Qt5Gui Qt5Core)

# Bundled OpenCV and RKNN headers are under include/.
INCLUDES := -Iinclude $(QT_CFLAGS)
LDDIRS   := -Llib
LIBS     := -lfacerecognition -lopencv_world $(QT_LIBS) -lpthread -ldl
# Resolve indirect dependencies from lib/ at link time.
# run.sh configures LD_LIBRARY_PATH for runtime loading.
RPATHLINK:= -Wl,-rpath-link,lib \
            -Wl,--allow-shlib-undefined

TARGET := face_cam_demo
SRC    := src/face_cam_demo.cpp

all: $(TARGET)

$(TARGET): $(SRC)
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(SRC) $(LDDIRS) $(LIBS) $(RPATHLINK) -o $(TARGET)

clean:
	rm -f $(TARGET)

.PHONY: all clean
