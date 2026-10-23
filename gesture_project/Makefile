# Build gesture_cam_demo for RK3576 Debian aarch64.
# SDK headers, libraries, and models are copied by setup_from_sdk.sh.
# Use system Qt5 for the UI and bundled OpenCV world for image processing.

CXX      ?= g++
CXXFLAGS := -std=c++17 -O2 -fPIC
QT_CFLAGS:= $(shell pkg-config --cflags Qt5Widgets Qt5Gui Qt5Core)
QT_LIBS  := $(shell pkg-config --libs Qt5Widgets Qt5Gui Qt5Core)

INCLUDES := -Iinclude -Ithird_party/opencv/include $(QT_CFLAGS)
LDDIRS   := -Llib -Lthird_party/opencv/lib
LIBS     := -lgesture -lopencv_world $(QT_LIBS) -lpthread -ldl
RPATHLINK:= -Wl,-rpath-link,lib \
            -Wl,-rpath-link,third_party/modelinfer \
            -Wl,-rpath-link,third_party/rknn \
            -Wl,-rpath-link,third_party/opencv/lib \
            -Wl,--allow-shlib-undefined

TARGET := gesture_cam_demo
SRC    := src/gesture_cam_demo.cpp

all: $(TARGET)

$(TARGET): $(SRC)
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(SRC) $(LDDIRS) $(LIBS) $(RPATHLINK) -o $(TARGET)

clean:
	rm -f $(TARGET)

.PHONY: all clean
