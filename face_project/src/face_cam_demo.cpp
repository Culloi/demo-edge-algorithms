/*
 * Face Recognition Demo for RK3576 Debian aarch64.
 *
 * Detect faces, estimate landmarks and pose, and match 512-dimensional features.
 * A worker handles capture and inference; the Qt GUI displays owned image copies.
 * Registered face crops persist as facedb/<name>.jpg.
 */
#include "faceEngine.h"

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/videoio.hpp>

#include <QApplication>
#include <QDialog>
#include <QDir>
#include <QFile>
#include <QFileDialog>
#include <QFileInfo>
#include <QFrame>
#include <QGridLayout>
#include <QHBoxLayout>
#include <QImage>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>
#include <QKeySequence>
#include <QLabel>
#include <QLineEdit>
#include <QMessageBox>
#include <QPixmap>
#include <QProgressBar>
#include <QPushButton>
#include <QSaveFile>
#include <QShortcut>
#include <QSizePolicy>
#include <QString>
#include <QStyle>
#include <QStyleFactory>
#include <QTimer>
#include <QVBoxLayout>
#include <QWidget>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include <csignal>
#include <dirent.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/ioctl.h>
#include <sys/statvfs.h>
#include <linux/videodev2.h>

enum Mode { MODE_IDLE = 0, MODE_CAMERA = 1, MODE_IMAGE = 2 };

static std::mutex            g_mtx;
static QImage                g_img;
static std::atomic<uint64_t> g_fid{0};
static std::atomic<bool>     g_running{true};

static std::atomic<int>      g_mode{MODE_IDLE};
static std::atomic<bool>     g_image_dirty{false};
static std::atomic<bool>     g_show_landmark{true};
static std::mutex            g_cmd_mtx;
static std::string           g_image_path;       /* Protected by g_cmd_mtx. */
static std::string           g_cam_dev = "auto";  /* Auto-detect, device path, or numeric index. */

/* Feature and name arrays share indices and are protected by g_db_mtx. */
static std::mutex                        g_db_mtx;
static std::vector<std::vector<float>>   g_db_feats;
static std::vector<std::string>          g_db_names;

/* The GUI requests registration; the worker enrolls the largest usable face. */
static std::atomic<bool>     g_register_pending{false};
static std::string           g_register_name;    /* Protected by g_cmd_mtx. */
static std::atomic<int>      g_register_result{0}; /* 1=saved, -1=no usable face, -2=save failed */

/* The worker pauses capture while the registration dialog is open. */
static std::atomic<bool>     g_freeze_request{false};
static std::atomic<bool>     g_unfreeze_request{false};
static std::atomic<bool>     g_frozen{false};
static std::mutex            g_frozen_mtx;
static cv::Mat               g_frozen_mat;              /* Protected by g_frozen_mtx. */

static float                 g_match_thresh = 0.6f;

static void on_sig(int) { g_running = false; }

/* Filter out ISP/VPSS nodes and select the first USB video-capture node. */
static std::string find_usb_camera() {
    std::vector<int> ids;
    if (DIR* d = ::opendir("/dev")) {
        while (struct dirent* e = ::readdir(d)) {
            if (std::strncmp(e->d_name, "video", 5) != 0) continue;
            char* end = nullptr;
            long n = std::strtol(e->d_name + 5, &end, 10);
            if (end && *end == '\0') ids.push_back((int)n);
        }
        ::closedir(d);
    }
    std::sort(ids.begin(), ids.end());
    for (int n : ids) {
        char path[64];
        std::snprintf(path, sizeof(path), "/dev/video%d", n);
        int fd = ::open(path, O_RDWR | O_NONBLOCK);
        if (fd < 0) continue;
        struct v4l2_capability cap;
        std::memset(&cap, 0, sizeof(cap));
        bool ok = false;
        if (::ioctl(fd, VIDIOC_QUERYCAP, &cap) == 0) {
            unsigned caps = (cap.capabilities & V4L2_CAP_DEVICE_CAPS) ? cap.device_caps
                                                                      : cap.capabilities;
            bool is_capture = caps & V4L2_CAP_VIDEO_CAPTURE;
            bool is_usb = std::strstr((const char*)cap.bus_info, "usb") != nullptr;
            if (is_capture && is_usb) ok = true;
        }
        ::close(fd);
        if (ok) { std::printf("[cam] Detected USB camera: %s (%s)\n", path, cap.card); return path; }
    }
    return "";
}

static std::string stem_of(const std::string& path) {
    size_t s = path.find_last_of("/\\");
    std::string name = (s == std::string::npos) ? path : path.substr(s + 1);
    size_t d = name.find_last_of('.');
    if (d != std::string::npos) name = name.substr(0, d);
    return name;
}

static bool valid_face_feature(const std::vector<float>& feature) {
    return feature.size() == 512 &&
           std::all_of(feature.begin(), feature.end(), [](float value) { return std::isfinite(value); });
}

static bool save_registered_face(const std::string& name, const cv::Mat& frame, const FaceResults_t& face) {
    if (name.empty() || name.size() > 128 || name.front() == '.' ||
        name.find_first_of("/\\<>:\"|?*") != std::string::npos ||
        std::any_of(name.begin(), name.end(), [](unsigned char value) { return value < 32 || value == 127; }) ||
        frame.empty() || face.width <= 0 || face.height <= 0 || !valid_face_feature(face.feature)) return false;
    QDir directory("facedb");
    if (!directory.mkpath(".")) return false;
    const QString path = directory.filePath(QString::fromStdString(name) + ".jpg");
    if (QFileInfo::exists(path)) return false;

    const double horizontalPadding = face.width * 0.3;
    const double verticalPadding = face.height * 0.3;
    const int left = static_cast<int>(std::clamp(face.x - horizontalPadding, 0.0, double(frame.cols)));
    const int top = static_cast<int>(std::clamp(face.y - verticalPadding, 0.0, double(frame.rows)));
    const int right = static_cast<int>(std::clamp(std::ceil(double(face.x) + face.width + horizontalPadding),
                                                0.0, double(frame.cols)));
    const int bottom = static_cast<int>(std::clamp(std::ceil(double(face.y) + face.height + verticalPadding),
                                                 0.0, double(frame.rows)));
    if (right <= left || bottom <= top) return false;
    std::vector<unsigned char> jpeg;
    if (!cv::imencode(".jpg", frame(cv::Rect(left, top, right - left, bottom - top)), jpeg,
                      {cv::IMWRITE_JPEG_QUALITY, 95})) return false;
    QSaveFile file(path);
    if (!file.open(QIODevice::WriteOnly)) return false;
    if (!file.setPermissions(QFileDevice::ReadOwner | QFileDevice::WriteOwner)) return false;
    const qint64 size = static_cast<qint64>(jpeg.size());
    if (file.write(reinterpret_cast<const char*>(jpeg.data()), size) != size) return false;
    return file.commit();
}

static void load_registered_faces() {
    QDir directory("facedb/registered");
    for (const QString& filename : directory.entryList(QStringList() << "*.json", QDir::Files, QDir::Name)) {
        QFile file(directory.filePath(filename));
        if (!file.open(QIODevice::ReadOnly)) continue;
        const QJsonObject record = QJsonDocument::fromJson(file.readAll()).object();
        const QString name = record.value("name").toString();
        if (record.value("version").toInt() != 1 || name.trimmed().isEmpty()) continue;
        const QJsonArray values = record.value("feature").toArray();
        std::vector<float> feature;
        bool valid = true;
        for (const QJsonValue& value : values) {
            if (!value.isDouble()) { valid = false; break; }
            feature.push_back(static_cast<float>(value.toDouble()));
        }
        if (!valid || !valid_face_feature(feature)) {
            std::printf("[facedb] Skipping invalid feature record: %s\n", filename.toStdString().c_str());
            continue;
        }
        std::lock_guard<std::mutex> lock(g_db_mtx);
        g_db_feats.push_back(std::move(feature));
        g_db_names.push_back(name.toStdString());
    }
}

/* Load reference JPEGs and retain compatibility with older JSON registrations. */
static void load_facedb(FaceEngine& engine, const std::string& dir) {
    std::vector<cv::String> files;
    cv::glob(dir, files);
    int ok = 0;
    for (const auto& f : files) {
        if (!QFileInfo(QString::fromStdString(std::string(f.c_str()))).isFile()) continue;
        cv::Mat img = cv::imread(f);
        if (img.empty()) continue;
        if (!img.isContinuous()) img = img.clone();
        std::vector<FaceResults_t> res;
        engine.inference(img.data, img.rows, img.cols, img.channels(), res);
        for (const auto& r : res) {
            if (r.code == FACE_OK && !r.feature.empty()) {
                std::lock_guard<std::mutex> lk(g_db_mtx);
                g_db_feats.push_back(r.feature);
                g_db_names.push_back(stem_of(std::string(f.c_str())));
                ok++;
                break;   /* Use one usable face per reference image. */
            }
        }
    }
    std::printf("[facedb] Loaded %d reference faces from %s\n", ok, dir.c_str());
    load_registered_faces();
}

struct FaceDraw {
    cv::Rect              box;
    float                 conf;
    float                 yaw, pitch, roll;
    std::vector<cv::Point> pts;
    std::string           label;
    float                 score;
    bool                  matched;
};

static void draw_results(cv::Mat& frame, const std::vector<FaceDraw>& faces,
                         double fps, const char* mode_name, int db_count) {
    for (const auto& f : faces) {
        cv::Scalar color = f.matched ? cv::Scalar(0, 255, 0) : cv::Scalar(0, 200, 255);
        cv::rectangle(frame, f.box, color, 2);

        if (g_show_landmark.load())
            for (const auto& p : f.pts) cv::circle(frame, p, 1, cv::Scalar(0, 255, 0), 2);

        char t[160];
        if (f.score >= 0.0f) std::snprintf(t, sizeof(t), "%s %.2f", f.label.c_str(), f.score);
        else                 std::snprintf(t, sizeof(t), "%s", f.label.c_str());
        int base = 0;
        cv::Size ts = cv::getTextSize(t, cv::FONT_HERSHEY_SIMPLEX, 0.6, 2, &base);
        int ty = std::max(f.box.y, ts.height + 4);
        cv::rectangle(frame, cv::Point(f.box.x, ty - ts.height - 4),
                      cv::Point(f.box.x + ts.width + 4, ty + 2), color, -1);
        cv::putText(frame, t, cv::Point(f.box.x + 2, ty - 2),
                    cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 0, 0), 2);

        char pose[128];
        std::snprintf(pose, sizeof(pose), "Y:%.0f P:%.0f R:%.0f", f.yaw, f.pitch, f.roll);
        cv::putText(frame, pose, cv::Point(f.box.x, f.box.y + f.box.height + 16),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, color, 1);
    }
    cv::rectangle(frame, cv::Point(0, 0), cv::Point(frame.cols, 28), cv::Scalar(0, 0, 0), -1);
    char hud[200];
    if (fps >= 0.0)
        std::snprintf(hud, sizeof(hud), "Mode: %-7s  FPS: %4.1f  Faces: %zu  DB: %d",
                      mode_name, fps, faces.size(), db_count);
    else
        std::snprintf(hud, sizeof(hud), "Mode: %-7s  Faces: %zu  DB: %d",
                      mode_name, faces.size(), db_count);
    cv::putText(frame, hud, cv::Point(8, 20), cv::FONT_HERSHEY_SIMPLEX, 0.6,
                cv::Scalar(0, 255, 255), 1);
}

/* Copy pixels so the GUI image outlives the worker's frame buffer. */
static void publish(const cv::Mat& bgr) {
    cv::Mat rgb;
    cv::cvtColor(bgr, rgb, cv::COLOR_BGR2RGB);
    QImage qi((const uchar*)rgb.data, rgb.cols, rgb.rows, (int)rgb.step, QImage::Format_RGB888);
    std::lock_guard<std::mutex> lk(g_mtx);
    g_img = qi.copy();
    g_fid++;
}

/* Save the largest usable face before reporting registration success. */
static void process_frame(FaceEngine* engine, cv::Mat& frame, std::vector<FaceDraw>& out,
                          bool do_register) {
    std::vector<FaceResults_t> results;
    engine->inference(frame.data, frame.rows, frame.cols, frame.channels(), results);

    /* Match against a snapshot to avoid racing with database updates. */
    std::vector<std::vector<float>> db;
    std::vector<std::string>        names;
    { std::lock_guard<std::mutex> lk(g_db_mtx); db = g_db_feats; names = g_db_names; }

    if (do_register && g_register_pending.load()) {
        int best = -1; long best_area = 0;
        for (size_t i = 0; i < results.size(); ++i) {
            const auto& r = results[i];
            if (r.code != FACE_OK || r.feature.empty()) continue;
            long area = (long)r.width * r.height;
            if (area > best_area) { best_area = area; best = (int)i; }
        }
        if (best >= 0) {
            std::string nm;
            { std::lock_guard<std::mutex> lk(g_cmd_mtx); nm = g_register_name; }
            bool saved = false;
            try {
                saved = save_registered_face(nm, frame, results[best]);
            } catch (const std::exception& error) {
                std::printf("[reg] Save failed: %s\n", error.what());
            }
            if (saved) {
                std::lock_guard<std::mutex> lk(g_db_mtx);
                g_db_feats.push_back(results[best].feature);
                g_db_names.push_back(nm);
                db = g_db_feats; names = g_db_names;   /* Match the newly registered face immediately. */
                g_register_result = 1;
                std::printf("[reg] Face saved. Database entries: %zu\n", names.size());
            } else {
                g_register_result = -2;
                std::printf("[reg] Unable to save facedb/<name>.jpg. Check the name, existing file, and write access.\n");
            }
        } else {
            std::printf("[reg] No usable face detected for registration.\n");
            g_register_result = -1;
        }
        g_register_pending = false;
    }

    out.clear();
    for (const auto& r : results) {
        FaceDraw fd;
        fd.box   = cv::Rect(r.x, r.y, r.width, r.height);
        fd.conf  = r.conf;
        fd.yaw   = r.yaw; fd.pitch = r.pitch; fd.roll = r.roll;
        for (const auto& p : r.key_pts) fd.pts.emplace_back(p.x, p.y);
        fd.matched = false;
        fd.score   = -1.0f;

        if (r.code != FACE_OK) {
            fd.label = "low-quality";          /* Detection exists, but the SDK rejected the face. */
        } else if (r.feature.empty() || db.empty()) {
            fd.label = "Unknown";
        } else {
            std::vector<float> feat = r.feature;
            std::pair<int, float> m = engine->cosine_similarity(db, feat, g_match_thresh);
            fd.score = m.second;
            if (m.first >= 0 && m.first < (int)names.size()) {
                fd.label = names[m.first];
                fd.matched = true;
            } else {
                fd.label = "Unknown";
            }
        }
        out.push_back(std::move(fd));
    }
}

static void worker_loop(FaceEngine* engine) {
    cv::VideoCapture cap;
    bool cam_open = false;

    auto t_last = std::chrono::steady_clock::now();
    double fps  = 0.0;

    while (g_running) {
        int mode = g_mode.load();

        if (mode != MODE_CAMERA && cam_open) { cap.release(); cam_open = false; }

        if (mode == MODE_CAMERA) {
            if (!cam_open) {
                std::string dev = g_cam_dev;
                bool ok = false;
                if (dev.empty() || dev == "auto") {
                    dev = find_usb_camera();
                    if (!dev.empty()) ok = cap.open(dev, cv::CAP_V4L2);
                } else if (dev.find_first_not_of("0123456789") == std::string::npos) {
                    ok = cap.open(std::atoi(dev.c_str()), cv::CAP_V4L2);
                } else {
                    ok = cap.open(dev, cv::CAP_V4L2);
                }
                if (!ok || !cap.isOpened()) {
                    std::string alt = find_usb_camera();
                    if (!alt.empty() && alt != dev) { dev = alt; ok = cap.open(dev, cv::CAP_V4L2); }
                }
                if (!ok || !cap.isOpened()) {
                    std::printf("[cam] Unable to open a camera. Check the USB camera connection.\n");
                    g_mode = MODE_IDLE;
                    std::this_thread::sleep_for(std::chrono::milliseconds(200));
                    continue;
                }
                g_cam_dev = dev;
                cap.set(cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'));
                cap.set(cv::CAP_PROP_FRAME_WIDTH, 640);
                cap.set(cv::CAP_PROP_FRAME_HEIGHT, 480);
                cap.set(cv::CAP_PROP_FPS, 30);
                cam_open = true;
                fps = 0.0;
                t_last = std::chrono::steady_clock::now();
                std::printf("[cam] Opened %s (%.0fx%.0f)\n", g_cam_dev.c_str(),
                            cap.get(cv::CAP_PROP_FRAME_WIDTH), cap.get(cv::CAP_PROP_FRAME_HEIGHT));
            }

            /* Keep the registration frame stable until confirmation or cancellation. */
            if (g_frozen.load()) {
                cv::Mat fr;
                { std::lock_guard<std::mutex> lk(g_frozen_mtx); fr = g_frozen_mat.clone(); }
                int dbc; { std::lock_guard<std::mutex> lk(g_db_mtx); dbc = (int)g_db_feats.size(); }
                if (g_unfreeze_request.exchange(false)) {
                    g_frozen = false;
                } else if (g_register_pending.load()) {
                    std::vector<FaceDraw> faces;
                    process_frame(engine, fr, faces, true);
                    draw_results(fr, faces, -1.0, "Register", dbc);
                    publish(fr);
                    g_frozen = false;
                } else {
                    std::vector<FaceDraw> faces;
                    process_frame(engine, fr, faces, false);
                    draw_results(fr, faces, -1.0, "Paused", dbc);
                    publish(fr);
                    std::this_thread::sleep_for(std::chrono::milliseconds(80));
                }
                continue;
            }

            cv::Mat frame;
            if (!cap.read(frame) || frame.empty()) {
                std::this_thread::sleep_for(std::chrono::milliseconds(5));
                continue;
            }
            if (frame.channels() != 3) cv::cvtColor(frame, frame, cv::COLOR_GRAY2BGR);
            if (!frame.isContinuous()) frame = frame.clone();

            /* Enrollment uses the same unannotated frame shown during the pause. */
            if (g_freeze_request.exchange(false)) {
                { std::lock_guard<std::mutex> lk(g_frozen_mtx); g_frozen_mat = frame.clone(); }
                g_frozen = true;
                continue;
            }

            auto t_now = std::chrono::steady_clock::now();
            double dt = std::chrono::duration<double>(t_now - t_last).count();
            t_last = t_now;
            if (dt > 0) fps = 0.9 * fps + 0.1 * (1.0 / dt);

            std::vector<FaceDraw> faces;
            process_frame(engine, frame, faces, false);
            int dbc; { std::lock_guard<std::mutex> lk(g_db_mtx); dbc = (int)g_db_feats.size(); }
            draw_results(frame, faces, fps, "Camera", dbc);
            publish(frame);

        } else if (mode == MODE_IMAGE) {
            if (g_image_dirty.exchange(false)) {
                std::string path;
                { std::lock_guard<std::mutex> lk(g_cmd_mtx); path = g_image_path; }
                cv::Mat img = cv::imread(path, cv::IMREAD_COLOR);
                if (img.empty()) {
                    std::printf("[img] Unable to read image: %s\n", path.c_str());
                } else {
                    if (!img.isContinuous()) img = img.clone();
                    std::vector<FaceDraw> faces;
                    process_frame(engine, img, faces, true);
                    int dbc; { std::lock_guard<std::mutex> lk(g_db_mtx); dbc = (int)g_db_feats.size(); }
                    std::printf("[img] %s -> faces=%zu\n", path.c_str(), faces.size());
                    draw_results(img, faces, -1.0, "Image", dbc);
                    publish(img);
                }
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(30));
        } else {
            std::this_thread::sleep_for(std::chrono::milliseconds(30));
        }
    }
    if (cam_open) cap.release();
    std::printf("[worker] stopped.\n");
}

/* Read usage statistics from procfs and the root filesystem. */
struct MemInfo { double used_pct; long used_mb; long total_mb; };

static MemInfo read_mem() {
    long total = 0, avail = 0;
    if (FILE* f = std::fopen("/proc/meminfo", "r")) {
        char key[64], unit[16]; long val;
        while (std::fscanf(f, "%63s %ld %15s", key, &val, unit) == 3) {
            if (std::strcmp(key, "MemTotal:") == 0)           total = val;
            else if (std::strcmp(key, "MemAvailable:") == 0) { avail = val; break; }
        }
        std::fclose(f);
    }
    MemInfo m{0, 0, 0};
    if (total > 0) {
        long used = total - avail;
        m.total_mb = total / 1024;
        m.used_mb  = used / 1024;
        m.used_pct = 100.0 * used / total;
    }
    return m;
}

static double read_cpu() {
    static long prev_total = 0, prev_idle = 0;
    FILE* f = std::fopen("/proc/stat", "r");
    if (!f) return 0.0;
    char cpu[8]; long u = 0, n = 0, s = 0, i = 0, io = 0, irq = 0, sirq = 0, st = 0;
    int got = std::fscanf(f, "%7s %ld %ld %ld %ld %ld %ld %ld %ld",
                          cpu, &u, &n, &s, &i, &io, &irq, &sirq, &st);
    std::fclose(f);
    if (got < 5) return 0.0;
    long idle = i + io;
    long total = u + n + s + i + io + irq + sirq + st;
    long dt = total - prev_total, di = idle - prev_idle;
    prev_total = total; prev_idle = idle;
    if (dt <= 0) return 0.0;
    double usage = 100.0 * (dt - di) / dt;
    return usage < 0 ? 0 : (usage > 100 ? 100 : usage);
}

static MemInfo read_rom() {
    MemInfo r{0, 0, 0};
    struct statvfs vfs;
    if (statvfs("/", &vfs) == 0 && vfs.f_blocks > 0) {
        unsigned long long bs = vfs.f_frsize ? vfs.f_frsize : vfs.f_bsize;
        unsigned long long total = (unsigned long long)vfs.f_blocks * bs;
        unsigned long long avail = (unsigned long long)vfs.f_bavail * bs;
        unsigned long long used = total - avail;
        r.total_mb = (long)(total / (1024 * 1024));
        r.used_mb  = (long)(used  / (1024 * 1024));
        r.used_pct = 100.0 * used / total;
    }
    return r;
}

static bool request_registration_name(QWidget* parent, QString& name) {
    QDialog dialog(parent);
    dialog.setObjectName("faceRegistrationDialog");
    dialog.setWindowTitle("Register Face");
    dialog.setMinimumWidth(420);
    dialog.setStyleSheet("QDialog#faceRegistrationDialog{background-color:#ffffff;}");

    QLabel label("Enter name / ID:", &dialog);
    label.setStyleSheet("background-color:#ffffff;color:#111827;font-size:16px;");
    QLineEdit input(&dialog);
    input.setObjectName("registrationName");
    input.setMinimumHeight(44);
    input.setStyleSheet(
        "QLineEdit{background-color:#ffffff;color:#111827;border:2px solid #6b7280;"
        "border-radius:4px;padding:8px;font-size:18px;"
        "selection-background-color:#245bb5;selection-color:#ffffff;}"
        "QLineEdit:focus{border-color:#245bb5;}");

    QPushButton registerButton("Register", &dialog);
    registerButton.setObjectName("registrationConfirm");
    registerButton.setDefault(true);
    registerButton.setEnabled(false);
    registerButton.setMinimumHeight(44);
    registerButton.setStyleSheet(
        "QPushButton{background-color:#245bb5;color:#ffffff;border:2px solid #245bb5;"
        "border-radius:4px;padding:8px 18px;font-size:16px;font-weight:bold;}"
        "QPushButton:hover{background-color:#19478f;}"
        "QPushButton:pressed{background-color:#12376f;}"
        "QPushButton:focus{border-color:#111827;}"
        "QPushButton:disabled{background-color:#e5e7eb;color:#4b5563;border-color:#9ca3af;}");
    QPushButton cancelButton("Cancel", &dialog);
    cancelButton.setObjectName("registrationCancel");
    cancelButton.setAutoDefault(false);
    cancelButton.setMinimumHeight(44);
    cancelButton.setStyleSheet(
        "QPushButton{background-color:#f3f4f6;color:#111827;border:2px solid #6b7280;"
        "border-radius:4px;padding:8px 18px;font-size:16px;}"
        "QPushButton:hover{background-color:#e5e7eb;}"
        "QPushButton:pressed{background-color:#d1d5db;}"
        "QPushButton:focus{border-color:#245bb5;}");

    /* Use a local style and per-widget CSS to isolate controls from desktop themes. */
    if (QStyle* style = QStyleFactory::create("Fusion")) {
        style->setParent(&dialog);
        dialog.setStyle(style);
        label.setStyle(style);
        input.setStyle(style);
        registerButton.setStyle(style);
        cancelButton.setStyle(style);
    }
    QHBoxLayout* buttons = new QHBoxLayout();
    buttons->addStretch();
    buttons->addWidget(&registerButton);
    buttons->addWidget(&cancelButton);
    QVBoxLayout layout(&dialog);
    layout.setContentsMargins(20, 20, 20, 20);
    layout.setSpacing(14);
    layout.addWidget(&label);
    layout.addWidget(&input);
    layout.addLayout(buttons);
    QObject::connect(&input, &QLineEdit::textChanged, &dialog, [&](const QString& value) {
        registerButton.setEnabled(!value.trimmed().isEmpty());
    });
    QObject::connect(&registerButton, &QPushButton::clicked, &dialog, &QDialog::accept);
    QObject::connect(&cancelButton, &QPushButton::clicked, &dialog, &QDialog::reject);
    input.setFocus();
    const bool accepted = dialog.exec() == QDialog::Accepted;
    name = input.text().trimmed();
    return accepted && !name.isEmpty();
}

int main(int argc, char** argv) {
    /* Usage: face_cam_demo [cam_dev] [auth_code]
     * cam_dev: device path, numeric index, or "auto" (default).
     * auth_code: argv[2] takes precedence over FACE_AUTH_CODE. */
    if (argc >= 2 && argv[1][0]) g_cam_dev = argv[1];
    const char* auth = (argc >= 3 && argv[2][0]) ? argv[2] : getenv("FACE_AUTH_CODE");

    std::signal(SIGINT, on_sig);
    std::signal(SIGTERM, on_sig);

    FaceEngine engine;

    char det_model[]  = "models/facedet_512_q_3576_enc.rknn";
    char land_model[] = "models/facelandmark_q_3576_enc.rknn";
    char reg_model[]  = "models/facereg_l_q_3576_enc.rknn";
    char license[512] = "face_license.ini";
    if (const char* license_path = std::getenv("FACE_LICENSE_FILE")) {
        if (license_path[0]) std::snprintf(license, sizeof(license), "%s", license_path);
    }

    /* Module 3 identifies RK3576; FACE_MODULE can override it. */
    int module = 3;
    if (const char* m = getenv("FACE_MODULE")) { if (m[0]) module = std::atoi(m); }
    int app_id = 145;
    if (const char* a = getenv("FACE_APP_ID")) { if (a[0]) app_id = std::atoi(a); }

    char auth_buf[512] = {0};
    char* p_auth = auth_buf;   /* Let the SDK handle an empty authorization code. */
    if (auth && auth[0]) std::snprintf(auth_buf, sizeof(auth_buf), "%s", auth);

    FACE_RES_CODE rc = engine.init(det_model, land_model, reg_model,
                                   license, module, p_auth, app_id);
    engine.set_param(0.35f, 0.45f);

    int alg_status  = engine.get_status(0);
    int auth_status = engine.get_status(1);
    std::printf("[init] rc=%d alg_status=%d auth_status=%d module=%d app_id=%d\n",
                (int)rc, alg_status, auth_status, module, app_id);
    if (alg_status != 0) {
        std::printf("[init] Initialization failed. Check FACE_AUTH_CODE, FACE_MODULE, and model paths.\n");
        return 1;
    }

    load_facedb(engine, "facedb");

    std::thread worker(worker_loop, &engine);

    QApplication app(argc, argv);
    QWidget win;
    win.setWindowTitle("Face Recognition");
    win.setStyleSheet("background:#1e1e2a;");

    QFrame* left = new QFrame(&win);
    left->setFixedWidth(210);
    left->setStyleSheet("QFrame{background:#262636;border-radius:10px;}");
    QVBoxLayout* leftV = new QVBoxLayout(left);
    leftV->setContentsMargins(12, 12, 12, 12);
    QLabel* leftTitle = new QLabel("Modules", left);
    leftTitle->setStyleSheet("color:#ffffff;font-size:17px;font-weight:bold;");
    leftV->addWidget(leftTitle);
    QLabel* mod = new QLabel("01   Face Recognition", left);
    mod->setStyleSheet("color:#ffffff;background:#4a3fb8;border-radius:8px;"
                       "padding:14px 12px;font-size:14px;");
    leftV->addWidget(mod);
    QLabel* caps = new QLabel("· Face Detection\n· 21 Landmarks\n· Head Pose\n· 1:N Recognition", left);
    caps->setStyleSheet("color:#b8b8cc;font-size:13px;padding:8px 10px;line-height:160%;");
    leftV->addWidget(caps);
    leftV->addStretch();

    QLabel* title = new QLabel("Face Recognition", &win);
    title->setStyleSheet("color:#ffffff;font-size:22px;font-weight:bold;");
    QLabel* subtitle = new QLabel("Detection · 21 Landmarks · Head Pose · 1:N Recognition", &win);
    subtitle->setStyleSheet("color:#9a9ab0;font-size:13px;");

    QLabel* view = new QLabel(&win);
    view->setAlignment(Qt::AlignCenter);
    view->setMinimumSize(640, 480);
    view->setSizePolicy(QSizePolicy::Ignored, QSizePolicy::Ignored);
    view->setStyleSheet("background:#202030;color:#cccccc;border-radius:10px;");
    view->setText("Click [Open Camera] for live recognition, or [Local Image] to pick a picture;\n"
                  "face the camera and click [Register Face] to enroll");

    QPushButton* btnCam  = new QPushButton("Open Camera", &win);
    QPushButton* btnImg  = new QPushButton("Local Image", &win);
    QPushButton* btnReg  = new QPushButton("Register Face", &win);
    QPushButton* btnLmk  = new QPushButton("Landmarks", &win);
    QPushButton* btnFull = new QPushButton("Full Screen", &win);
    QPushButton* btnQuit = new QPushButton("Quit", &win);
    const char* btnStyle =
        "QPushButton{background:#3a3a4e;color:#fff;border:1px solid #55557a;"
        "border-radius:6px;padding:10px 16px;font-size:16px;}"
        "QPushButton:hover{background:#4a4a63;}"
        "QPushButton:checked{background:#2e7d32;}";
    btnCam->setCheckable(true);
    btnLmk->setCheckable(true);
    btnLmk->setChecked(true);
    btnCam->setStyleSheet(btnStyle);
    btnImg->setStyleSheet(btnStyle);
    btnReg->setStyleSheet(btnStyle);
    btnLmk->setStyleSheet(btnStyle);
    btnFull->setStyleSheet(btnStyle);
    btnQuit->setStyleSheet(btnStyle);

    QGridLayout* bar = new QGridLayout();
    bar->addWidget(btnCam, 0, 0);
    bar->addWidget(btnImg, 0, 1);
    bar->addWidget(btnReg, 0, 2);
    bar->addWidget(btnLmk, 1, 0);
    bar->addWidget(btnFull, 1, 1);
    bar->addWidget(btnQuit, 1, 2);
    for (int column = 0; column < 3; ++column) bar->setColumnStretch(column, 1);

    QVBoxLayout* centerV = new QVBoxLayout();
    centerV->setContentsMargins(4, 0, 4, 0);
    centerV->addWidget(title);
    centerV->addWidget(subtitle);
    centerV->addWidget(view, 1);
    centerV->addLayout(bar);

    const char* statLabStyle = "color:#cfcfe0;font-size:13px;";
    const char* statTxtStyle = "color:#9a9ab0;font-size:12px;";
    const char* barStyle =
        "QProgressBar{background:#15151f;border:none;border-radius:6px;"
        "min-height:12px;max-height:12px;}"
        "QProgressBar::chunk{background:#6c5ce7;border-radius:6px;}";

    QFrame* right = new QFrame(&win);
    right->setFixedWidth(230);
    right->setStyleSheet("QFrame{background:#262636;border-radius:10px;}");
    QVBoxLayout* rightV = new QVBoxLayout(right);
    rightV->setContentsMargins(14, 14, 14, 14);
    QLabel* sysTitle = new QLabel("Device Status", right);
    sysTitle->setStyleSheet("color:#ffffff;font-size:17px;font-weight:bold;");
    rightV->addWidget(sysTitle);

    QLabel* dbLab = new QLabel("Face DB", right);  dbLab->setStyleSheet(statLabStyle);
    QLabel* dbTxt = new QLabel("-", right);        dbTxt->setStyleSheet(statTxtStyle);
    rightV->addWidget(dbLab); rightV->addWidget(dbTxt);

    QLabel* ramLab = new QLabel("RAM", right);  ramLab->setStyleSheet(statLabStyle);
    QProgressBar* ramBar = new QProgressBar(right);
    ramBar->setRange(0, 100); ramBar->setTextVisible(false); ramBar->setStyleSheet(barStyle);
    QLabel* ramTxt = new QLabel("-", right);  ramTxt->setStyleSheet(statTxtStyle);
    rightV->addWidget(ramLab); rightV->addWidget(ramBar); rightV->addWidget(ramTxt);

    QLabel* romLab = new QLabel("ROM", right);  romLab->setStyleSheet(statLabStyle);
    QProgressBar* romBar = new QProgressBar(right);
    romBar->setRange(0, 100); romBar->setTextVisible(false);
    romBar->setStyleSheet("QProgressBar{background:#15151f;border:none;border-radius:6px;"
                          "min-height:12px;max-height:12px;}"
                          "QProgressBar::chunk{background:#0984e3;border-radius:6px;}");
    QLabel* romTxt = new QLabel("-", right);  romTxt->setStyleSheet(statTxtStyle);
    rightV->addWidget(romLab); rightV->addWidget(romBar); rightV->addWidget(romTxt);

    QLabel* cpuLab = new QLabel("CPU", right);  cpuLab->setStyleSheet(statLabStyle);
    QProgressBar* cpuBar = new QProgressBar(right);
    cpuBar->setRange(0, 100); cpuBar->setTextVisible(false);
    cpuBar->setStyleSheet("QProgressBar{background:#15151f;border:none;border-radius:6px;"
                          "min-height:12px;max-height:12px;}"
                          "QProgressBar::chunk{background:#00b894;border-radius:6px;}");
    QLabel* cpuTxt = new QLabel("-", right);  cpuTxt->setStyleSheet(statTxtStyle);
    rightV->addWidget(cpuLab); rightV->addWidget(cpuBar); rightV->addWidget(cpuTxt);

    rightV->addStretch();

    QHBoxLayout* root = new QHBoxLayout(&win);
    root->setContentsMargins(12, 12, 12, 12);
    root->setSpacing(12);
    root->addWidget(left);
    root->addLayout(centerV, 1);
    root->addWidget(right);
    win.resize(1180, 720);
    win.show();

    QObject::connect(btnCam, &QPushButton::toggled, [&](bool on) {
        if (on) { g_mode = MODE_CAMERA; btnCam->setText("Close Camera"); }
        else {
            g_mode = MODE_IDLE;
            btnCam->setText("Open Camera");
            view->clear();
            view->setText("Click [Open Camera] for live recognition, or [Local Image] to pick a picture;\n"
                          "face the camera and click [Register Face] to enroll");
        }
    });
    QObject::connect(btnImg, &QPushButton::clicked, [&]() {
        /* Override the parent's dark theme for readable filenames. */
        QFileDialog dlg(&win, "Select Image", QDir::homePath(), "Images (*.jpg *.jpeg *.png *.bmp)");
        dlg.setOption(QFileDialog::DontUseNativeDialog, true);
        dlg.setFileMode(QFileDialog::ExistingFile);
        dlg.setStyleSheet(
            "QWidget{background:#f5f5f5;color:#202020;}"
            "QLineEdit,QComboBox,QListView,QTreeView,QAbstractItemView{background:#ffffff;color:#202020;}"
            "QPushButton{background:#e6e6e6;color:#202020;border:1px solid #bbb;border-radius:4px;padding:4px 12px;}"
            "QPushButton:hover{background:#d4d4d4;}");
        if (dlg.exec() != QDialog::Accepted) return;
        QStringList sel = dlg.selectedFiles();
        QString file = sel.isEmpty() ? QString() : sel.first();
        if (file.isEmpty()) return;
        if (btnCam->isChecked()) btnCam->setChecked(false);
        { std::lock_guard<std::mutex> lk(g_cmd_mtx); g_image_path = file.toStdString(); }
        g_image_dirty = true;
        g_mode = MODE_IMAGE;
    });
    QObject::connect(btnReg, &QPushButton::clicked, [&]() {
        int mode = g_mode.load();
        if (mode == MODE_IDLE) {
            QMessageBox::information(&win, "Register Face",
                "Open the camera or a local image first, and make sure a face is visible");
            return;
        }
        /* Pause capture while the user enters the registration name. */
        if (mode == MODE_CAMERA) {
            g_unfreeze_request = false;
            g_freeze_request = true;
        }
        QString name;
        if (!request_registration_name(&win, name)) {
            if (mode == MODE_CAMERA) g_unfreeze_request = true;
            return;
        }
        { std::lock_guard<std::mutex> lk(g_cmd_mtx); g_register_name = name.trimmed().toStdString(); }
        g_register_result = 0;
        g_register_pending = true;
        if (mode == MODE_IMAGE) g_image_dirty = true;   /* Reprocess the selected image for enrollment. */
    });
    QObject::connect(btnLmk, &QPushButton::toggled, [&](bool on) {
        g_show_landmark = on;
    });
    QObject::connect(btnFull, &QPushButton::clicked, [&]() {
        if (win.isFullScreen()) { win.showNormal(); btnFull->setText("Full Screen"); }
        else                    { win.showFullScreen(); btnFull->setText("Exit Full Screen"); }
    });
    QObject::connect(btnQuit, &QPushButton::clicked, &win, &QWidget::close);

    QObject::connect(new QShortcut(QKeySequence(Qt::Key_Escape), &win),
                     &QShortcut::activated, &win, &QWidget::close);
    QObject::connect(new QShortcut(QKeySequence(Qt::Key_Q), &win),
                     &QShortcut::activated, &win, &QWidget::close);
    QObject::connect(new QShortcut(QKeySequence(Qt::Key_F), &win),
                     &QShortcut::activated, btnFull, &QPushButton::click);

    uint64_t last_fid = 0;
    QTimer timer;
    timer.setInterval(25);
    QObject::connect(&timer, &QTimer::timeout, [&]() {
        if (!g_running) { qApp->quit(); return; }
        int rr = g_register_result.exchange(0);
        if (rr == 1)       view->setToolTip("Registered and saved");
        else if (rr == -1) QMessageBox::information(&win, "Register Face", "No registrable face detected. Please face the camera and retry.");
        else if (rr == -2) QMessageBox::warning(&win, "Register Face", "Unable to save facedb/<name>.jpg. Use a unique name without path characters, and check disk space and write access.");
        uint64_t fid = g_fid.load();
        if (fid == last_fid) return;
        last_fid = fid;
        if (g_mode.load() == MODE_IDLE) return;
        QImage img;
        { std::lock_guard<std::mutex> lk(g_mtx); img = g_img; }
        if (img.isNull()) return;
        view->setPixmap(QPixmap::fromImage(img).scaled(
            view->size(), Qt::KeepAspectRatio, Qt::SmoothTransformation));
    });
    timer.start();

    QTimer statsTimer;
    statsTimer.setInterval(1000);
    QObject::connect(&statsTimer, &QTimer::timeout, [&]() {
        int dbc; { std::lock_guard<std::mutex> lk(g_db_mtx); dbc = (int)g_db_feats.size(); }
        dbTxt->setText(QString("%1 faces").arg(dbc));

        MemInfo r = read_mem();
        int rp = (int)(r.used_pct + 0.5);
        ramBar->setValue(rp);
        ramTxt->setText(QString("%1%  ·  %2 / %3 MB").arg(rp).arg(r.used_mb).arg(r.total_mb));

        MemInfo d = read_rom();
        int dp = (int)(d.used_pct + 0.5);
        romBar->setValue(dp);
        romTxt->setText(QString("%1%  ·  %2 / %3 GB").arg(dp)
                        .arg(d.used_mb / 1024.0, 0, 'f', 1).arg(d.total_mb / 1024.0, 0, 'f', 1));

        int cp = (int)(read_cpu() + 0.5);
        cpuBar->setValue(cp);
        cpuTxt->setText(QString("%1%").arg(cp));
    });
    statsTimer.start();

    QObject::connect(&app, &QApplication::aboutToQuit, [&]() { g_running = false; });

    int ret = app.exec();

    g_running = false;
    if (worker.joinable()) worker.join();
    engine.uinit();
    std::printf("[exit] done\n");
    return ret;
}
