/*
 * gesture_cam_demo.cpp - Gesture Recognition Demo (RK3576 / Debian aarch64)
 *
 * UI: Three-column Qt5 layout with full-screen support.
 * Flow: USB camera or local image -> Gesture inference -> overlay -> Qt display.
 * Threads: Worker captures, infers, and draws; the GUI thread displays shared QImage frames.
 */
#include "gesture.h"

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/videoio.hpp>

#include <QApplication>
#include <QFileDialog>
#include <QFrame>
#include <QHBoxLayout>
#include <QImage>
#include <QKeySequence>
#include <QLabel>
#include <QPixmap>
#include <QProgressBar>
#include <QPushButton>
#include <QShortcut>
#include <QString>
#include <QTimer>
#include <QVBoxLayout>
#include <QWidget>

#include <algorithm>
#include <atomic>
#include <chrono>
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
static std::mutex            g_cmd_mtx;
static std::string           g_image_path;       /* Protected by g_cmd_mtx. */
static std::string           g_cam_dev = "auto";  /* "auto", /dev/videoN, or a numeric index. */

static void on_sig(int) { g_running = false; }

/* RK boards expose ISP/VPSS nodes, so select a USB node with video-capture capability. */
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

static void draw_results(cv::Mat& frame, const std::vector<DetRes>& results,
                         double fps, const char* mode_name) {
    for (const auto& r : results) {
        cv::rectangle(frame, cv::Rect(r.x, r.y, r.width, r.height),
                      cv::Scalar(0, 255, 0), 2);
        char t[128];
        std::snprintf(t, sizeof(t), "%s %.2f", r.label, r.cls_score);
        int base = 0;
        cv::Size ts = cv::getTextSize(t, cv::FONT_HERSHEY_SIMPLEX, 0.6, 2, &base);
        int ty = std::max(r.y, ts.height + 4);
        cv::rectangle(frame, cv::Point(r.x, ty - ts.height - 4),
                      cv::Point(r.x + ts.width + 4, ty + 2), cv::Scalar(0, 255, 0), -1);
        cv::putText(frame, t, cv::Point(r.x + 2, ty - 2),
                    cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 0, 0), 2);
    }
    cv::rectangle(frame, cv::Point(0, 0), cv::Point(frame.cols, 28), cv::Scalar(0, 0, 0), -1);
    char hud[160];
    if (fps >= 0.0)
        std::snprintf(hud, sizeof(hud), "Mode: %-8s  FPS: %4.1f  Hands: %zu",
                      mode_name, fps, results.size());
    else
        std::snprintf(hud, sizeof(hud), "Mode: %-8s  Hands: %zu", mode_name, results.size());
    cv::putText(frame, hud, cv::Point(8, 20), cv::FONT_HERSHEY_SIMPLEX, 0.6,
                cv::Scalar(0, 255, 255), 1);
}

/* Convert BGR to an owned RGB image before sharing it with the GUI thread. */
static void publish(const cv::Mat& bgr) {
    cv::Mat rgb;
    cv::cvtColor(bgr, rgb, cv::COLOR_BGR2RGB);
    QImage qi((const uchar*)rgb.data, rgb.cols, rgb.rows, (int)rgb.step, QImage::Format_RGB888);
    std::lock_guard<std::mutex> lk(g_mtx);
    g_img = qi.copy();
    g_fid++;
}

static void worker_loop(Gesture* g) {
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
                    std::string alt = find_usb_camera();  /* Fall back to USB-node detection. */
                    if (!alt.empty() && alt != dev) { dev = alt; ok = cap.open(dev, cv::CAP_V4L2); }
                }
                if (!ok || !cap.isOpened()) {
                    std::printf("[cam] Unable to open a camera. Check that a USB camera is connected.\n");
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
            cv::Mat frame;
            if (!cap.read(frame) || frame.empty()) {
                std::this_thread::sleep_for(std::chrono::milliseconds(5));
                continue;
            }
            if (frame.channels() != 3) cv::cvtColor(frame, frame, cv::COLOR_GRAY2BGR);
            if (!frame.isContinuous()) frame = frame.clone();

            auto t_now = std::chrono::steady_clock::now();
            double dt = std::chrono::duration<double>(t_now - t_last).count();
            t_last = t_now;
            if (dt > 0) fps = 0.9 * fps + 0.1 * (1.0 / dt);

            std::vector<DetRes> results;
            g->inference(frame.data, frame.rows, frame.cols, frame.channels(), results);
            draw_results(frame, results, fps, "Camera");
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
                    std::vector<DetRes> results;
                    g->inference(img.data, img.rows, img.cols, img.channels(), results);
                    std::printf("[img] %s -> detected hands: %zu\n", path.c_str(), results.size());
                    draw_results(img, results, -1.0, "Image");
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

/* Read device usage statistics from procfs and the root filesystem. */
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

/* Return total CPU usage since the previous sample; the first sample is zero. */
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

int main(int argc, char** argv) {
    /* Usage: gesture_cam_demo [cam_dev] [auth_code]
     * cam_dev: USB camera path, numeric index, or "auto" (default).
     * auth_code: argv[2] takes precedence over GESTURE_AUTH_CODE. */
    if (argc >= 2 && argv[1][0]) g_cam_dev = argv[1];
    const char* auth = (argc >= 3 && argv[2][0]) ? argv[2] : getenv("GESTURE_AUTH_CODE");

    std::signal(SIGINT, on_sig);
    std::signal(SIGTERM, on_sig);

    /* Initialize the RK3576 gesture detection and classification pipeline. */
    Gesture g;
    float conf = 0.25f, iou = 0.45f, det_thr = 0.0f, cls_thr = 0.20f;
    g.set_param(conf, iou, det_thr, cls_thr);

    char det_model[] = "models/board6_RK3576_Ubuntu_hand_det.engine";
    char cls_model[] = "models/board6_RK3576_Ubuntu_gesture_cls.engine";
    char license_buf[512] = "authorization.lic";
    if (const char* license_path = std::getenv("GESTURE_LICENSE_FILE")) {
        if (license_path[0]) std::snprintf(license_buf, sizeof(license_buf), "%s", license_path);
    }
    char auth_buf[512] = {0};
    char* p_auth = nullptr;
    if (auth && auth[0]) { std::snprintf(auth_buf, sizeof(auth_buf), "%s", auth); p_auth = auth_buf; }

    GESTURE_RES_CODE rc = g.init(det_model, cls_model, license_buf, 3, p_auth);
    std::printf("[init] rc=%d status_init=%d status_auth=%d\n",
                (int)rc, g.get_status(0), g.get_status(1));
    if (rc != GESTURE_PROCESS_OK) {
        std::printf("[init] Initialization failed. Check GESTURE_AUTH_CODE and model paths.\n");
        return 1;
    }

    std::thread worker(worker_loop, &g);

    QApplication app(argc, argv);
    QWidget win;
    win.setWindowTitle("Gesture Recognition");
    win.setStyleSheet("background:#1e1e2a;");

    QFrame* left = new QFrame(&win);
    left->setFixedWidth(210);
    left->setStyleSheet("QFrame{background:#262636;border-radius:10px;}");
    QVBoxLayout* leftV = new QVBoxLayout(left);
    leftV->setContentsMargins(12, 12, 12, 12);
    QLabel* leftTitle = new QLabel("Modules", left);
    leftTitle->setStyleSheet("color:#ffffff;font-size:17px;font-weight:bold;");
    leftV->addWidget(leftTitle);
    QLabel* mod = new QLabel("01   Gesture Recognition", left);
    mod->setStyleSheet("color:#ffffff;background:#4a3fb8;border-radius:8px;"
                       "padding:14px 12px;font-size:14px;");
    leftV->addWidget(mod);
    leftV->addStretch();

    QLabel* title = new QLabel("Gesture Recognition", &win);
    title->setStyleSheet("color:#ffffff;font-size:22px;font-weight:bold;");
    QLabel* subtitle = new QLabel("Detect hand gestures and output class and confidence", &win);
    subtitle->setStyleSheet("color:#9a9ab0;font-size:13px;");

    QLabel* view = new QLabel(&win);
    view->setAlignment(Qt::AlignCenter);
    view->setMinimumSize(640, 480);
    view->setStyleSheet("background:#202030;color:#cccccc;border-radius:10px;");
    view->setText("Click [Open Camera] for live detection, or [Local Image] to run on a picture");

    QPushButton* btnCam  = new QPushButton("Open Camera", &win);
    QPushButton* btnImg  = new QPushButton("Local Image", &win);
    QPushButton* btnFull = new QPushButton("Full Screen", &win);
    QPushButton* btnQuit = new QPushButton("Quit", &win);
    const char* btnStyle =
        "QPushButton{background:#3a3a4e;color:#fff;border:1px solid #55557a;"
        "border-radius:6px;padding:10px 18px;font-size:16px;}"
        "QPushButton:hover{background:#4a4a63;}"
        "QPushButton:checked{background:#2e7d32;}";
    btnCam->setCheckable(true);
    btnCam->setStyleSheet(btnStyle);
    btnImg->setStyleSheet(btnStyle);
    btnFull->setStyleSheet(btnStyle);
    btnQuit->setStyleSheet(btnStyle);

    QHBoxLayout* bar = new QHBoxLayout();
    bar->addStretch();
    bar->addWidget(btnCam);
    bar->addWidget(btnImg);
    bar->addWidget(btnFull);
    bar->addWidget(btnQuit);
    bar->addStretch();

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
    win.resize(1120, 700);
    win.show();

    QObject::connect(btnCam, &QPushButton::toggled, [&](bool on) {
        if (on) { g_mode = MODE_CAMERA; btnCam->setText("Close Camera"); }
        else {
            g_mode = MODE_IDLE;
            btnCam->setText("Open Camera");
            view->clear();   /* Clear the last frame when camera mode stops. */
            view->setText("Click [Open Camera] for live detection, or [Local Image] to run on a picture");
        }
    });
    QObject::connect(btnImg, &QPushButton::clicked, [&]() {
        /* Use a light non-native dialog so filenames remain readable. */
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
        uint64_t fid = g_fid.load();
        if (fid == last_fid) return;
        last_fid = fid;
        if (g_mode.load() == MODE_IDLE) return;   /* Do not repaint stale frames while idle. */
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
    g.uinit();
    std::printf("[exit] done\n");
    return ret;
}
