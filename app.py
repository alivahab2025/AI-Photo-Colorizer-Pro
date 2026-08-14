import os
import sys
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

import cv2
import requests
import numpy as np
import threading
import collections
import collections.abc
from PIL import Image

import customtkinter as ctk
from tkinter import filedialog, messagebox

# Bypass pkg_resources version checks when frozen with PyInstaller
import pkg_resources
_orig_require = pkg_resources.require
def _patched_require(*args, **kwargs):
    try:
        return _orig_require(*args, **kwargs)
    except pkg_resources.DistributionNotFound:
        return []
pkg_resources.require = _patched_require



# Set CustomTkinter Theme
ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")

# ==============================================================================
# 🎨 COLOR PALETTE & UI CUSTOMIZATION (تنظیمات کامل تم و رنگ‌بندی)
# ==============================================================================
THEME_CONFIG = {
    # --- Background Colors (رنگ‌های پس‌زمینه) ---
    "BG_MAIN": "#141416",  # Main app background / پس‌زمینه اصلی برنامه
    "PANEL_LEFT_BG": "#1a1a1e",  # Left Control Center panel background / پس‌زمینه پنل چپ
    "PANEL_RIGHT_BG": "#141416",  # Right Viewport panel background / پس‌زمینه پنل راست

    # --- Text Colors (رنگ‌های متن عمومی) ---
    "TEXT_TITLE": "#ffffff",  # Main titles text color / رنگ عنوان‌های اصلی
    "TEXT_LABEL": "#e0e0e0",  # Sub-labels text color / رنگ لیبل‌ها و عناوین فرعی
    "TEXT_HARDWARE": "#ff9800",  # Hardware status label color / رنگ وضعیت سخت‌افزار
    "TEXT_STATUS": "#aaaaaa",  # Status bar text color / رنگ متن نوار وضعیت
    "TEXT_DISABLED": "#555555",  # Disabled labels text color / رنگ متن‌های غیرفعال (مرده)

    # --- 1. Open Image Button States (حالت‌های دکمه انتخاب عکس) ---
    "BTN_LOAD_BG": "#2b2d42",  # Active background / پس‌زمینه فعال
    "BTN_LOAD_HOVER": "#3d3f58",  # Hover background / پس‌زمینه هنگام رفتن موس
    "BTN_LOAD_TEXT": "#ffffff",  # Active text color / رنگ متن فعال
    "BTN_LOAD_DIS_BG": "#222226",  # Disabled background / پس‌زمینه غیرفعال
    "BTN_LOAD_DIS_TEXT": "#666666",  # Disabled text color / رنگ متن غیرفعال

    # --- 2. Colorize & Enhance Button States (حالت‌های دکمه شروع پردازش) ---
    "BTN_PROCESS_BG": "#2ba84a",  # Active background (Green) / پس‌زمینه فعال (سبز)
    "BTN_PROCESS_HOVER": "#208b3a",  # Hover background / پس‌زمینه هنگام رفتن موس
    "BTN_PROCESS_TEXT": "#ffffff",  # Active text color / رنگ متن در حالت فعال
    "BTN_PROCESS_DIS_BG": "#1c3823",  # Disabled background (Dimmed Green) / پس‌زمینه در حالت غیرفعال
    "BTN_PROCESS_DIS_TEXT": "#5e8265",  # Disabled text color / رنگ متن دکمه زمانی که غیرفعال است

    # --- 3. Save Result Button States (حالت‌های دکمه ذخیره‌سازی) ---
    "BTN_SAVE_BG": "#1f538d",  # Active background (Blue) / پس‌زمینه فعال (آبی)
    "BTN_SAVE_HOVER": "#14375a",  # Hover background / پس‌زمینه هنگام رفتن موس
    "BTN_SAVE_TEXT": "#ffffff",  # Active text color / رنگ متن در حالت فعال
    "BTN_SAVE_DIS_BG": "#182b40",  # Disabled background / پس‌زمینه غیرفعال
    "BTN_SAVE_DIS_TEXT": "#586f87",  # Disabled text color / رنگ متن غیرفعال

    # --- Interactive Widgets (اسلایدرها و عناصر تعاملی) ---
    "SLIDER_ACTIVE": "#1f6aa5",  # Active slider handle & progress color / رنگ اسلایدر فعال
    "SLIDER_HOVER": "#144870",  # Active slider hover color / رنگ هوور اسلایدر فعال
    "SLIDER_DISABLED": "#333333",  # Disabled slider button color / رنگ دستگیره اسلایدر غیرفعال
    "SLIDER_TRACK_DIS": "#2b2b2b",  # Disabled slider track color / رنگ نوار اسلایدر غیرفعال

    # --- Progress Bar (نوار پیشرفت) ---
    "PROGRESS_BAR": "#2ba84a",  # Progress bar fill color / رنگ نوار پیشرفت
}

# ==============================================================================
# 0. Environment Setup & Compatibility Patches
# ==============================================================================
os.makedirs("models", exist_ok=True)
os.makedirs("dummy", exist_ok=True)
os.makedirs("result_images", exist_ok=True)

for attr in ['Iterable', 'Mapping', 'Sequence', 'MutableMapping', 'Callable', 'Container', 'MutableSet',
             'MutableSequence']:
    if not hasattr(collections, attr) and hasattr(collections.abc, attr):
        setattr(collections, attr, getattr(collections.abc, attr))

try:
    import torch

    if not hasattr(torch, 'typename'):
        torch.typename = lambda obj: obj.__class__.__module__ + '.' + obj.__class__.__name__ if hasattr(obj,
                                                                                                        '__class__') else str(
            type(obj))
except Exception:
    pass

HAS_DEOLDIFY = False
DEOLDIFY_ERROR = ""

try:
    from deoldify import device
    from deoldify.device_id import DeviceId
    from deoldify.visualize import get_image_colorizer

    HAS_DEOLDIFY = True
except Exception as e:
    HAS_DEOLDIFY = False
    DEOLDIFY_ERROR = str(e)

# ==============================================================================
# 1. Model File Paths
# ==============================================================================
MODEL_DIR = "models"
PROTOTXT_PATH = os.path.join(MODEL_DIR, "colorization_deploy_v2.prototxt")
PTS_PATH = os.path.join(MODEL_DIR, "pts_in_hull.npy")
MODEL_PATH_CAFFE = os.path.join(MODEL_DIR, "colorization_release_v2.caffemodel")
MODEL_PATH_ARTISTIC = os.path.join(MODEL_DIR, "ColorizeArtistic_gen.pth")
MODEL_PATH_STABLE = os.path.join(MODEL_DIR, "ColorizeStable_gen.pth")


# ==============================================================================
# 2. AI Model Loaders
# ==============================================================================
def load_caffe_model(progress_callback):
    if not (os.path.exists(PROTOTXT_PATH) and os.path.exists(MODEL_PATH_CAFFE) and os.path.exists(PTS_PATH)):
        raise FileNotFoundError("Caffe model files not found in 'models' directory!")

    progress_callback("Loading Zhang (Caffe) model...")
    try:
        net = cv2.dnn.readNetFromCaffe(PROTOTXT_PATH, MODEL_PATH_CAFFE)
    except AttributeError:
        net = cv2.dnn.readNet(PROTOTXT_PATH, MODEL_PATH_CAFFE)

    pts = np.load(PTS_PATH)
    class8 = net.getLayerId("class8_ab")
    conv8 = net.getLayerId("conv8_313_rh")
    pts = pts.transpose().reshape(2, 313, 1, 1)
    net.getLayer(class8).blobs = [pts.astype(np.float32)]
    net.getLayer(conv8).blobs = [np.full((1, 313, 1, 1), 2.606, dtype="float32")]

    return net


def load_deoldify_model(artistic, selected_device, progress_callback):
    if not HAS_DEOLDIFY:
        raise RuntimeError(f"DeOldify library failed to load:\n{DEOLDIFY_ERROR}")

    target_file = "ColorizeArtistic_gen.pth" if artistic else "ColorizeStable_gen.pth"
    file_path = os.path.join(MODEL_DIR, target_file)

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Model file not found:\n{file_path}")

    if selected_device == "GPU" and torch.cuda.is_available():
        try:
            device.set_gpu(DeviceId.GPU0)
        except Exception:
            pass
    else:
        try:
            device.set_cpu()
        except Exception:
            pass

    model_type = "Artistic" if artistic else "Stable"
    progress_callback(f"Loading DeOldify ({model_type}) model...")

    colorizer = get_image_colorizer(artistic=artistic)
    return colorizer


# ==============================================================================
# 3. Processing Pipeline
# ==============================================================================
def process_caffe(img_path, net):
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img.copy()

    rgb_img = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB).astype("float32") / 255.0
    lab_img = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2LAB)
    l_channel = lab_img[:, :, 0]

    resized_l = cv2.resize(l_channel, (224, 224)) - 50
    net.setInput(cv2.dnn.blobFromImage(resized_l))
    ab_pred = net.forward()[0, :, :, :].transpose((1, 2, 0))

    ab_pred_orig = cv2.resize(ab_pred, (img.shape[1], img.shape[0]))
    ab_pred_orig[:, :, 0] = cv2.bilateralFilter(ab_pred_orig[:, :, 0], d=9, sigmaColor=15, sigmaSpace=15)
    ab_pred_orig[:, :, 1] = cv2.bilateralFilter(ab_pred_orig[:, :, 1], d=9, sigmaColor=15, sigmaSpace=15)

    colorized_lab = np.concatenate((l_channel[:, :, np.newaxis], ab_pred_orig), axis=2)
    colorized_bgr = cv2.cvtColor(colorized_lab, cv2.COLOR_LAB2BGR)
    return (np.clip(colorized_bgr, 0, 1) * 255).astype("uint8")


def process_deoldify(img_path, colorizer, render_factor):
    pil_result = colorizer.get_transformed_image(path=img_path, render_factor=render_factor, watermarked=False)
    img_rgb = np.array(pil_result)
    return cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)


def apply_post_processing(bgr_img, sat, contrast, brightness, sharpness, tonemap, gamma):
    img = bgr_img.astype(np.float32)

    if brightness != 100:
        img = img * (brightness / 100.0)

    if contrast != 100:
        mean = np.mean(img, axis=(0, 1), keepdims=True)
        img = (img - mean) * (contrast / 100.0) + mean

    img = np.clip(img, 0, 255).astype(np.uint8)

    if sat != 100:
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * (sat / 100.0), 0, 255)
        img = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    if sharpness > 100:
        kernel = np.array([[0, -1, 0], [-1, 5 + (sharpness - 100) / 50.0, -1], [0, -1, 0]])
        img = cv2.filter2D(img, -1, kernel)

    if gamma != 100:
        gamma_val = gamma / 100.0
        inv_gamma = 1.0 / max(gamma_val, 0.01)
        table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
        img = cv2.LUT(img, table)

    if tonemap > 0:
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=1.0 + (tonemap / 25.0), tileGridSize=(8, 8))
        l = clahe.apply(l)
        img = cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)

    return img


# ==============================================================================
# 4. Modern GUI Application (CustomTkinter)
# ==============================================================================
class AIColorizerProApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("AI Photo Colorizer & Restorer Pro Studio")
        self.geometry("1300x780")
        self.minsize(1100, 680)
        self.configure(fg_color=THEME_CONFIG["BG_MAIN"])

        self.input_path = None
        self.base_colorized_bgr = None
        self.processed_bgr = None
        self.loaded_model = None
        self.current_model_name = None

        self.setup_ui()

    def setup_ui(self):
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # ---------------- LEFT CONTROL PANEL ----------------
        self.left_panel = ctk.CTkFrame(self, width=320, corner_radius=12, fg_color=THEME_CONFIG["PANEL_LEFT_BG"])
        self.left_panel.grid(row=0, column=0, sticky="nsew", padx=12, pady=12)
        self.left_panel.grid_rowconfigure(12, weight=1)

        # Title Section
        lbl_title = ctk.CTkLabel(
            self.left_panel,
            text="Control Center",
            font=ctk.CTkFont(size=20, weight="bold"),
            text_color=THEME_CONFIG["TEXT_TITLE"]
        )
        lbl_title.pack(pady=(16, 2))

        hardware_type = "GPU (CUDA)" if (HAS_DEOLDIFY and torch.cuda.is_available()) else "CPU"
        self.lbl_hw = ctk.CTkLabel(
            self.left_panel,
            text=f"Hardware: {hardware_type}",
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color=THEME_CONFIG["TEXT_HARDWARE"]
        )
        self.lbl_hw.pack(pady=(0, 12))

        # Open Image Button
        self.btn_load = ctk.CTkButton(
            self.left_panel,
            text="📁 Open Image",
            font=ctk.CTkFont(size=13, weight="bold"),
            fg_color=THEME_CONFIG["BTN_LOAD_BG"],
            hover_color=THEME_CONFIG["BTN_LOAD_HOVER"],
            text_color=THEME_CONFIG["BTN_LOAD_TEXT"],
            height=38,
            corner_radius=8,
            command=self.load_image
        )
        self.btn_load.pack(fill="x", padx=16, pady=4)

        # Model Target Combo
        lbl_mod = ctk.CTkLabel(
            self.left_panel,
            text="Colorization Model Target:",
            font=ctk.CTkFont(size=12),
            text_color=THEME_CONFIG["TEXT_LABEL"]
        )
        lbl_mod.pack(anchor="w", padx=16, pady=(10, 2))
        self.combo_model = ctk.CTkOptionMenu(self.left_panel,
                                             values=["DeOldify Stable", "DeOldify Artistic", "Zhang (Caffe)"],
                                             corner_radius=8, height=34)
        self.combo_model.set("DeOldify Stable")
        self.combo_model.pack(fill="x", padx=16, pady=(0, 6))

        # Render Factor
        lbl_rf = ctk.CTkLabel(
            self.left_panel,
            text="Render Factor (DeOldify Precision):",
            font=ctk.CTkFont(size=12),
            text_color=THEME_CONFIG["TEXT_LABEL"]
        )
        lbl_rf.pack(anchor="w", padx=16, pady=(6, 2))
        self.combo_rf = ctk.CTkOptionMenu(self.left_panel,
                                          values=["15 (Fast)", "28 (Standard)", "35 (High)", "42 (Ultra)"],
                                          corner_radius=8, height=34)
        self.combo_rf.set("28 (Standard)")
        self.combo_rf.pack(fill="x", padx=16, pady=(0, 6))

        # Switch: Advanced Enhancements
        self.sw_adv = ctk.CTkSwitch(
            self.left_panel,
            text="Enable Image Enhancements",
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color=THEME_CONFIG["TEXT_TITLE"],
            command=self.toggle_sliders
        )
        self.sw_adv.pack(anchor="w", padx=16, pady=10)
        self.sw_adv.select()

        # Sliders Frame
        self.frame_sliders = ctk.CTkFrame(self.left_panel, fg_color="transparent")
        self.frame_sliders.pack(fill="x", padx=16, pady=2)

        self.sliders = {}
        self.slider_labels = []
        controls = [
            ("Saturation", 10, 200, 110, "%"),
            ("Contrast", 50, 150, 100, "%"),
            ("Brightness", 50, 150, 100, "%"),
            ("Sharpness", 100, 200, 100, "%"),
            ("Gamma", 10, 200, 100, "(x0.01)")
        ]

        for name, f, t, default, unit in controls:
            lbl_s = ctk.CTkLabel(
                self.frame_sliders,
                text=f"{name} ({default}{unit}):",
                font=ctk.CTkFont(size=11),
                text_color=THEME_CONFIG["TEXT_LABEL"]
            )
            lbl_s.pack(anchor="w", pady=(4, 0))
            self.slider_labels.append(lbl_s)

            s = ctk.CTkSlider(
                self.frame_sliders,
                from_=f,
                to=t,
                number_of_steps=100,
                height=16,
                button_color=THEME_CONFIG["SLIDER_ACTIVE"],
                button_hover_color=THEME_CONFIG["SLIDER_HOVER"],
                progress_color=THEME_CONFIG["SLIDER_ACTIVE"]
            )
            s.set(default)
            s.pack(fill="x", pady=(0, 4))
            s.configure(command=lambda val, name=name, lbl=lbl_s, unit=unit: self.on_slider_move(name, val, lbl, unit))
            self.sliders[name] = s

        # Action Button (Enhance & Colorize)
        self.btn_process = ctk.CTkButton(
            self.left_panel,
            text="⚡ Colorize & Enhance",
            font=ctk.CTkFont(size=14, weight="bold"),
            fg_color=THEME_CONFIG["BTN_PROCESS_BG"],
            hover_color=THEME_CONFIG["BTN_PROCESS_HOVER"],
            text_color=THEME_CONFIG["BTN_PROCESS_TEXT"],
            height=42,
            corner_radius=8,
            command=self.start_thread
        )
        self.btn_process.pack(fill="x", padx=16, pady=(16, 6))

        # Progress Bar
        self.progress = ctk.CTkProgressBar(
            self.left_panel,
            height=8,
            corner_radius=4,
            progress_color=THEME_CONFIG["PROGRESS_BAR"]
        )
        self.progress.set(0)
        self.progress.pack(fill="x", padx=16, pady=4)

        # Status Label
        self.lbl_status = ctk.CTkLabel(
            self.left_panel,
            text="Status: Ready",
            font=ctk.CTkFont(size=12),
            text_color=THEME_CONFIG["TEXT_STATUS"]
        )
        self.lbl_status.pack(pady=4)

        # Save Button
        self.btn_save = ctk.CTkButton(
            self.left_panel,
            text="💾 Save Result",
            font=ctk.CTkFont(size=13, weight="bold"),
            fg_color=THEME_CONFIG["BTN_SAVE_BG"],
            hover_color=THEME_CONFIG["BTN_SAVE_HOVER"],
            text_color=THEME_CONFIG["BTN_SAVE_TEXT"],
            height=38,
            corner_radius=8,
            command=self.save_image
        )
        self.btn_save.pack(fill="x", padx=16, pady=(4, 16))

        # حالت اولیه دکمه‌ها
        self.set_button_state(self.btn_process, False, THEME_CONFIG["BTN_PROCESS_BG"],
                              THEME_CONFIG["BTN_PROCESS_HOVER"], THEME_CONFIG["BTN_PROCESS_TEXT"],
                              THEME_CONFIG["BTN_PROCESS_DIS_BG"], THEME_CONFIG["BTN_PROCESS_DIS_TEXT"])
        self.set_button_state(self.btn_save, False, THEME_CONFIG["BTN_SAVE_BG"], THEME_CONFIG["BTN_SAVE_HOVER"],
                              THEME_CONFIG["BTN_SAVE_TEXT"], THEME_CONFIG["BTN_SAVE_DIS_BG"],
                              THEME_CONFIG["BTN_SAVE_DIS_TEXT"])

        # ---------------- RIGHT VIEWPORT PANEL ----------------
        self.right_panel = ctk.CTkFrame(self, corner_radius=12, fg_color=THEME_CONFIG["PANEL_RIGHT_BG"])
        self.right_panel.grid(row=0, column=1, sticky="nsew", padx=(0, 12), pady=12)
        self.right_panel.grid_rowconfigure(1, weight=1)
        self.right_panel.grid_columnconfigure(0, weight=1)

        # Segmented Control Tabs (Original vs Enhanced Result)
        self.segmented_button = ctk.CTkSegmentedButton(self.right_panel, values=["Enhanced Result", "Original Image"],
                                                       command=self.switch_view)
        self.segmented_button.set("Enhanced Result")
        self.segmented_button.grid(row=0, column=0, pady=12, padx=12, sticky="n")

        # Image Container
        self.img_container = ctk.CTkFrame(self.right_panel, fg_color="transparent")
        self.img_container.grid(row=1, column=0, sticky="nsew", padx=12, pady=(0, 12))
        self.img_container.grid_rowconfigure(0, weight=1)
        self.img_container.grid_columnconfigure(0, weight=1)

        self.lbl_viewport = ctk.CTkLabel(self.img_container, text="No image loaded.", font=ctk.CTkFont(size=16),
                                         text_color="#666666")
        self.lbl_viewport.grid(row=0, column=0, sticky="nsew")

    # ---------------- Helper for Dynamic Button States ----------------
    def set_button_state(self, button, enabled, bg_active, hover_active, text_active, bg_disabled, text_disabled):
        if enabled:
            button.configure(
                state="normal",
                fg_color=bg_active,
                hover_color=hover_active,
                text_color=text_active
            )
        else:
            button.configure(
                state="disabled",
                fg_color=bg_disabled,
                text_color=text_disabled
            )

    # ---------------- UI Callbacks & Visual States ----------------
    def toggle_sliders(self):
        is_active = self.sw_adv.get()
        state = "normal" if is_active else "disabled"

        btn_color = THEME_CONFIG["SLIDER_ACTIVE"] if is_active else THEME_CONFIG["SLIDER_DISABLED"]
        btn_hover = THEME_CONFIG["SLIDER_HOVER"] if is_active else THEME_CONFIG["SLIDER_DISABLED"]
        prog_color = THEME_CONFIG["SLIDER_ACTIVE"] if is_active else THEME_CONFIG["SLIDER_TRACK_DIS"]
        label_color = THEME_CONFIG["TEXT_LABEL"] if is_active else THEME_CONFIG["TEXT_DISABLED"]

        for s in self.sliders.values():
            s.configure(
                state=state,
                button_color=btn_color,
                button_hover_color=btn_hover,
                progress_color=prog_color
            )

        for lbl in self.slider_labels:
            lbl.configure(text_color=label_color)

        if self.base_colorized_bgr is not None:
            if is_active:
                self.update_realtime_preview()
            else:
                self.processed_bgr = self.base_colorized_bgr
                if self.segmented_button.get() == "Enhanced Result":
                    rgb_img = cv2.cvtColor(self.base_colorized_bgr, cv2.COLOR_BGR2RGB)
                    pil_img = Image.fromarray(rgb_img)
                    self.display_pil_image(pil_img)

    def on_slider_move(self, name, val, lbl, unit):
        lbl.configure(text=f"{name} ({int(val)}{unit}):")
        if self.base_colorized_bgr is not None and self.sw_adv.get():
            self.update_realtime_preview()

    def switch_view(self, value):
        if value == "Original Image" and self.input_path:
            pil_img = Image.open(self.input_path)
            self.display_pil_image(pil_img)
        elif value == "Enhanced Result" and self.processed_bgr is not None:
            rgb_img = cv2.cvtColor(self.processed_bgr, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb_img)
            self.display_pil_image(pil_img)

    def update_realtime_preview(self):
        if self.base_colorized_bgr is None:
            return

        if self.sw_adv.get():
            out_bgr = apply_post_processing(
                self.base_colorized_bgr,
                sat=self.sliders["Saturation"].get(),
                contrast=self.sliders["Contrast"].get(),
                brightness=self.sliders["Brightness"].get(),
                sharpness=self.sliders["Sharpness"].get(),
                tonemap=0,
                gamma=self.sliders["Gamma"].get()
            )
        else:
            out_bgr = self.base_colorized_bgr

        self.processed_bgr = out_bgr
        if self.segmented_button.get() == "Enhanced Result":
            rgb_img = cv2.cvtColor(out_bgr, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb_img)
            self.display_pil_image(pil_img)

    def load_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg *.jpeg *.png *.bmp *.webp")])
        if file_path:
            self.input_path = file_path
            pil_img = Image.open(file_path)
            self.display_pil_image(pil_img)
            self.segmented_button.set("Original Image")

            self.set_button_state(self.btn_process, True, THEME_CONFIG["BTN_PROCESS_BG"],
                                  THEME_CONFIG["BTN_PROCESS_HOVER"], THEME_CONFIG["BTN_PROCESS_TEXT"],
                                  THEME_CONFIG["BTN_PROCESS_DIS_BG"], THEME_CONFIG["BTN_PROCESS_DIS_TEXT"])
            self.lbl_status.configure(text="Status: Image Loaded")

    def start_thread(self):
        self.set_button_state(self.btn_process, False, THEME_CONFIG["BTN_PROCESS_BG"],
                              THEME_CONFIG["BTN_PROCESS_HOVER"], THEME_CONFIG["BTN_PROCESS_TEXT"],
                              THEME_CONFIG["BTN_PROCESS_DIS_BG"], THEME_CONFIG["BTN_PROCESS_DIS_TEXT"])
        self.set_button_state(self.btn_load, False, THEME_CONFIG["BTN_LOAD_BG"], THEME_CONFIG["BTN_LOAD_HOVER"],
                              THEME_CONFIG["BTN_LOAD_TEXT"], THEME_CONFIG["BTN_LOAD_DIS_BG"],
                              THEME_CONFIG["BTN_LOAD_DIS_TEXT"])
        self.set_button_state(self.btn_save, False, THEME_CONFIG["BTN_SAVE_BG"], THEME_CONFIG["BTN_SAVE_HOVER"],
                              THEME_CONFIG["BTN_SAVE_TEXT"], THEME_CONFIG["BTN_SAVE_DIS_BG"],
                              THEME_CONFIG["BTN_SAVE_DIS_TEXT"])
        self.progress.set(0.2)

        threading.Thread(target=self.run_processing, daemon=True).start()

    def run_processing(self):
        try:
            model_choice = self.combo_model.get()
            device_choice = "GPU" if "GPU" in self.lbl_hw.cget("text") else "CPU"

            rf_text = self.combo_rf.get()
            render_factor = int(rf_text.split(" ")[0])

            self.update_status("Loading AI Model...")
            self.progress.set(0.4)

            if "Zhang" in model_choice:
                if self.loaded_model is None or self.current_model_name != "Zhang":
                    self.loaded_model = load_caffe_model(self.update_status)
                    self.current_model_name = "Zhang"
                colorized_bgr = process_caffe(self.input_path, self.loaded_model)
            else:
                artistic = ("Artistic" in model_choice)
                model_key = "Artistic" if artistic else "Stable"

                if self.loaded_model is None or self.current_model_name != model_key:
                    self.loaded_model = load_deoldify_model(artistic, device_choice, self.update_status)
                    self.current_model_name = model_key

                colorized_bgr = process_deoldify(self.input_path, self.loaded_model, render_factor)

            self.progress.set(0.8)
            self.update_status("Enhancing image...")

            self.base_colorized_bgr = colorized_bgr

            # به‌روزرسانی نهایی رابط کاربری در نخ اصلی
            self.after(0, self.finish_processing)

        except Exception as e:
            self.update_status(f"Error: {str(e)}")
            self.progress.set(0)
            messagebox.showerror("Processing Error", str(e))
            self.after(0, lambda: self.set_button_state(self.btn_process, True, THEME_CONFIG["BTN_PROCESS_BG"],
                                                        THEME_CONFIG["BTN_PROCESS_HOVER"],
                                                        THEME_CONFIG["BTN_PROCESS_TEXT"],
                                                        THEME_CONFIG["BTN_PROCESS_DIS_BG"],
                                                        THEME_CONFIG["BTN_PROCESS_DIS_TEXT"]))
            self.after(0, lambda: self.set_button_state(self.btn_load, True, THEME_CONFIG["BTN_LOAD_BG"],
                                                        THEME_CONFIG["BTN_LOAD_HOVER"], THEME_CONFIG["BTN_LOAD_TEXT"],
                                                        THEME_CONFIG["BTN_LOAD_DIS_BG"],
                                                        THEME_CONFIG["BTN_LOAD_DIS_TEXT"]))

    def finish_processing(self):
        self.update_realtime_preview()
        self.progress.set(1.0)
        self.update_status("Status: Colorization Complete")

        # سوییچ خودکار تب و نمایش آنی عکس رنگی
        self.segmented_button.set("Enhanced Result")
        self.update_idletasks()

        rgb_img = cv2.cvtColor(self.processed_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb_img)
        self.display_pil_image(pil_img)

        # فعال‌سازی مجدد دکمه‌ها
        self.set_button_state(self.btn_save, True, THEME_CONFIG["BTN_SAVE_BG"], THEME_CONFIG["BTN_SAVE_HOVER"],
                              THEME_CONFIG["BTN_SAVE_TEXT"], THEME_CONFIG["BTN_SAVE_DIS_BG"],
                              THEME_CONFIG["BTN_SAVE_DIS_TEXT"])
        self.set_button_state(self.btn_process, True, THEME_CONFIG["BTN_PROCESS_BG"], THEME_CONFIG["BTN_PROCESS_HOVER"],
                              THEME_CONFIG["BTN_PROCESS_TEXT"], THEME_CONFIG["BTN_PROCESS_DIS_BG"],
                              THEME_CONFIG["BTN_PROCESS_DIS_TEXT"])
        self.set_button_state(self.btn_load, True, THEME_CONFIG["BTN_LOAD_BG"], THEME_CONFIG["BTN_LOAD_HOVER"],
                              THEME_CONFIG["BTN_LOAD_TEXT"], THEME_CONFIG["BTN_LOAD_DIS_BG"],
                              THEME_CONFIG["BTN_LOAD_DIS_TEXT"])

    def update_status(self, text):
        self.lbl_status.configure(text=text)

    def display_pil_image(self, pil_img):
        w_box = self.img_container.winfo_width() if self.img_container.winfo_width() > 100 else 800
        h_box = self.img_container.winfo_height() if self.img_container.winfo_height() > 100 else 600

        img_copy = pil_img.copy()
        img_copy.thumbnail((w_box - 20, h_box - 20), Image.Resampling.LANCZOS)

        ctk_img = ctk.CTkImage(light_image=img_copy, dark_image=img_copy, size=img_copy.size)
        self.lbl_viewport.configure(image=ctk_img, text="")
        self.lbl_viewport.image = ctk_img

    def save_image(self):
        if self.processed_bgr is not None:
            save_path = filedialog.asksaveasfilename(
                defaultextension=".png",
                filetypes=[("PNG Image", "*.png"), ("JPEG Image", "*.jpg")]
            )
            if save_path:
                cv2.imwrite(save_path, self.processed_bgr)
                messagebox.showinfo("Saved", f"Output image saved successfully:\n{save_path}")


if __name__ == "__main__":
    app = AIColorizerProApp()
    app.mainloop()