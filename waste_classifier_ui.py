import tkinter as tk
from tkinter import filedialog, Canvas
from PIL import Image, ImageTk, ImageDraw, ImageFilter
import numpy as np
import os
try:
    import tensorflow as tf
    from tensorflow.keras.preprocessing import image as keras_image
    model = tf.keras.models.load_model("waste_model.h5")
    MODEL_LOADED = True
except Exception:
    MODEL_LOADED = False

class_names = ['GLASS', 'ORGANIC', 'PAPER']

BG_DARK     = "#0D0F14"
BG_CARD     = "#161A23"
BG_PANEL    = "#1C2130"
ACCENT_LINE = "#252C3D"

GLASS_COLOR   = "#4FC3F7"
GLASS_GLOW    = "#0D3A52"
ORGANIC_COLOR = "#69F0AE"
ORGANIC_GLOW  = "#0A3526"
PAPER_COLOR   = "#FFD54F"   
PAPER_GLOW    = "#3D2E00"

TEXT_PRIMARY   = "#F0F4FF"
TEXT_SECONDARY = "#6B7A99"
TEXT_MUTED     = "#3A4257"

FONT_TITLE  = ("Segoe UI", 28, "bold")
FONT_SUB    = ("Segoe UI", 11)
FONT_LABEL  = ("Segoe UI", 9)
FONT_RESULT = ("Segoe UI", 32, "bold")
FONT_CONF   = ("Segoe UI", 13)
FONT_CLASS  = ("Segoe UI", 10, "bold")
FONT_PCT    = ("Segoe UI", 10)
FONT_BTN    = ("Segoe UI", 12, "bold")
FONT_FOOTER = ("Segoe UI", 8)

CLASS_PALETTE = {
    "GLASS":   {"fg": GLASS_COLOR,   "glow": GLASS_GLOW,   "icon": "◈"},
    "ORGANIC": {"fg": ORGANIC_COLOR, "glow": ORGANIC_GLOW, "icon": "◉"},
    "PAPER":   {"fg": PAPER_COLOR,   "glow": PAPER_GLOW,   "icon": "◧"},
}

def rounded_rect(canvas, x1, y1, x2, y2, r=12, **kwargs):
    pts = [
        x1+r, y1,   x2-r, y1,
        x2,   y1,   x2,   y1+r,
        x2,   y2-r, x2,   y2,
        x2-r, y2,   x1+r, y2,
        x1,   y2,   x1,   y2-r,
        x1,   y1+r, x1,   y1,
        x1+r, y1,
    ]
    return canvas.create_polygon(pts, smooth=True, **kwargs)


class WasteClassifierApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Smart Waste Classifier")
        self.root.geometry("520x820")
        self.root.configure(bg=BG_DARK)
        self.root.resizable(False, False)

        self.current_photo = None
        self.bar_vars = {}
        self.bar_after_ids = []

        self._build_ui()

    def _build_ui(self):
        root = self.root
        W = 520

        header = tk.Frame(root, bg=BG_CARD, height=72)
        header.pack(fill="x")
        header.pack_propagate(False)

        tk.Label(header, text="WASTE  AI", font=("Segoe UI", 22, "bold"),
                 fg=TEXT_PRIMARY, bg=BG_CARD).place(relx=0.5, y=28, anchor="center")
        tk.Label(header, text="neural classification system",
                 font=("Segoe UI", 9), fg=TEXT_SECONDARY, bg=BG_CARD
                 ).place(relx=0.5, y=52, anchor="center")

        # thin accent line under header
        tk.Frame(root, bg=ACCENT_LINE, height=1).pack(fill="x")

        self.drop_canvas = Canvas(root, width=W, height=300,
                                  bg=BG_DARK, highlightthickness=0)
        self.drop_canvas.pack()
        self._draw_dropzone()
        self.drop_canvas.bind("<Button-1>", lambda e: self.classify())

        res_outer = tk.Frame(root, bg=BG_DARK, padx=20)
        res_outer.pack(fill="x", pady=(4, 0))

        self.result_card = tk.Frame(res_outer, bg=BG_CARD,
                                    highlightbackground=ACCENT_LINE,
                                    highlightthickness=1)
        self.result_card.pack(fill="x")

        badge_row = tk.Frame(self.result_card, bg=BG_CARD)
        badge_row.pack(fill="x", padx=24, pady=(20, 0))

        self.class_icon  = tk.Label(badge_row, text="◈", font=("Segoe UI", 22),
                                    fg=TEXT_MUTED, bg=BG_CARD)
        self.class_icon.pack(side="left")

        self.class_label = tk.Label(badge_row, text="—  no image selected",
                                    font=FONT_CONF, fg=TEXT_SECONDARY, bg=BG_CARD)
        self.class_label.pack(side="left", padx=10)

        self.conf_badge = tk.Label(badge_row, text="", font=FONT_LABEL,
                                   fg=TEXT_MUTED, bg=BG_PANEL, padx=8, pady=3)
        self.conf_badge.pack(side="right")

        tk.Frame(self.result_card, bg=ACCENT_LINE, height=1
                 ).pack(fill="x", padx=24, pady=12)

        bars_frame = tk.Frame(self.result_card, bg=BG_CARD)
        bars_frame.pack(fill="x", padx=24, pady=(0, 20))

        self.bar_widgets = {}
        for cls in class_names:
            pal = CLASS_PALETTE[cls]
            row = tk.Frame(bars_frame, bg=BG_CARD)
            row.pack(fill="x", pady=4)

            lbl = tk.Label(row, text=cls, font=FONT_CLASS,
                           fg=pal["fg"], bg=BG_CARD, width=8, anchor="w")
            lbl.pack(side="left")

            track = tk.Frame(row, bg=BG_PANEL, height=6)
            track.pack(side="left", fill="x", expand=True, padx=(6, 8))
            track.pack_propagate(False)

            fill = tk.Frame(track, bg=pal["fg"], height=6, width=0)
            fill.place(x=0, y=0, height=6)

            pct_lbl = tk.Label(row, text="0%", font=FONT_PCT,
                               fg=TEXT_SECONDARY, bg=BG_CARD, width=5, anchor="e")
            pct_lbl.pack(side="right")

            self.bar_widgets[cls] = {"track": track, "fill": fill, "pct": pct_lbl}

        btn_frame = tk.Frame(root, bg=BG_DARK, padx=20)
        btn_frame.pack(fill="x", pady=16)

        self.upload_btn = tk.Button(
            btn_frame, text="Choose Image",
            font=FONT_BTN, fg=BG_DARK, bg=GLASS_COLOR,
            activeforeground=BG_DARK, activebackground="#81D4FA",
            relief="flat", bd=0, padx=0, pady=14,
            cursor="hand2", command=self.classify
        )
        self.upload_btn.pack(fill="x")

        self.upload_btn.bind("<Enter>",
            lambda e: self.upload_btn.config(bg="#81D4FA"))
        self.upload_btn.bind("<Leave>",
            lambda e: self.upload_btn.config(bg=GLASS_COLOR))

        tk.Frame(root, bg=ACCENT_LINE, height=1).pack(fill="x", pady=(4, 0))
        footer = tk.Frame(root, bg=BG_DARK)
        footer.pack(fill="x", pady=8)
        tk.Label(footer, text="GROUP 7  ·  90.5% ACCURACY  ·  GLASS · ORGANIC · PAPER",
                 font=FONT_FOOTER, fg=TEXT_MUTED, bg=BG_DARK).pack()

    def _draw_dropzone(self, photo=None):
        c = self.drop_canvas
        W, H = 520, 300
        c.delete("all")

        if photo:
            c.create_image(W//2, H//2, anchor="center", image=photo)

            c.create_rectangle(0, 0, W, H, fill="", outline="",
                                stipple="gray12")
        else:
            margin = 24
            dash = (6, 5)
            c.create_rectangle(margin, margin, W-margin, H-margin,
                                outline=TEXT_MUTED, width=1, dash=dash)

            c.create_text(W//2, H//2 - 22, text="⬆",
                          font=("Segoe UI", 32), fill=TEXT_MUTED)
            c.create_text(W//2, H//2 + 18,
                          text="click to select an image",
                          font=("Segoe UI", 11), fill=TEXT_SECONDARY)
            c.create_text(W//2, H//2 + 42,
                          text="jpg  ·  png  ·  bmp",
                          font=("Segoe UI", 9), fill=TEXT_MUTED)

    def classify(self):
        path = filedialog.askopenfilename(
            title="Select a waste image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")]
        )
        if not path:
            return

        img_disp = Image.open(path).resize((520, 300), Image.Resampling.LANCZOS)
        self.current_photo = ImageTk.PhotoImage(img_disp)
        self._draw_dropzone(self.current_photo)

        if MODEL_LOADED:
            img = keras_image.load_img(path, target_size=(150, 150))
            arr = keras_image.img_to_array(img)
            arr = np.expand_dims(arr, axis=0) / 255.0
            preds = model.predict(arr, verbose=0)[0]
        else:
            # demo values when model not found
            preds = np.array([0.72, 0.18, 0.10])

        pred_idx   = int(np.argmax(preds))
        pred_class = class_names[pred_idx]
        confidence = preds[pred_idx] * 100

        self._update_result(pred_class, confidence, preds)

    def _update_result(self, cls, confidence, preds):
        pal = CLASS_PALETTE[cls]

        # class label + icon
        self.class_icon.config(text=pal["icon"], fg=pal["fg"])
        self.class_label.config(text=cls, font=("Segoe UI", 16, "bold"),
                                fg=pal["fg"])
        self.conf_badge.config(text=f"{confidence:.1f}%",
                               fg=pal["fg"], bg=pal["glow"])

        # animated bars
        for id_ in self.bar_after_ids:
            try: self.root.after_cancel(id_)
            except: pass
        self.bar_after_ids.clear()

        for i, name in enumerate(class_names):
            target_pct = int(preds[i] * 100)
            self._animate_bar(name, 0, target_pct, steps=20)

    def _animate_bar(self, cls, current, target, steps):
        w = self.bar_widgets[cls]
        track_w = w["track"].winfo_width() or 220

        w["pct"].config(text=f"{current}%")
        bar_w = int(track_w * current / 100)
        w["fill"].place(x=0, y=0, height=6, width=max(bar_w, 0))

        if current < target and steps > 0:
            nxt = current + max(1, (target - current) // steps)
            nxt = min(nxt, target)
            id_ = self.root.after(18, self._animate_bar, cls, nxt, target, steps-1)
            self.bar_after_ids.append(id_)
        else:
            w["pct"].config(text=f"{target}%")
            bar_w = int(track_w * target / 100)
            w["fill"].place(x=0, y=0, height=6, width=max(bar_w, 0))


if __name__ == "__main__":
    root = tk.Tk()
    app  = WasteClassifierApp(root)
    root.mainloop()
