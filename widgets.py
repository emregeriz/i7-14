"""TTO için yeniden tasarlanmış CustomTkinter bileşenleri."""

from __future__ import annotations

import tkinter as tk
from typing import Callable

import cv2
import customtkinter as ctk
from PIL import Image, ImageTk

import theme


class MetricCard(ctk.CTkFrame):
    def __init__(self, master, title: str, accent: str, icon: str, **kwargs):
        super().__init__(
            master,
            fg_color=theme.SURFACE,
            corner_radius=16,
            border_width=1,
            border_color=theme.BORDER,
            **kwargs,
        )
        self.grid_columnconfigure(1, weight=1)

        icon_box = ctk.CTkLabel(
            self,
            text=icon,
            width=42,
            height=42,
            corner_radius=11,
            fg_color=theme.GREEN_DARK,
            text_color=accent,
            font=ctk.CTkFont(theme.FONT, 19, "bold"),
        )
        icon_box.grid(row=0, column=0, rowspan=2, padx=(16, 12), pady=16)

        ctk.CTkLabel(
            self,
            text=title,
            anchor="w",
            text_color=theme.TEXT_MUTED,
            font=ctk.CTkFont(theme.FONT, 10, "bold"),
        ).grid(row=0, column=1, sticky="sw", pady=(15, 0))

        self.value = ctk.CTkLabel(
            self,
            text="0",
            anchor="w",
            text_color=accent,
            font=ctk.CTkFont(theme.FONT, 31, "bold"),
        )
        self.value.grid(row=1, column=1, sticky="nw", pady=(0, 12))

    def set(self, value):
        self.value.configure(text=str(value))


class ModeCard(ctk.CTkFrame):
    def __init__(
        self,
        master,
        number: str,
        subtitle: str,
        active: bool,
        command: Callable | None = None,
    ):
        super().__init__(
            master,
            width=320,
            height=310,
            fg_color=theme.SURFACE if active else theme.BG_RAISED,
            corner_radius=24,
            border_width=2 if active else 1,
            border_color=theme.GREEN if active else theme.BORDER,
        )
        self.grid_propagate(False)
        self.grid_columnconfigure(0, weight=1)

        badge_text = "AKTİF" if active else "YAKINDA"
        badge_color = theme.GREEN if active else theme.SURFACE_LIGHT
        badge_text_color = theme.BG if active else theme.TEXT_MUTED
        ctk.CTkLabel(
            self,
            text=f"  {badge_text}  ",
            height=24,
            corner_radius=8,
            fg_color=badge_color,
            text_color=badge_text_color,
            font=ctk.CTkFont(theme.FONT, 9, "bold"),
        ).grid(row=0, column=0, pady=(25, 4))

        ctk.CTkLabel(
            self,
            text=number,
            text_color=theme.GREEN if active else theme.TEXT_MUTED,
            font=ctk.CTkFont(theme.FONT, 78, "bold"),
        ).grid(row=1, column=0, pady=(2, 0))
        ctk.CTkLabel(
            self,
            text=f"{number} KASA",
            text_color=theme.TEXT if active else theme.TEXT_MUTED,
            font=ctk.CTkFont(theme.FONT, 16, "bold"),
        ).grid(row=2, column=0)
        ctk.CTkLabel(
            self,
            text=subtitle,
            text_color=theme.TEXT_SOFT if active else theme.TEXT_MUTED,
            font=ctk.CTkFont(theme.FONT, 11),
        ).grid(row=3, column=0, pady=(5, 16))

        ctk.CTkButton(
            self,
            text=f"{number} Kasa Modunu Aç  →" if active else "Henüz Kullanılamıyor",
            command=command,
            state="normal" if active else "disabled",
            width=252,
            height=44,
            corner_radius=12,
            fg_color=theme.GREEN if active else theme.SURFACE_LIGHT,
            hover_color=theme.GREEN_HOVER,
            text_color=theme.BG if active else theme.TEXT_MUTED,
            font=ctk.CTkFont(theme.FONT, 12, "bold"),
        ).grid(row=4, column=0, pady=(0, 20))


class CameraTile(ctk.CTkFrame):
    def __init__(
        self,
        master,
        index: int,
        capture_command: Callable,
        upload_command: Callable,
        select_command: Callable,
        review_command: Callable | None = None,
        **kwargs,
    ):
        super().__init__(
            master,
            fg_color=theme.SURFACE,
            corner_radius=15,
            border_width=1,
            border_color=theme.BORDER,
            **kwargs,
        )
        self.index = index
        self.camera = None
        self.last_result = None
        self._photo = None
        self._last_image = None
        self.image_aspect = 0.75  # yükseklik / genişlik; ilk görüntüyle güncellenir
        self._badge = None
        self._unread_count = 0
        self._count_badge = None
        self._count_labels = None
        # Boş durum kartı, meşgul halkası ve çift dokunuş zamanlayıcısı
        self._placeholder = None
        self._placeholder_labels = None
        self._spinner = None
        self._spinner_canvas = None
        self._spinner_job = None
        self._spinner_running = False
        self._spinner_angle = 0
        self._select_job = None
        self.fullscreen_command = None  # (tile) -> None, çift dokunuş
        self.cancel_command = None      # () -> None, sayımı iptal
        self.capture_command = capture_command
        self.upload_command = upload_command
        self.select_command = select_command
        self.review_command = review_command
        # Tekli görünümde görüntü üzerinde zoom + kaydırma + kasa seçimi
        self.crate_click_command = None  # (tile, image_x, image_y)
        self.interactive = False
        self.zoom = 1.0
        self.center_x = 0.5
        self.center_y = 0.5
        self._view = None  # (x0, y0, scale, off_x, off_y) — tıklama eşlemesi
        self._press = None
        self._dragged = False

        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)

        header = ctk.CTkFrame(self, fg_color="transparent", height=50)
        header.grid(row=0, column=0, sticky="ew", padx=13, pady=(7, 3))
        header.grid_columnconfigure(1, weight=1)
        self.header = header

        self.dot = ctk.CTkLabel(
            header, text="●", width=15, text_color=theme.TEXT_MUTED,
            font=ctk.CTkFont(size=12),
        )
        self.dot.grid(row=0, column=0, rowspan=2, padx=(0, 8))
        self.custom_name = ""
        self.rename_command = None  # (tile) -> None; başlığa çift dokunuş
        self.title = ctk.CTkLabel(
            header,
            text=f"Kamera {index + 1}",
            anchor="w",
            text_color=theme.TEXT,
            font=ctk.CTkFont(theme.FONT, 13, "bold"),
        )
        self.title.grid(row=0, column=1, sticky="sw")
        # Kamera adına çift dokununca yeniden adlandırma açılır (görüntüdeki
        # çift dokunuş tam ekran; ikisi ayrı widget, çakışmaz).
        self.title.bind(
            "<Double-Button-1>",
            lambda _event: self.rename_command(self) if self.rename_command else None,
        )
        self.subtitle = ctk.CTkLabel(
            header,
            text="Bağlantı bekleniyor",
            anchor="w",
            text_color=theme.TEXT_MUTED,
            font=ctk.CTkFont(theme.MONO, 9),
        )
        self.subtitle.grid(row=1, column=1, sticky="nw")

        self.upload_btn = ctk.CTkButton(
            header,
            text="Yükle",
            width=58,
            height=29,
            corner_radius=8,
            fg_color=theme.SURFACE_LIGHT,
            hover_color=theme.BORDER_ACTIVE,
            font=ctk.CTkFont(theme.FONT, 9, "bold"),
            command=lambda: self.upload_command(self),
        )
        self.upload_btn.grid(row=0, column=2, rowspan=2, padx=(8, 3))

        self.capture_btn = ctk.CTkButton(
            header,
            text="Çek",
            width=52,
            height=29,
            corner_radius=8,
            state="disabled",
            fg_color=theme.SURFACE_LIGHT,
            hover_color=theme.BORDER_ACTIVE,
            font=ctk.CTkFont(theme.FONT, 9, "bold"),
            command=lambda: self.capture_command(self),
        )
        self.capture_btn.grid(row=0, column=3, rowspan=2)

        self.image_frame = ctk.CTkFrame(
            self,
            fg_color=theme.IMAGE_BG,
            corner_radius=10,
            border_width=1,
            border_color=theme.BORDER,
        )
        self.image_frame.grid(row=1, column=0, sticky="nsew", padx=12, pady=5)
        self.image_frame.grid_propagate(False)
        # İçerik pack ile yerleştiği için pack yayılımı da kapatılmalı; yoksa
        # configure(height=...) ile verilen yükseklik etiket boyutuna düşer.
        self.image_frame.pack_propagate(False)
        self.image_label = tk.Label(
            self.image_frame,
            bg=theme.IMAGE_BG,
            fg=theme.TEXT_MUTED,
            text="Görüntü bekleniyor",
            font=(theme.FONT, 10),
            cursor="hand2",
        )
        self.image_label.pack(fill="both", expand=True)
        self.image_frame.bind("<Configure>", lambda _event: self._render())
        self.image_label.bind("<ButtonPress-1>", self._on_press)
        self.image_label.bind("<Double-Button-1>", self._on_double)
        self.image_label.bind("<B1-Motion>", self._on_drag)
        self.image_label.bind("<ButtonRelease-1>", self._on_release)
        self.image_label.bind("<MouseWheel>", self._on_wheel)

        # Dokunmatik panel için ekran üstü zoom düğmeleri (tekli görünümde).
        self._zoom_buttons = []
        for text, command in (
            ("＋", lambda: self._zoom_step(1.3)),
            ("−", lambda: self._zoom_step(1 / 1.3)),
            ("⤢", self.reset_zoom),
        ):
            button = tk.Button(
                self.image_frame,
                text=text,
                font=(theme.FONT, 16, "bold"),
                bg="#FFFFFF",
                fg=theme.TEXT,
                activebackground=theme.BORDER_ACTIVE,
                bd=0,
                cursor="hand2",
                command=command,
            )
            self._zoom_buttons.append(button)

        self.stats = {"crate": 0, "barcode": 0, "unread": 0}
        self._update_placeholder()

        exposure = ctk.CTkFrame(self, fg_color="transparent")
        exposure.grid(row=2, column=0, sticky="ew", padx=13, pady=(6, 10))
        self.exposure_row = exposure
        exposure.grid_columnconfigure(2, weight=1)
        ctk.CTkLabel(
            exposure, text="Pozlama", text_color=theme.TEXT_MUTED,
            font=ctk.CTkFont(theme.FONT, 9, "bold"),
        ).grid(row=0, column=0, padx=(0, 7))
        self.exposure_auto = ctk.CTkSwitch(
            exposure,
            text="Oto",
            width=60,
            progress_color=theme.GREEN,
            text_color=theme.TEXT_SOFT,
            font=ctk.CTkFont(theme.FONT, 9),
            command=self._toggle_exposure,
        )
        self.exposure_auto.grid(row=0, column=1)
        self.exposure_auto.select()
        self.exposure_entry = ctk.CTkEntry(
            exposure,
            width=85,
            height=28,
            placeholder_text="µs",
            fg_color=theme.BG_RAISED,
            border_color=theme.BORDER,
            font=ctk.CTkFont(theme.MONO, 9),
        )
        self.exposure_entry.grid(row=0, column=3, padx=5)
        self.exposure_apply = ctk.CTkButton(
            exposure,
            text="Uygula",
            width=58,
            height=28,
            corner_radius=7,
            fg_color=theme.SURFACE_LIGHT,
            hover_color=theme.BORDER_ACTIVE,
            font=ctk.CTkFont(theme.FONT, 9, "bold"),
            command=self._apply_exposure,
        )
        self.exposure_apply.grid(row=0, column=4)
        self._set_exposure_entry_state(True)

    # Görüntü dışındaki sabit yükseklik: başlık + pozlama satırı + iç boşluklar.
    # Kart yüksekliği bundan hesaplandığı için ölçülerek alınır; yazı tipi ya da
    # düğme boyu değişse bile kamera ızgarası ekrana tam sığmayı sürdürür.
    _CHROME_PADDING = 36

    def chrome_height(self) -> int:
        try:
            head = self.header.winfo_reqheight()
            exposure = self.exposure_row.winfo_reqheight()
        except Exception:
            return 122
        return head + exposure + self._CHROME_PADDING

    def set_custom_name(self, name):
        """Operatörün verdiği konum adını başlığa yazar ("1 · Sağ Üst").

        Kamera numarası korunur: log'lar ve ortak-kasa etiketleri K1..K6
        kodunu kullanıyor, operatör ikisini eşleştirebilmeli.
        """
        self.custom_name = (name or "").strip()
        self.title.configure(
            text=f"{self.index + 1} · {self.custom_name}"
            if self.custom_name
            else f"Kamera {self.index + 1}"
        )

    def attach_camera(self, camera):
        self.camera = camera
        self._update_placeholder()
        self.dot.configure(text_color=theme.GREEN)
        self.subtitle.configure(text=f"{camera.ip or 'USB'}  ·  {camera.serial or '—'}")
        self.capture_btn.configure(state="normal")
        self.configure(border_color=theme.BORDER_ACTIVE)
        self.refresh_exposure()

    def detach_camera(self):
        self.camera = None
        self._update_placeholder()
        self.dot.configure(text_color=theme.TEXT_MUTED)
        self.subtitle.configure(text="Bağlantı bekleniyor")
        self.capture_btn.configure(state="disabled")
        self.configure(border_color=theme.BORDER)

    def set_busy(self, busy: bool):
        if busy:
            # Boş durum yazısı halkanın arkasında kalmasın.
            if self._placeholder is not None:
                self._placeholder.place_forget()
            self._ensure_spinner()
            self._spinner.place(relx=0.5, rely=0.5, anchor="center")
            if not self._spinner_running:
                self._spinner_running = True
                self._spin()
        else:
            self._stop_spinner()
            self._update_placeholder()
        self.capture_btn.configure(
            state="disabled" if busy else ("normal" if self.camera else "disabled"),
            text="…" if busy else "Çek",
        )
        self.upload_btn.configure(state="disabled" if busy else "normal")
        self.dot.configure(text_color=theme.AMBER if busy else (theme.GREEN if self.camera else theme.TEXT_MUTED))
        self.configure(border_color=theme.AMBER if busy else (theme.BORDER_ACTIVE if self.camera else theme.BORDER))

    def set_stats(self, crate=0, barcode=0, unread=0):
        self.stats = {"crate": crate, "barcode": barcode, "unread": unread}

    def set_image_height(self, height: int):
        """Görüntü alanının yüksekliğini piksel olarak sabitler.

        CTk, configure değerini DPI ölçeğiyle çarptığı için gerçek piksel
        hedefine ulaşmak üzere ölçeğe bölerek veririz.
        """
        height = max(120, int(height))
        if getattr(self, "_image_height", None) != height:
            self._image_height = height
            try:
                scaling = self.image_frame._get_widget_scaling()
            except Exception:
                scaling = 1.0
            self.image_frame.configure(height=int(height / scaling))

    # ------------------------------------------------------- boş durum kartı
    def _update_placeholder(self):
        """Görüntü yokken kameranın NEDEN boş olduğunu anlatan kart.

        Düz "Görüntü bekleniyor" yazısı yerine bağlantı durumuna göre ne
        yapılması gerektiğini söyler.
        """
        if self._last_image is not None:
            if self._placeholder is not None:
                self._placeholder.place_forget()
            return
        if self._placeholder is None:
            bg = theme.IMAGE_BG
            frame = tk.Frame(self.image_frame, bg=bg)
            icon = tk.Label(frame, bg=bg, font=(theme.FONT, 38))
            icon.pack()
            title = tk.Label(
                frame, bg=bg, fg=theme.TEXT_SOFT, font=(theme.FONT, 12, "bold")
            )
            title.pack(pady=(6, 0))
            hint = tk.Label(frame, bg=bg, fg=theme.TEXT_MUTED, font=(theme.FONT, 9))
            hint.pack()
            self._placeholder = frame
            self._placeholder_labels = {"icon": icon, "title": title, "hint": hint}
        labels = self._placeholder_labels
        if self.camera is None:
            labels["icon"].configure(text="⊘", fg=theme.TEXT_MUTED)
            labels["title"].configure(text="Bağlantı bekleniyor")
            labels["hint"].configure(text="⌁ Kameraları Tara ile bağlanın")
        else:
            labels["icon"].configure(text="▣", fg=theme.BORDER_ACTIVE)
            labels["title"].configure(text="Çekim bekleniyor")
            labels["hint"].configure(text="TÜMÜNÜ OKU ya da Çek")
        self.image_label.configure(text="")
        self._placeholder.place(relx=0.5, rely=0.5, anchor="center")

    # ----------------------------------------------------- meşgul halkası
    def _ensure_spinner(self):
        if self._spinner is not None:
            return
        bg = theme.OVERLAY_BG
        frame = tk.Frame(self.image_frame, bg=bg, padx=12, pady=9)
        canvas = tk.Canvas(
            frame, width=42, height=42, bg=bg, bd=0, highlightthickness=0
        )
        canvas.pack()
        tk.Label(
            frame,
            text="OKUNUYOR…",
            bg=bg,
            fg=theme.OVERLAY_SOFT,
            font=(theme.FONT, 8, "bold"),
        ).pack(pady=(5, 0))
        # Yanlışlıkla başlatılan sayımın bitmesini beklemek gerekmesin.
        tk.Button(
            frame,
            text="✕ İPTAL",
            bg=theme.RED,
            fg="#FFFFFF",
            activebackground="#B23A3A",
            activeforeground="#FFFFFF",
            bd=0,
            padx=8,
            pady=2,
            cursor="hand2",
            font=(theme.FONT, 8, "bold"),
            command=lambda: self.cancel_command() if self.cancel_command else None,
        ).pack(pady=(6, 0))
        self._spinner = frame
        self._spinner_canvas = canvas

    def _spin(self):
        """Dönen yay: sayım sürerken kameranın çalıştığını gösterir.

        Çalışma durumu açık bir bayrakla tutulur — place() hemen ardından
        winfo_ismapped() henüz 0 döndüğü için ona bakılırsa animasyon hiç
        başlamıyor.
        """
        self._spinner_job = None
        canvas = self._spinner_canvas
        if canvas is None or not self._spinner_running:
            return
        try:
            size, pad, width = 42, 5, 5
            canvas.delete("all")
            canvas.create_oval(
                pad, pad, size - pad, size - pad, outline="#274A35", width=width
            )
            canvas.create_arc(
                pad, pad, size - pad, size - pad,
                start=self._spinner_angle, extent=105,
                style="arc", outline=theme.GREEN, width=width,
            )
            self._spinner_angle = (self._spinner_angle - 26) % 360
            self._spinner_job = self.after(70, self._spin)
        except tk.TclError:
            self._spinner_job = None
            self._spinner_running = False

    def _stop_spinner(self):
        self._spinner_running = False
        if self._spinner_job is not None:
            try:
                self.after_cancel(self._spinner_job)
            except (tk.TclError, ValueError):
                pass
            self._spinner_job = None
        if self._spinner is not None:
            try:
                self._spinner.place_forget()
            except tk.TclError:
                pass

    # ------------------------------------------------------ çift dokunuş
    def current_image(self):
        """Kartta gösterilen son görüntü (BGR) — tam ekran için."""
        return self._last_image

    def _fire_select(self):
        self._select_job = None
        if self.select_command:
            self.select_command(self)

    def _on_double(self, _event=None):
        """Çift dokunuş: görüntüyü tam ekran aç."""
        if self._select_job is not None:
            try:
                self.after_cancel(self._select_job)
            except (tk.TclError, ValueError):
                pass
            self._select_job = None
        if self.fullscreen_command:
            self.fullscreen_command(self)
        return "break"

    def set_count_badge(
        self, own=None, raw=0, kept=0, given=0, given_text="", pending=False
    ):
        """Görüntünün sol üst köşesinde bu kameranın sayım payını gösterir.

        pending=True : tekilleştirme henüz yapılmadı; ham tespit sayısı yazar.
        own          : tekilleştirme sonrası BU kameranın adına yazılan kasa.
        kept         : başka kamerayla ortak görülüp BU kameraya yazılan kasa.
        given        : bu kameranın da gördüğü ama başka kameraya yazılan kasa.
        given_text   : nereye gittiği ("K1×8 K2×6" gibi).

        own=None ve pending=False → rozet kaldırılır.
        """
        if own is None and not pending:
            if self._count_badge is not None:
                self._count_badge.destroy()
                self._count_badge = None
                self._count_labels = None
            return
        if self._count_badge is None:
            bg = theme.OVERLAY_BG
            frame = tk.Frame(self.image_frame, bg=bg, padx=9, pady=5)
            frame.place(relx=0.0, rely=0.0, anchor="nw", x=9, y=9)
            caption = tk.Label(
                frame, bg=bg, fg=theme.OVERLAY_MUTED, anchor="w",
                font=(theme.FONT, 7, "bold"),
            )
            caption.pack(anchor="w")
            row = tk.Frame(frame, bg=bg)
            row.pack(anchor="w")
            value = tk.Label(
                row, bg=bg, fg=theme.OVERLAY_TEXT, font=(theme.FONT, 17, "bold")
            )
            value.pack(side="left")
            unit = tk.Label(
                row, bg=bg, fg=theme.OVERLAY_MUTED, font=(theme.FONT, 9)
            )
            unit.pack(side="left", pady=(5, 0))
            detail = tk.Label(
                frame, bg=bg, fg=theme.OVERLAY_SOFT, anchor="w", justify="left",
                font=(theme.MONO, 8),
            )
            detail.pack(anchor="w")
            self._count_badge = frame
            self._count_labels = {
                "caption": caption, "value": value, "unit": unit, "detail": detail,
            }
        labels = self._count_labels
        if pending:
            labels["caption"].configure(text="HAM TESPİT")
            labels["value"].configure(text=str(raw))
            labels["unit"].configure(text="  kasa")
            labels["detail"].configure(text="ortak hesaplanıyor…")
            return
        labels["caption"].configure(text="BU KAMERA SAYDI")
        labels["value"].configure(text=str(own))
        labels["unit"].configure(text="  kasa")
        parts = [f"{raw} tespit"]
        if kept:
            parts.append(f"{kept} ortak bende")
        if given:
            parts.append(f"{given}→{given_text}" if given_text else f"{given} verildi")
        if not kept and not given:
            parts.append("ortak yok")
        labels["detail"].configure(text=" · ".join(parts))

    def set_unread_badge(self, count: int):
        """Okunamayan kasa varsa görüntü üstünde yanıp sönen uyarı gösterir."""
        self._unread_count = int(count or 0)
        if self._unread_count <= 0:
            if self._badge is not None:
                self._badge.destroy()
                self._badge = None
            return
        if self._badge is None:
            self._badge = tk.Label(
                self.image_frame,
                bg=theme.RED,
                fg="#FFFFFF",
                font=(theme.FONT, 11, "bold"),
                padx=14,
                pady=6,
                cursor="hand2",
            )
            self._badge.place(relx=0.5, rely=1.0, anchor="s", y=-8)
            self._badge.bind(
                "<Button-1>",
                lambda _event: self.review_command(self) if self.review_command else None,
            )
        self._badge.configure(
            text=f"⚠ {self._unread_count} OKUNAMAYAN · DÜZELT"
        )

    def blink_badge(self, on: bool):
        if self._badge is not None:
            self._badge.configure(bg=theme.RED if on else "#8F1F1F")

    def show_image(self, image_bgr):
        if image_bgr is None:
            return
        self._last_image = image_bgr
        self._update_placeholder()
        h, w = image_bgr.shape[:2]
        if w > 0:
            self.image_aspect = h / w
        self._render()

    # ------------------------------------------------ zoom / kaydırma / seçim
    def set_interactive(self, on: bool):
        """Tekli görünümde zoom + kasa seçimini açar, 6'lı görünümde kapatır."""
        on = bool(on)
        if on == self.interactive:
            return
        self.interactive = on
        if on:
            for position, button in enumerate(self._zoom_buttons):
                button.place(
                    relx=1.0, x=-10, y=10 + position * 56,
                    anchor="ne", width=48, height=48,
                )
        else:
            for button in self._zoom_buttons:
                button.place_forget()
            self.reset_zoom(render=False)
        self._render()

    def reset_zoom(self, render=True):
        self.zoom = 1.0
        self.center_x = 0.5
        self.center_y = 0.5
        if render:
            self._render()

    def _box_size(self):
        return (
            max(180, self.image_frame.winfo_width() - 4),
            max(130, self.image_frame.winfo_height() - 4),
        )

    def _zoom_step(self, factor):
        box_w, box_h = self._box_size()
        self._zoom_at(factor, box_w / 2, box_h / 2)

    def _on_wheel(self, event):
        if not self.interactive:
            return
        self._zoom_at(1.2 if event.delta > 0 else 1 / 1.2, event.x, event.y)

    def _zoom_at(self, factor, event_x, event_y):
        if self._last_image is None:
            return
        new_zoom = min(12.0, max(1.0, self.zoom * factor))
        if self._view is not None and new_zoom != self.zoom:
            x0, y0, scale, off_x, off_y = self._view
            height, width = self._last_image.shape[:2]
            image_x = x0 + (event_x - off_x) / scale
            image_y = y0 + (event_y - off_y) / scale
            box_w, box_h = self._box_size()
            new_scale = scale / self.zoom * new_zoom
            # İmleç altındaki nokta ekranda sabit kalacak şekilde merkezi kaydır.
            self.center_x = min(1.0, max(0.0, (image_x - (event_x - off_x - box_w / 2) / new_scale) / width))
            self.center_y = min(1.0, max(0.0, (image_y - (event_y - off_y - box_h / 2) / new_scale) / height))
        self.zoom = new_zoom
        self._render()

    def _on_press(self, event):
        if not self.interactive:
            # Tekli seçimi kısa süre geciktir: çift dokunuş gelirse iptal edilip
            # tam ekran açılır, yoksa normal seçim çalışır.
            if self._select_job is not None:
                try:
                    self.after_cancel(self._select_job)
                except (tk.TclError, ValueError):
                    pass
            self._select_job = self.after(240, self._fire_select)
            return
        self._press = (event.x, event.y)
        self._dragged = False

    def _on_drag(self, event):
        if not self.interactive or self._press is None or self._view is None:
            return
        dx = event.x - self._press[0]
        dy = event.y - self._press[1]
        if abs(dx) + abs(dy) > 4:
            self._dragged = True
        scale = self._view[2]
        height, width = self._last_image.shape[:2]
        self.center_x = min(1.0, max(0.0, self.center_x - dx / scale / width))
        self.center_y = min(1.0, max(0.0, self.center_y - dy / scale / height))
        self._press = (event.x, event.y)
        self._render()

    def _on_release(self, event):
        if not self.interactive:
            return
        pressed = self._press is not None
        self._press = None
        if not pressed or self._dragged or self._view is None:
            return
        x0, y0, scale, off_x, off_y = self._view
        image_x = x0 + (event.x - off_x) / scale
        image_y = y0 + (event.y - off_y) / scale
        height, width = self._last_image.shape[:2]
        if 0 <= image_x <= width and 0 <= image_y <= height and self.crate_click_command:
            self.crate_click_command(self, image_x, image_y)

    def _render(self):
        if self._last_image is None:
            return
        try:
            img = self._last_image
            height, width = img.shape[:2]
            box_w, box_h = self._box_size()
            fit_scale = min(box_w / width, box_h / height)
            scale = fit_scale * self.zoom
            view_w = min(width, box_w / scale)
            view_h = min(height, box_h / scale)
            cx = min(width - view_w / 2, max(view_w / 2, self.center_x * width))
            cy = min(height - view_h / 2, max(view_h / 2, self.center_y * height))
            self.center_x = cx / width
            self.center_y = cy / height
            x0 = max(0, int(round(cx - view_w / 2)))
            y0 = max(0, int(round(cy - view_h / 2)))
            x1 = min(width, x0 + max(1, int(round(view_w))))
            y1 = min(height, y0 + max(1, int(round(view_h))))
            x0 = max(0, x1 - max(1, int(round(view_w))))
            y0 = max(0, y1 - max(1, int(round(view_h))))
            crop = img[y0:y1, x0:x1]
            disp_w = max(1, int((x1 - x0) * scale))
            disp_h = max(1, int((y1 - y0) * scale))
            interpolation = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
            resized = cv2.resize(crop, (disp_w, disp_h), interpolation=interpolation)
            if len(resized.shape) == 2:
                rgb = cv2.cvtColor(resized, cv2.COLOR_GRAY2RGB)
            else:
                rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            self._photo = ImageTk.PhotoImage(Image.fromarray(rgb))
            self.image_label.configure(image=self._photo, text="")
            label_w = self.image_label.winfo_width() or box_w
            label_h = self.image_label.winfo_height() or box_h
            self._view = (
                x0, y0, scale,
                (label_w - disp_w) / 2,
                (label_h - disp_h) / 2,
            )
        except Exception:
            pass

    def refresh_exposure(self):
        if not self.camera:
            return
        info = self.camera.read_exposure_info()
        current = info.get("cur")
        is_auto = info.get("auto") == 2 if info.get("auto") is not None else True
        if current is not None:
            self.exposure_entry.configure(state="normal")
            self.exposure_entry.delete(0, "end")
            self.exposure_entry.insert(0, str(int(current)))
        if is_auto:
            self.exposure_auto.select()
        else:
            self.exposure_auto.deselect()
        self._set_exposure_entry_state(is_auto)

    def _set_exposure_entry_state(self, is_auto: bool):
        state = "disabled" if is_auto else "normal"
        self.exposure_entry.configure(state=state)
        self.exposure_apply.configure(state=state)

    def _toggle_exposure(self):
        if not self.camera:
            return
        is_auto = bool(self.exposure_auto.get())
        self.camera.set_exposure_auto(continuous=is_auto)
        self._set_exposure_entry_state(is_auto)
        if not is_auto:
            self.refresh_exposure()

    def _apply_exposure(self):
        if not self.camera:
            return
        try:
            value = float(self.exposure_entry.get().strip())
        except ValueError:
            return
        self.camera.set_exposure_us(value)


class InfoCard(ctk.CTkFrame):
    """Doğrulama ekranı bilgi kartı: simge + başlık, büyük değer, alt satır.

    ``rows`` verilirse büyük değer yerine alt alta etiket/değer çiftleri
    kurulur (ör. Barkodlu / Barkodsuz). ``custom=True`` gövdeyi boş bırakır;
    çağıran ``card.body`` içine kendi bileşenlerini yerleştirir. ``command``
    verilirse kartın tamamı tıklanabilir olur ve başlıkta ok görünür.
    """

    def __init__(
        self,
        master,
        title: str,
        accent: str,
        icon: str,
        rows: tuple[str, ...] | None = None,
        custom: bool = False,
        command: Callable | None = None,
        **kwargs,
    ):
        super().__init__(
            master,
            fg_color=theme.SURFACE,
            corner_radius=16,
            border_width=1,
            border_color=theme.BORDER,
            **kwargs,
        )
        self.accent = accent
        self.command = command
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)

        header = ctk.CTkFrame(self, fg_color="transparent")
        header.grid(row=0, column=0, sticky="ew", padx=14, pady=(12, 0))
        ctk.CTkLabel(
            header,
            text=icon,
            width=34,
            height=34,
            corner_radius=9,
            fg_color=theme.GREEN_DARK,
            text_color=accent,
            font=ctk.CTkFont(theme.FONT, 15, "bold"),
        ).pack(side="left")
        self.title = ctk.CTkLabel(
            header,
            text=title,
            anchor="w",
            justify="left",
            text_color=theme.TEXT_MUTED,
            font=ctk.CTkFont(theme.FONT, 10, "bold"),
        )
        self.title.pack(side="left", padx=(10, 0))
        if command is not None:
            ctk.CTkLabel(
                header,
                text="›",
                width=14,
                text_color=accent,
                font=ctk.CTkFont(theme.FONT, 22, "bold"),
            ).pack(side="right")

        self.body = ctk.CTkFrame(self, fg_color="transparent")
        self.body.grid(row=1, column=0, sticky="nsew", padx=14, pady=(6, 12))

        self.value = None
        self.row_values: list[ctk.CTkLabel] = []
        self.sub = None
        if custom:
            return
        if rows:
            for label_text in rows:
                row = ctk.CTkFrame(self.body, fg_color="transparent")
                row.pack(fill="x", pady=1)
                ctk.CTkLabel(
                    row,
                    text=label_text,
                    anchor="w",
                    text_color=theme.TEXT_SOFT,
                    font=ctk.CTkFont(theme.FONT, 11, "bold"),
                ).pack(side="left")
                value = ctk.CTkLabel(
                    row,
                    text="0",
                    anchor="e",
                    text_color=accent,
                    font=ctk.CTkFont(theme.FONT, 22, "bold"),
                )
                value.pack(side="right")
                self.row_values.append(value)
        else:
            self.value = ctk.CTkLabel(
                self.body,
                text="0",
                anchor="w",
                text_color=accent,
                font=ctk.CTkFont(theme.FONT, 31, "bold"),
            )
            self.value.pack(anchor="w")
        # wraplength: uzun alt yazı kartın en küçük genişliğini büyütmesin;
        # 5 kart yan yana 1380 px'lik en küçük pencereye sığmalı.
        self.sub = ctk.CTkLabel(
            self.body,
            text="",
            anchor="w",
            justify="left",
            wraplength=170,
            text_color=theme.TEXT_MUTED,
            font=ctk.CTkFont(theme.FONT, 10),
        )
        self.sub.pack(anchor="w", fill="x")
        if command is not None:
            self.bind_click(self)

    def bind_click(self, widget):
        """Kartın tüm alt bileşenlerinde tıklamayı komuta bağlar."""
        if self.command is None:
            return
        try:
            widget.bind("<Button-1>", lambda _event: self.command())
        except Exception:
            pass
        try:
            widget.configure(cursor="hand2")
        except Exception:
            pass
        for child in widget.winfo_children():
            self.bind_click(child)

    def set(self, value=None, sub=None, rows=None, sub_color=None):
        if value is not None and self.value is not None:
            self.value.configure(text=str(value))
        if rows is not None:
            for label, text in zip(self.row_values, rows):
                label.configure(text=str(text))
        if sub is not None and self.sub is not None:
            self.sub.configure(text=sub, text_color=sub_color or theme.TEXT_MUTED)
