# -*- coding: utf-8 -*-
"""TTO arka plan / boy / sütun filtresi.

TTO/app.py içindeki TTOApplication filtre metodlarının BİREBİR kopyası
(2026-09-25). Test paketinin TTO'nun tamamına ve Hikrobot SDK'sına bağımlı
olmaması için ayrı modüle alındı. Mantık ve sabitler değiştirilmedi.
"""
import os

import cv2


class ArkaPlanFiltresi:
    def __init__(self, log_fn=None):
        self._log_fn = log_fn

    def log(self, message):
        if self._log_fn:
            self._log_fn(message)

    # Arka plandaki (aynı paletin arka sırası ya da komşu palet) kasaları
    # sayımdan ayıklamak için iki filtre birlikte çalışır:
    #  1) Genişlik: ort. genişliğin %85'inden dar kutular (yandan yarım
    #     görünen arka kasalar) elenir — ODAI'deki filtre.
    #  2) Sütun: kutular x-merkezine göre kümelenir; palet düzeninde her
    #     kamera 1-2 dolu sütun görür. Az üyeli kümeler (kenardan sızan
    #     arka sıra) elenir. 4x2 dizilimde arka sıra tam genişlikte bile
    #     görünse kendi sütununda az kutu olduğundan yakalanır.
    WIDTH_FILTER_RATIO = 0.85
    WIDTH_FILTER_MIN_CRATES = 6
    COLUMN_GAP_RATIO = 0.6      # yeni sütun sayılması için gereken cx boşluğu
    COLUMN_MIN_FRACTION = 0.15  # geçerli sütun için asgari tespit oranı
    COLUMN_MIN_CRATES = 4       # geçerli sütun için asgari kutu sayısı

    # Yükseklik (y) aykırı değer filtresi. Paletteki kasalar aynı ürün olduğu
    # için bir kutunun yüksekliği KOMŞULARININKİNE çok yakındır. Komşularından
    # ±%20'den fazla sapan kutu:
    #   · belirgin KISA  → arka sıradan sızmış / yarım görünen, kasa değil,
    #   · belirgin UZUN  → YOLO'nun iki kasayı tek kutuda birleştirmesi.
    #
    # Referans KÜRESEL medyan DEĞİL, aynı sütunda en yakın komşuların
    # medyanıdır. Sebep ölçüldü: perspektif yüzünden aynı paletin kasaları
    # kare boyunca 101–209px arasında değişiyor; küresel ±%20 eşiği 313
    # kutunun 34'ünü (gerçek kasaları) eliyordu. Komşu penceresi bu kademeli
    # kaymayı soğurur, yalnız tek başına sırıtan kutuyu yakalar (7 kutu).
    # Ölçülen gerçek davranış (6 kamera, 313 kutu):
    #   · Perspektif yüzünden boy sütun boyunca kademeli değişiyor
    #     (K4'te 124→252px). Küresel medyan eşiği 34 GERÇEK kasayı eliyordu.
    #   · Komşu MEDYANI da kenarlarda haksız: sütunun en üst/en alt kutusunun
    #     bütün komşuları tek yönde kaldığı için referans kayıyor
    #     (K1 #0: doğrusal tahminden -%2 sapıyor ama komşu medyanından -%13).
    # Bu yüzden referans, komşulara uydurulan DOĞRU üzerinde tahmin edilen
    # boydur; kenar kutularda eğilim ileri taşınır (ekstrapolasyon). Eğim
    # Theil-Sen (ikili eğimlerin medyanı) ile bulunur: penceredeki tek bozuk
    # komşu referansı bozamaz.
    # Eşikler ölçülerek seçildi (6 kamera, 295 kutu). Sapma dağılımı ASİMETRİK:
    #   kısa taraf : normal kasalar -%12.6'ya kadar iniyor, aykırılar -%25.7'den
    #                başlıyor  → -%20 tam boşluğun ortası
    #   uzun taraf : normal kasalar +%11.7'ye kadar çıkıyor, aykırılar
    #                +%53.2'den başlıyor → +%35 güvenli
    # Uzun tarafın bol bırakılması bilinçli: arka plandaki/yarım kasa KISA olur,
    # uzun sapma ise ancak YOLO iki kasayı birleştirince oluşur ve o çok büyük
    # (+%50 üstü) çıkar. Dar tutulursa perspektifte uzayan gerçek kasalar
    # haksız eleniyordu.
    HEIGHT_FILTER_SHORT_TOLERANCE = 0.20
    HEIGHT_FILTER_TALL_TOLERANCE = 0.35
    # Kare kenarına DEĞEN kutu (üstten/alttan kesilmiş) yarım görünür; arka
    # planda kalan kasalar çoğunlukla böyle. Bunlar için daha sıkı eşik
    # kullanılır — ama yalnız KISA tarafta: tam boy görünen, sadece kenarda
    # biten gerçek kasa elenmesin.
    HEIGHT_FILTER_EDGE_TOLERANCE = 0.08
    HEIGHT_FILTER_EDGE_MARGIN = 4  # px: kenara bu kadar yakınsa "değiyor"
    HEIGHT_FILTER_MIN_CRATES = 6
    HEIGHT_FILTER_WINDOW = 8       # karşılaştırmaya giren komşu sayısı
    HEIGHT_FILTER_MIN_COLUMN = 6   # bu kadar kutusu olmayan sütunda eleme yok
    # Bu orandan fazlasını eleyecekse referansın kendisi şüphelidir; dokunma.
    HEIGHT_FILTER_MAX_DROP = 0.30

    @staticmethod
    def _robust_line(xs, ys):
        """Theil-Sen doğrusu: eğim = ikili eğimlerin medyanı. (eğim, sabit)

        En küçük kareler penceredeki tek aykırı komşudan etkilenir; ikili
        eğimlerin medyanı etkilenmez.
        """
        slopes = []
        for i in range(len(xs)):
            for j in range(i + 1, len(xs)):
                dx = xs[j] - xs[i]
                if abs(dx) > 1e-6:
                    slopes.append((ys[j] - ys[i]) / dx)
        if not slopes:
            return 0.0, (sum(ys) / len(ys) if ys else 0.0)
        slope = sorted(slopes)[len(slopes) // 2]
        intercepts = sorted(y - slope * x for x, y in zip(xs, ys))
        return slope, intercepts[len(intercepts) // 2]

    def _column_clusters(self, kasalar):
        """Kutuları x-merkezine göre sütunlara ayırır (palet dizilimi)."""
        med_width = sorted(
            float(k["bbox"][2] - k["bbox"][0]) for k in kasalar
        )[len(kasalar) // 2]
        ordered = sorted(
            kasalar, key=lambda k: (k["bbox"][0] + k["bbox"][2]) / 2.0
        )
        clusters = [[ordered[0]]]
        for kasa in ordered[1:]:
            cx = (kasa["bbox"][0] + kasa["bbox"][2]) / 2.0
            prev = clusters[-1][-1]
            prev_cx = (prev["bbox"][0] + prev["bbox"][2]) / 2.0
            if cx - prev_cx > med_width * self.COLUMN_GAP_RATIO:
                clusters.append([kasa])
            else:
                clusters[-1].append(kasa)
        return clusters

    def _height_outliers(self, kasalar, frame_height=None):
        """Komşularının boyuna uymayan kutuları döner.

        [(kasa, boy, referans), ...] — çağıran nedeni görüntüye yazabilsin.
        frame_height verilirse karenin üst/alt kenarına değen kutulara daha
        sıkı eşik uygulanır (yarım görünen arka plan kasaları).
        """
        if len(kasalar) < self.HEIGHT_FILTER_MIN_CRATES:
            return []
        margin = self.HEIGHT_FILTER_EDGE_MARGIN
        window = self.HEIGHT_FILTER_WINDOW
        outliers = []
        for column in self._column_clusters(kasalar):
            if len(column) < self.HEIGHT_FILTER_MIN_COLUMN:
                continue  # kısa sütunda güvenilir referans yok
            ordered = sorted(
                column, key=lambda k: (k["bbox"][1] + k["bbox"][3]) / 2.0
            )
            heights = [float(k["bbox"][3] - k["bbox"][1]) for k in ordered]
            centers = [(k["bbox"][1] + k["bbox"][3]) / 2.0 for k in ordered]
            for position, kasa in enumerate(ordered):
                end = min(len(ordered), max(0, position - window // 2) + window + 1)
                start = max(0, end - window - 1)
                nx = [
                    centers[offset]
                    for offset in range(start, end)
                    if offset != position
                ]
                ny = [
                    heights[offset]
                    for offset in range(start, end)
                    if offset != position
                ]
                if len(ny) < 4:
                    continue
                slope, intercept = self._robust_line(nx, ny)
                reference = slope * centers[position] + intercept
                if reference <= 0:
                    continue
                height = heights[position]
                short_tol = self.HEIGHT_FILTER_SHORT_TOLERANCE
                # Karenin üst/alt kenarında biten kutu yarım görünüyor demektir;
                # KISA tarafta daha sıkı davran (uzun tarafa dokunma).
                if frame_height:
                    y1, y2 = kasa["bbox"][1], kasa["bbox"][3]
                    if y1 <= margin or y2 >= frame_height - margin:
                        short_tol = self.HEIGHT_FILTER_EDGE_TOLERANCE
                low = reference * (1.0 - short_tol)
                high = reference * (1.0 + self.HEIGHT_FILTER_TALL_TOLERANCE)
                if not (low < height < high):
                    outliers.append((kasa, height, reference))
        if len(outliers) > len(kasalar) * self.HEIGHT_FILTER_MAX_DROP:
            return []  # emniyet: referans şüpheli, eleme yapma
        return outliers

    def _column_outliers(self, kasalar):
        """Ana sütunlara oturmayan (arka sıradan sızan) kutuları döner."""
        clusters = self._column_clusters(kasalar)
        min_members = max(
            self.COLUMN_MIN_CRATES, int(len(kasalar) * self.COLUMN_MIN_FRACTION)
        )
        if not any(len(c) >= min_members for c in clusters):
            return []  # hiç güçlü sütun yoksa eleme yapma (emniyet)
        outliers = []
        for cluster in clusters:
            if len(cluster) < min_members:
                outliers.extend(cluster)
        return outliers

    def _filter_background_crates(self, index, result, frame_height=None):
        """Arka plan kutularını sonuçtan çıkarır; sayaçları düzeltir."""
        kasalar = result.get("kasalar") or []
        if frame_height is None:
            annotated = result.get("annotated_image")
            if annotated is not None:
                frame_height = annotated.shape[0]
        if len(kasalar) < self.WIDTH_FILTER_MIN_CRATES:
            return
        widths = [float(k["bbox"][2] - k["bbox"][0]) for k in kasalar]
        avg_width = sum(widths) / len(widths)
        threshold = avg_width * self.WIDTH_FILTER_RATIO
        kept, dropped = [], []
        reasons = {}  # id(kasa) -> görüntüye yazılacak eleme nedeni (ASCII)
        for kasa in kasalar:
            width = float(kasa["bbox"][2] - kasa["bbox"][0])
            if width >= threshold:
                kept.append(kasa)
            else:
                dropped.append(kasa)
                reasons[id(kasa)] = "ARKA PLAN"
        # Yükseklik aykırıları: medyandan ±%20 sapan kutular
        if len(kept) >= self.HEIGHT_FILTER_MIN_CRATES:
            height_dropped = self._height_outliers(kept, frame_height)
            if height_dropped:
                for kasa, height, reference in height_dropped:
                    reasons[id(kasa)] = "AYKIRI BOY %d/%d" % (
                        int(height), int(reference)
                    )
                dropped.extend(kasa for kasa, _, _ in height_dropped)
                dropped_ids = {id(kasa) for kasa, _, _ in height_dropped}
                kept = [k for k in kept if id(k) not in dropped_ids]
        if len(kept) >= self.WIDTH_FILTER_MIN_CRATES:
            column_dropped = self._column_outliers(kept)
            if column_dropped:
                dropped.extend(column_dropped)
                for kasa in column_dropped:
                    reasons[id(kasa)] = "SUTUN DISI"
                dropped_ids = {id(kasa) for kasa in column_dropped}
                kept = [k for k in kept if id(k) not in dropped_ids]
        if not dropped:
            return
        result["kasalar"] = kept
        result["toplam_kasa"] = len(kept)
        dropped_unread_nos = {
            kasa["kasa_no"] for kasa in dropped if not kasa.get("barkodlar")
        }
        result["okunan_barkod"] = max(
            0,
            result.get("okunan_barkod", 0)
            - sum(len(kasa.get("barkodlar") or []) for kasa in dropped),
        )
        result["okunamayan_barkod"] = max(
            0, result.get("okunamayan_barkod", 0) - len(dropped_unread_nos)
        )

        def is_dropped(path):
            base = os.path.basename(str(path))
            return any(f"_kasa_{no}." in base for no in dropped_unread_nos)

        result["barkod_bulunamayan_yollar"] = [
            p for p in result.get("barkod_bulunamayan_yollar") or [] if not is_dropped(p)
        ]
        result["barkod_bulunamayan_isimler"] = [
            n for n in result.get("barkod_bulunamayan_isimler") or [] if not is_dropped(n)
        ]
        annotated = result.get("annotated_image")
        if annotated is not None:
            for kasa in dropped:
                x1, y1, x2, y2 = (int(v) for v in kasa["bbox"])
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (140, 140, 140), 4)
                cv2.putText(
                    annotated,
                    reasons.get(id(kasa), "ARKA PLAN"),
                    (x1 + 6, max(24, y1 + 26)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (140, 140, 140),
                    2,
                    cv2.LINE_AA,
                )
        self.log(
            f"K{index + 1}: {len(dropped)} kutu elendi "
            f"(dar eşik {threshold:.0f}px · komşu boyundan -%"
            f"{int(self.HEIGHT_FILTER_SHORT_TOLERANCE * 100)}/+%"
            f"{int(self.HEIGHT_FILTER_TALL_TOLERANCE * 100)} sapma · sütun dışı) → "
            + ", ".join(
                f"#{k['kasa_no']}[{reasons.get(id(k), 'ARKA PLAN')}]"
                for k in dropped
            )
        )

