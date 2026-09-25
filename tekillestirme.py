# -*- coding: utf-8 -*-
"""Kasa tekilleştirme (CrateAggregator).

Kamera_gui/multi_camera_gui_2.py içindeki _UnionFind + CrateAggregator'ın
BİREBİR kopyası (2026-09-25); sabitler dahil değiştirilmedi.
"""
import numpy as np

DEDUP_TOLERANCE_RATIO = 0.5   # ortak barkodla hizalamada kasa boyutunun % oranı
DEDUP_MIN_SHARED = 1          # pair için gerekli min ortak barkod
DEDUP_BBOX_IOU = 0.25         # spatial eşleşme için min IoU


class _UnionFind:
    """Basit union-find (disjoint-set) yapısı."""

    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return False
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1
        return True


class CrateAggregator:
    """
    3 kameradan gelen kasa tespitlerini birleştirip gerçek toplam kasa sayısını
    hesaplar.

    Mantık:
      1) Aynı barkod birden çok kamerada okunduysa → aynı kasadır (birleşir).
      2) Ortak barkodlar kameralar arası (dx, dy) ofseti verir.
         Tespit edilen ofset + medyan kasa boyutu kullanılarak, barkodu
         okunamamış komşu kasalar (ortak barkodun altı/üstü/yanı) da
         spatial eşleştirmeyle tekilleştirilir.
      3) Hiç ortak barkod yoksa (kameralar kesişmiyor) dedup yapılmaz,
         tüm tespitler kendine özgü sayılır.
    """

    def __init__(
        self,
        tolerance_ratio: float = DEDUP_TOLERANCE_RATIO,
        min_shared: int = DEDUP_MIN_SHARED,
        bbox_iou: float = DEDUP_BBOX_IOU,
    ):
        self.tolerance_ratio = tolerance_ratio
        self.min_shared = min_shared
        self.bbox_iou = bbox_iou

    # ---- yardımcılar ----
    @staticmethod
    def _bbox_iou(a, b):
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        ix1 = max(ax1, bx1)
        iy1 = max(ay1, by1)
        ix2 = min(ax2, bx2)
        iy2 = min(ay2, by2)
        iw = max(0.0, ix2 - ix1)
        ih = max(0.0, iy2 - iy1)
        inter = iw * ih
        if inter <= 0:
            return 0.0
        area_a = max(0.0, (ax2 - ax1)) * max(0.0, (ay2 - ay1))
        area_b = max(0.0, (bx2 - bx1)) * max(0.0, (by2 - by1))
        union = area_a + area_b - inter
        return inter / union if union > 0 else 0.0

    def _flatten(self, camera_results):
        """Tüm kasaları global listeye düzleştir."""
        crates = []
        for cam_id, res in enumerate(camera_results):
            if res is None:
                continue
            for k in res.get("kasalar", []) or []:
                x1, y1, x2, y2 = k["bbox"]
                w = max(1.0, float(x2 - x1))
                h = max(1.0, float(y2 - y1))
                bcs = [str(b).strip() for b in (k.get("barkodlar") or []) if b]
                crates.append({
                    "cam": cam_id,
                    "kasa_no": k.get("kasa_no"),
                    "bbox": (float(x1), float(y1), float(x2), float(y2)),
                    "cx": (float(x1) + float(x2)) / 2.0,
                    "cy": (float(y1) + float(y2)) / 2.0,
                    "w": w,
                    "h": h,
                    "barkodlar": bcs,
                })
        return crates

    def aggregate(self, camera_results):
        """
        camera_results: her kamera için process_image dönüşü (veya None).

        Döner:
          {
            "raw_total": int,        # kameraların tespit toplamı (çift sayımlı)
            "unique_total": int,     # tekilleştirilmiş gerçek kasa sayısı
            "duplicates": int,       # çift sayılan kasa adedi
            "unique_barcodes": int,  # tekil barkod sayısı
            "per_camera": [ {cam, kasa, barkod, okunamayan} ... ],
            "pair_offsets": { (a,b): {dx,dy,shared,tol_x,tol_y} },
            "shared_barcodes_total": int,
            "barkod_cakisan": int,   # barkod üzerinden tekilleştirilen adet
            "spatial_cakisan": int,  # spatial eşleştirmeyle tekilleştirilen
          }
        """
        crates = self._flatten(camera_results)
        n = len(crates)
        per_cam = []
        for cam_id, res in enumerate(camera_results):
            if res is None:
                per_cam.append({"cam": cam_id, "kasa": 0, "barkod": 0, "okunamayan": 0})
            else:
                per_cam.append({
                    "cam": cam_id,
                    "kasa": int(res.get("toplam_kasa", 0) or 0),
                    "barkod": int(res.get("okunan_barkod", 0) or 0),
                    "okunamayan": int(res.get("okunamayan_barkod", 0) or 0),
                })

        if n == 0:
            for c in per_cam:
                c["kasa_unique"] = 0
            return {
                "raw_total": 0,
                "unique_total": 0,
                "duplicates": 0,
                "unique_barcodes": 0,
                "per_camera": per_cam,
                "pair_offsets": {},
                "shared_barcodes_total": 0,
                "barkod_cakisan": 0,
                "spatial_cakisan": 0,
                "unique_per_camera": [0] * len(camera_results),
                "crates": [],
                "groups": [],
            }

        uf = _UnionFind(n)

        # 1) Barkod tabanlı birleştirme
        barkod_to_idx = {}
        for i, c in enumerate(crates):
            for bc in c["barkodlar"]:
                barkod_to_idx.setdefault(bc, []).append(i)

        barkod_cakisan = 0
        for bc, idxs in barkod_to_idx.items():
            # Tüm idxs'leri ilkine bağla. Farklı kamerada aynı barkod varsa dup.
            cams_set = {crates[i]["cam"] for i in idxs}
            if len(cams_set) > 1:
                # Her ekstra kamera temsilcisi kadar çakışma sayılır
                barkod_cakisan += len(idxs) - len(cams_set) + (len(cams_set) - 1)
                # Not: len(idxs) - 1 basitçe de doğru sayıyı verir; aşağıda bunu
                # tekrar hesaplayacağımız için bu değer sadece log/gösterim için.
            for j in idxs[1:]:
                uf.union(idxs[0], j)

        # 2) Kamera çiftleri için ortak barkoddan (dx, dy) ofseti çıkar
        pair_offsets = {}
        # Kamera-id → (kasa_idx, barkod) map'i
        per_cam_bc = {}
        for i, c in enumerate(crates):
            for bc in c["barkodlar"]:
                per_cam_bc.setdefault((c["cam"], bc), []).append(i)

        cams_in = sorted({c["cam"] for c in crates})
        for a in cams_in:
            for b in cams_in:
                if a >= b:
                    continue
                # a ve b'de aynı barkodu olan kasa çiftleri
                shared = []
                # a kamerasının barkodlarını bul
                a_barkodlar = {bc: idxs for (cam, bc), idxs in per_cam_bc.items() if cam == a}
                b_barkodlar = {bc: idxs for (cam, bc), idxs in per_cam_bc.items() if cam == b}
                for bc, a_idxs in a_barkodlar.items():
                    if bc in b_barkodlar:
                        # Her iki taraftan ilk eşleşmeyi kullan (aynı kasa iki kere okunmuş olabilir)
                        ia = a_idxs[0]
                        ib = b_barkodlar[bc][0]
                        shared.append((ia, ib))

                if len(shared) < self.min_shared:
                    continue

                dxs = [crates[ib]["cx"] - crates[ia]["cx"] for ia, ib in shared]
                dys = [crates[ib]["cy"] - crates[ia]["cy"] for ia, ib in shared]
                ws = [crates[ia]["w"] for ia, _ in shared] + [crates[ib]["w"] for _, ib in shared]
                hs = [crates[ia]["h"] for ia, _ in shared] + [crates[ib]["h"] for _, ib in shared]
                dx = float(np.median(dxs))
                dy = float(np.median(dys))
                tol_x = float(np.median(ws)) * self.tolerance_ratio
                tol_y = float(np.median(hs)) * self.tolerance_ratio
                pair_offsets[(a, b)] = {
                    "dx": dx,
                    "dy": dy,
                    "shared": len(shared),
                    "tol_x": tol_x,
                    "tol_y": tol_y,
                }

        # 3) Ofset bilinen her çift için spatial eşleştirme
        spatial_cakisan = 0
        for (a, b), off in pair_offsets.items():
            a_indices = [i for i, c in enumerate(crates) if c["cam"] == a]
            b_indices = [i for i, c in enumerate(crates) if c["cam"] == b]
            used_b = set()
            # b'nin kasalarının a koordinat sistemine düzeltilmiş merkezleri:
            # b'deki (cx_b, cy_b) ↔ a'daki (cx_b - dx, cy_b - dy)
            for ia in a_indices:
                ca = crates[ia]
                best = None
                best_score = None
                for ib in b_indices:
                    if ib in used_b:
                        continue
                    cb = crates[ib]
                    # BARKOD ÇELİŞKİSİ VETOSU: iki tarafın da barkodu okunmuş
                    # ve kesişmiyorsa bunlar KESİN farklı kasalardır — IoU ne
                    # kadar yüksek olursa olsun birleştirilmez. (Bu veto
                    # olmadan ince/istifli kasalarda komşu satırlar IoU≥eşik
                    # ile birbirine zincirlenip onlarca kasa tek sayılıyordu.)
                    barcodes_match = bool(
                        set(ca["barkodlar"]) & set(cb["barkodlar"])
                    )
                    if (
                        ca["barkodlar"]
                        and cb["barkodlar"]
                        and not barcodes_match
                    ):
                        continue
                    # b'nin a'ya projeksiyonu
                    pcx = cb["cx"] - off["dx"]
                    pcy = cb["cy"] - off["dy"]
                    if abs(pcx - ca["cx"]) > off["tol_x"]:
                        continue
                    if abs(pcy - ca["cy"]) > off["tol_y"]:
                        continue
                    # bbox IoU kontrolü (projekte edilmiş bbox ile)
                    bx1 = cb["bbox"][0] - off["dx"]
                    by1 = cb["bbox"][1] - off["dy"]
                    bx2 = cb["bbox"][2] - off["dx"]
                    by2 = cb["bbox"][3] - off["dy"]
                    iou = self._bbox_iou(ca["bbox"], (bx1, by1, bx2, by2))
                    score_d = abs(pcx - ca["cx"]) + abs(pcy - ca["cy"])
                    if iou < self.bbox_iou and not barcodes_match:
                        # Düşük örtüşmede yalnız aynı barkod birleşebilir;
                        # barkodsuz taraflar için yeterli kanıt yok.
                        continue
                    # En iyi adayı seç
                    if best_score is None or score_d < best_score:
                        best_score = score_d
                        best = ib
                if best is not None:
                    if uf.find(ia) != uf.find(best):
                        uf.union(ia, best)
                        spatial_cakisan += 1
                    used_b.add(best)

        # Gerçek tekil sayı
        roots = {uf.find(i) for i in range(n)}
        unique_total = len(roots)
        raw_total = n
        duplicates = raw_total - unique_total

        # Tekil barkod: farklı kök sayısı barkoda sahip olanlar için
        unique_barcodes = len({bc for bc in barkod_to_idx.keys()})
        # Kameralar arası ortak barkod (en az 2 kamerada geçen)
        shared_bc_total = sum(
            1
            for bc, idxs in barkod_to_idx.items()
            if len({crates[i]["cam"] for i in idxs}) > 1
        )

        # ----------------------------------------------------------------
        # Her kameranın kendi sorumluluğundaki tekil kasa sayısı
        # ----------------------------------------------------------------
        # Mantık: union-find sonrası her kasa bir gruba (kök) bağlanır;
        # kasalar arası birleşmiş bir grup tek bir fiziksel kasayı temsil
        # eder. Bu fiziksel kasayı, gruba dahil olan kameralardan en
        # küçük cam_id'lisine "atıyoruz". Böylece sum(unique_per_camera)
        # tam olarak unique_total'e eşit olur.
        unique_per_cam = [0] * len(camera_results)
        groups: dict[int, list[int]] = {}
        for i in range(n):
            groups.setdefault(uf.find(i), []).append(i)
        for _, members in groups.items():
            owner = min(crates[m]["cam"] for m in members)
            if 0 <= owner < len(unique_per_cam):
                unique_per_cam[owner] += 1
        for i, c in enumerate(per_cam):
            c["kasa_unique"] = unique_per_cam[i] if i < len(unique_per_cam) else 0

        return {
            "raw_total": raw_total,
            "unique_total": unique_total,
            "duplicates": duplicates,
            "unique_barcodes": unique_barcodes,
            "per_camera": per_cam,
            "pair_offsets": pair_offsets,
            "shared_barcodes_total": shared_bc_total,
            "barkod_cakisan": barkod_cakisan,
            "spatial_cakisan": spatial_cakisan,
            "unique_per_camera": unique_per_cam,
            "crates": crates,
            "groups": list(groups.values()),
        }
