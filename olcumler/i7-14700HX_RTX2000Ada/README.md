# Ölçüm — Intel Core i7-14700HX (20 çekirdek), RTX 2000 Ada Laptop

Aynı 6 görüntü, OCR + kayıt açık, 25.09.2026. YOLO işlemcide (torch CPU),
OCR ekran kartında (paddle CUDA 12.6). HALCON bu makinede kurulu değil.

| Koşu | Barkod motoru | Mod | Barkod okuma | Toplam | Sonuç |
|---|---|---|---|---|---|
| 10:22:53 | Aremak | 6 kamera aynı anda | 9,1 sn | **18,0 sn** | 246 kasa · 243 barkod · 3 okunamayan |
| 10:24:48 | Aremak | 6 kamera aynı anda | 7,0 sn | **14,8 sn** | aynı |
| 10:32:32 | Aremak | 6 kamera aynı anda | 8,8 sn | **16,2 sn** | aynı |

Adım süreleri (sn, üç koşu):

| Adım | 1 | 2 | 3 |
|---|---|---|---|
| okuma | 0,71 | 0,66 | 0,69 |
| ham_kayit | 1,20 | 1,16 | 1,20 |
| yolo | 0,43 | 0,53 | 0,47 |
| barkod | 9,10 | 6,98 | 8,77 |
| filtre | 0,13 | 0,42 | 0,10 |
| kesit_kayit | 4,67 | 3,51 | 3,61 |
| tekil | 0,00 | 0,00 | 0,00 |
| ocr | 0,76 | 0,56 | 0,66 |

Referans (HP ZBook, Ultra 7 155H) aynı modda: barkod 10,9 sn · toplam 15,3 sn.

Ham raporlar: `*_sure_raporu.json`.
