# i7-14 · TTO Sayım Süre Testi

TTO palet sayımının adımlarını (YOLO kasa tespiti → barkod → filtre →
tekilleştirme → OCR) kamera olmadan, 6 kayıtlı görüntü üzerinde çalıştırır ve
her adımın süresini ölçer. Amaç: aynı 6 görüntüyü farklı bilgisayarlarda
çalıştırıp süreleri karşılaştırmak.

`gorseller/` içindeki 6 görüntü 07.09.2026 17:02 paletidir (6 kamera, ~280
kasa). Referans makinede (HP ZBook, Core Ultra 7 155H, RTX 2000 Ada Laptop)
ölçülen değerler en altta.

## Kurulum (Windows 11)

1. **Python 3.11** — https://www.python.org/downloads/ (3.11.x). Kurulumda
   "Add python.exe to PATH" ve "py launcher" işaretli olsun. 3.12+ ile
   PaddleOCR/Paddle uyumsuz.
2. **NVIDIA sürücüsü** güncel olsun (OCR ekran kartında çalışır).
3. `kurulum.bat` çalıştırın (~5-10 dk, internet gerekir). Sonunda
   `paddle CUDA: True | GPU sayisi: 1` yazmalı.
4. **Aremak Kod Okuyucu** (isteğe bağlı): `C:\AremakCodeReaderCN` klasörüne
   kurulu olmalı ve USB dongle takılı olmalı. Yoksa uygulama açılır ama
   "Aremak" seçeneği kapalı kalır.
5. **HALCON** (isteğe bağlı): MVTec HALCON 26.05 Progress kurulu ve lisanslı
   olmalı (`mvtec-halcon` Python paketi requirements'ta). Yoksa "HALCON"
   seçeneği kapalı kalır.
6. `baslat.bat` ile açın. İlk açılışta PaddleOCR modelleri (~200 MB) indirilir;
   sonraki açılışlar ~30 sn hazırlık ister ("Hazır" yazana kadar bekleyin).

Uygulama VS Code / başka bir Python ile açılırsa kendini `py -3.11` ile
yeniden başlatır.

## Kullanım

1. **↑ 1–6 Görüntü Seç** → `gorseller/` klasöründeki 6 dosyayı K1…K6 sırasıyla
   seçin (dosya adları sıralı).
2. Üst çubuk:
   - **Barkod:** Kapalı / Aremak / HALCON
   - **6 kamera aynı anda (paralel barkod):** işaretliyse her kamera kendi
     okuyucusuyla aynı anda okunur; değilse TTO'daki gibi sırayla.
   - **OCR dahil:** barkod açıkken yalnız barkodu okunamayan kasalar,
     kapalıyken tüm kasalar OCR'a girer.
   - **Görüntüleri kaydet:** TTO'nun sayımda yazdığı ham kare / kesit /
     işaretli görüntü kayıtlarını da yapar (TTO ile aynı iş yükü).
3. **▶ ÇALIŞTIR**. Kronometre iş bitince durur; sağdaki tablo adım adım süreyi
   ve payını gösterir. Her görüntü kartında o kameranın barkod süresi yazar.
4. Her çalıştırmanın raporu `sonuclar/<tarih>/sure_raporu.json` dosyasına
   yazılır (adım süreleri, kasa/barkod sayıları, motor ve mod).

Karşılaştırılabilir ölçüm için her makinede aynı 4 koşuyu yapın:

| # | Barkod | Paralel | OCR | Kaydet |
|---|---|---|---|---|
| 1 | Aremak | ✗ | ✓ | ✓ | (bugünkü TTO ile aynı)
| 2 | Aremak | ✓ | ✓ | ✓ |
| 3 | HALCON | ✗ | ✓ | ✓ |
| 4 | HALCON | ✓ | ✓ | ✓ |

Ekran kartını ısıtmak için ilk koşuyu bir kez tekrarlayın; ilk çalıştırma
her zaman biraz yavaştır.

## Referans ölçüm — HP ZBook Power G11 (Ultra 7 155H, 16 çekirdek)

Aynı 6 görüntü, OCR + kayıt açık, 22-24.09.2026:

| Barkod motoru | Mod | Barkod okuma | Toplam | Sonuç |
|---|---|---|---|---|
| Aremak | sırayla | 40,1 sn | **44,6 sn** | 246 kasa · 243 barkod · 3 okunamayan |
| Aremak | 6 kamera aynı anda | 10,9 sn | **15,3 sn** | aynı |
| HALCON | sırayla | 5,9 sn | **10,2 sn** | aynı |
| HALCON | 6 kamera aynı anda | 1,4 sn | **5,7 sn** | aynı |
| Kapalı | — | — | 3,5–4 sn | 282 kutu (tekilleştirme yok) |

Barkod dışındaki adımlar (YOLO işlemcide ~0,7 sn, kayıt ~1,5 sn, OCR ~0,5 sn)
her koşuda ~4 sn. Not: YOLO bu pakette işlemcide çalışır (torch CPU
derlemesi); TTO'da da öyle.

## Dosyalar

| Dosya | Ne |
|---|---|
| `sure_testi.py` | Uygulama |
| `tto_filtre.py` | TTO `app.py` arka plan / boy / sütun filtresi — birebir kopya |
| `tekillestirme.py` | TTO `CrateAggregator` (kameralar arası tekilleştirme) — birebir kopya |
| `tto_tespit.py` | YOLO eşiği (0,65), model yolu, barkod ön işleme (`camera_gui_v2`'den) |
| `ocr_reader.py`, `ocr_engine.py`, `app_ui.py` | TTO'nun PaddleOCR seri okuma kodu |
| `aremak_barkod/` | Aremak .NET SDK Python sarmalayıcısı |
| `models/V8LAST.pt` | TTO'nun kullandığı YOLO modeli |
| `gorseller/` | Test paleti, 6 kamera |
| `theme.py`, `widgets.py` | Arayüz |

## Bilinen noktalar

- **Blackwell ekran kartları (RTX 50xx, RTX PRO Blackwell):** `kurulum.bat`
  Paddle'ın CUDA 12.6 derlemesini kurar; bu kartlarda çalışmayabilir. Kontrol
  satırında `GPU sayisi: 0` çıkarsa ya OCR işlemcide çalışır (kasa başına
  ~5-10× yavaş) ya da Paddle'ın CUDA 12.8+ derlemesi kurulmalıdır:
  `py -3.11 -m pip install paddlepaddle-gpu==3.2.2 -i https://www.paddlepaddle.org.cn/packages/stable/cu129/`
- Aremak paralel okumada tek dongle ile 6 okuyucu açılır; lisans şartlarını
  Aremak'a teyit ettirin.
- Aremak dongle'ı takılmadan açıldıysa "↻ Dongle'ı dene" ile yeniden
  sınanabilir, uygulamayı kapatmak gerekmez.
