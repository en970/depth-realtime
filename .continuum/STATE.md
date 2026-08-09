# Şu an

Görev 1 ve 2 bitti: ölçüm düzeneği kuruldu ve taban çizgisi kaydedildi.
Kalite araştırması (5 eksenli workflow, run wf_ae9c86ae-7be) arka planda
çalışıyor; sonucu Faz 1'in uygulama sırasını belirleyecek.

# Sıradaki adım

Araştırma sonucu geldiğinde Görev 3'e başla: kenar korumalı yükseltme
(cross-bilateral / guided filter) `src/lib/depth-renderer.ts` içindeki fragment
shader'a. Kamera görüntüsünü rehber olarak shader'a bağlamak gerekiyor — şu an
video ayrı bir DOM elementi, WebGL context'inde dokusu yok; her karede
`texImage2D(video)` maliyetini ölç.

Araştırma gelmeden önce yapılabilecek ayrı iş: sabit girdide ölçülen 7.16/255
zamansal kaymanın kaynağını bul (aşağıya bak) — bu muhtemelen gerçek bir hata.

# Tamamlananlar

- `tests/quality.mjs` yazıldı: sabit y4m girdiyle kenar hizalama, yüksek frekans
  ve zamansal kayma ölçüyor; `--save` / `--compare` ile kayıt tutuyor
- Taban çizgisi `.continuum/quality/baseline.json`: edgeAlignment 5.891,
  highFrequency 24.15, temporalDrift 7.158, 9.6 fps (ölçüm modunda res=350 sabit,
  adaptive kapalı, üç sahne ortalaması)
- Temel uygulama çalışıyor ve canlıda: https://en970.github.io/depth-realtime/
- Yön (dikey ters), en-boy oranı (cover formülü tersti) ve kadraj (panel kutusu
  kamera oranına uymuyordu, yatay FOV'un ~%64'ü atılıyordu) düzeltildi
- Performans: 10 -> 28 fps, çıkarım 74 -> 38 ms, ön işleme 5.4 -> 0.33 ms
- Üç gizli hata düzeltildi: in-flight kilitlenmesi, ölü adapt(), zehirlenen
  transformers.js promise zinciri (worker tek seferlik restart)
- Git geçmişi `en970 <enesozile@gmail.com>` kimliğine yeniden yazıldı (yerelde)

# Açık soru: sabit girdide titreme

`tests/quality.mjs` girdiyi tamamen donduruyor (tek kareli y4m), buna rağmen
ardışık okumalar arasında piksel başına ortalama 7.16/255 fark ölçülüyor
(demo01 8.5, demo20 3.1, hf-depth 9.8). Deterministik bir modelde sabit girdi
sabit çıktı vermeli. Olası kaynaklar, denenme sırası:
1. `smooth=0` ile piksel EMA kapalıydı; 0.35 ile ölçüp farkı gör.
2. Renderer'ın iki doku arası `mix` geçişi — iki doku aynıysa fark yaratmamalı.
3. Capture yolundaki `drawImage(video)` zamanlaması: video karesi ile canvas
   içeriği arasında yarım kare kayma olabilir.
4. Normalizasyon sınırlarının EMA'sı sabit girdide yakınsamıyor olabilir.
Bu titreme, kullanıcının "ayrıntı yok" izlenimini besliyor olabilir: kararsız
bir alan gözde bulanık algılanır.

# Bilinmesi gerekenler

- Başsız Chromium'un yazılım WebGPU adaptöründe ONNX Runtime ilk çıkarımda
  asılıyor. Bu ortam kaynaklı; testler `HEADED=1` ile gerçek GPU'da koşulmalı.
  Başsız koşu WASM yedeğini test eder, o da faydalı.
- Playwright kalıcı profil kullanılıyor:
  `/private/tmp/claude-501/-Users-enes-Developer-depth-realtime/914d624e-f8a0-4f36-a0c6-39e63e8a2cd2/scratchpad/chrome-profile`
  Model ağırlıkları orada önbellekte; temiz profil her koşuda 50 MB indirir.
- `tests/renderer.mjs` dev server ister (`npm run dev`, port 5173);
  `tests/smoke.mjs` preview ister (`npm run preview`, port 4173).
- Dev server açıkken kaynak dosya değişirse HMR sayfayı yeniler ve Playwright
  "Execution context was destroyed" hatası verir; build'den sonra 2-3 sn bekle.
- GitHub Actions kuyruğu tıkanabiliyor; workflow artık `cancel-in-progress: true`.
- ONNX Runtime session options denendi ve hepsi elendi (gerekçe
  `src/lib/depth-worker.ts` içinde yorum olarak duruyor) — tekrar denenmesin.
- Git geçmişi yeniden yazıldı ve force push edildi; GitHub'daki 11 commit'in
  hepsi artık en970 hesabına bağlı (doğrulandı). `--force-with-lease` "stale
  info" derse önce `git fetch origin`, sonra
  `--force-with-lease=main:$(git rev-parse origin/main)` kullan — filter-branch
  remote-tracking ref'i de yeniden yazdığı için lease bayatlıyor.
- Ölçüm sahneleri (y4m) scratchpad'de, repoda değil:
  `.../scratchpad/refdata/{demo01,demo20,hf-depth}.y4m`. Kaybolursa
  Depth-Anything-V2 deposundaki assets/examples/*.jpg dosyalarından
  `ffmpeg -vf scale=1280:720:force_original_aspect_ratio=increase,crop=1280:720
  -pix_fmt yuv420p -frames:v 1 -f yuv4mpegpipe` ile yeniden üretilir.
- upload.wikimedia.org bu ortamdan erişilemiyor (HTTP 400); raw.githubusercontent
  ve huggingface.co erişilebilir.

# Son güncelleme

2026-08-09
