# Şu an

Görev 1-3 ve 8'in ilk hâli bitti: ölçüm düzeneği, taban çizgisi, kamera rehberli
kenar korumalı yükseltme ve Kalite modu (518 px + tam rehber) yayında değil ama
yerelde çalışıyor. Kalite araştırması (workflow wf_ae9c86ae-7be) hâlâ çalışıyor;
sonucu gelince yüzey gölgelendirme ve büyük model kararları verilecek.

# Sıradaki adım

Görev 4: yüzey gölgelendirme (normal-from-depth + yapay ışık) fragment shader'a.
`src/lib/depth-renderer.ts` içinde renk aramasından hemen önce, derinlik
alanından merkezi farkla normal hesapla ve Lambert terimiyle parlaklığı modüle
et; renk skalasının sırası bozulmamalı (legend hâlâ geçerli kalmalı). Kontrolü
`index.html` içindeki Model grubuna slider olarak ekle, ölçümü
`SCENES=demo01,demo20 node tests/quality.mjs` ile yap.

# Tamamlananlar

- `tests/quality.mjs` yazıldı: sabit y4m girdiyle kenar hizalama, yüksek frekans
  ve zamansal kayma ölçüyor; `--save` / `--compare` ile kayıt tutuyor
- Taban çizgisi `.continuum/quality/baseline.json`: edgeAlignment 5.891,
  highFrequency 24.15, temporalDrift 7.158, 9.6 fps (ölçüm modunda res=350 sabit,
  adaptive kapalı, üç sahne ortalaması)
- Kamera rehberli kenar korumalı yükseltme (5x5 cross-bilateral) eklendi:
  `guide` slider'ı, varsayılan 0.85; sigma URL'den (`sigma=`) ayarlanabilir.
  Ölçüm: edgeAlignment 5.78 -> 6.38 (+%10), fps 11.75 -> 11.4
- Kalite modu eklendi: 518 px girdi + tam rehber + adaptif kapalı, ~5 fps
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

# Ölçülmüş bulgular

- Çözünürlük 518'in ÜSTÜ ZARARLI: 644'te yüz yapısı düzleşiyor, ayrıntı
  kayboluyor. Model 518'de eğitilmiş; üstü dağılım dışı. LADDER 518'de bitiyor,
  slider max 518. Bu bir kalite düğmesi değil, uçurum.
- Zamansal yumuşatma (smooth=0.35) ayrıntıya ZARAR VERMİYOR: edgeAlignment
  6.34 (smooth=0) -> 6.42 (0.35), drift 3.80 -> 2.39. Önceki "zarar veriyor"
  sonucu kararsız ölçümden gelmişti; medyan alma bunu düzeltti.
- Rehber sigma taraması (tek sahne, 3 tekrar, gürültülü): 0.03 en yüksek
  edgeAlignment ama drift 4x; 0.25 çok kararlı ama filtre düz gauss'a dönüyor.
  0.06 makul denge olarak seçildi. Daha güvenilir tarama için REPEATS>=7 ve
  en az 2 sahne gerekir.
- Ölçüm gürültüsü ayarlara göre değişiyor: spread %4 ile %141 arasında. Küçük
  farklar (<%10) tek koşuyla değerlendirilemez.

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
