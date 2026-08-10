# Şu an

Görev 1-3 ve 8'in ilk hâli bitti: ölçüm düzeneği, taban çizgisi, kamera rehberli
kenar korumalı yükseltme ve Kalite modu (518 px + tam rehber) yayında değil ama
yerelde çalışıyor. Kalite araştırması (workflow wf_ae9c86ae-7be) hâlâ çalışıyor;
sonucu gelince yüzey gölgelendirme ve büyük model kararları verilecek.

# Sıradaki adım

ÖNCELİK: gölgelemeyi iki geçişe ayır. Şu an Sobel'in 8 tap'i + bilateral'in 75
tap'i EKRAN çözünürlüğünde çalışıyor ve ölçülen bedel büyük: structure=0.6 ile
fps 18.55 -> 12.15 (-%34). Araştırmanın önerisi: gölgelemeyi alanın kendi
çözünürlüğünde (350x196) bir FBO'da hesapla, ekran geçişini sadece büyütmeye
bırak. Gölgelemenin taşıdığı bilgi alanın çözünürlüğünden ince olamaz, yani bu
bir kayıp değil. Beklenen: fps'in büyük kısmı geri gelir, mobilde kritik.

Sonra: görev 5 (kontur çizgileri), 6 (salınımlı 3B), 7 (nokta bulutu).

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
- Gate'li yüzey gölgelemesi (Sobel + half-Lambert, NNW ışık) eklendi:
  edgeAlignment 6.06 -> 7.42 (+%22), highFrequency 22.6 -> 34.4 (+%52),
  drift 6.13 -> 4.24. Bedeli: fps 18.55 -> 12.15
- Kabartma gölgelemesi (depth darkening + cavity) eklendi: mip zincirinden
  okunuyor, ayrı blur geçişi yok. Ölçüm: edgeAlignment 4.32 -> 5.97 (+%38),
  highFrequency 24.9 -> 45.4 (+%82), drift 8.78 -> 5.44. Görsel fark büyük:
  gölgelemesiz yüz düz beyaz leke, gölgelemeli burun/göz/çene hattı okunuyor.
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

- Kalite modu ön ayarı açılışta YENİDEN UYGULANMAMALI: uyguladığında URL'den
  gelen res/guide/structure değerlerini eziyordu ve iki karşılaştırma varyantı
  aynı ayarla çalışıyordu. Artık açılışta yalnızca UI durumu yansıtılıyor.
  Test scriptleri de her koşuda localStorage temizliyor.
- Gölgeleme harmanı LINEER ışıkta sınırlı kazanç olmalı (clamp 0.86-1.10).
  Ölçülmüş: matplotlib 'overlay' tam salınımla L* sıralamasını aralığın
  %52.5'inde TERSİNE çeviriyor, 'soft light' %43.1. Sınırlı kazançta belirsizlik
  ~%4, yani modelin kendi kare-arası gürültüsü mertebesinde.
- Gölgeleme cross-fade edilmiş değerden DEĞİL, yalnızca uCurr'dan hesaplanmalı:
  mix=0.5'te hareketli bir kenar iki yarım basamak olur ve çift sırt çizilir.
- Lisans: learnopengl SSAO kodu CC BY-NC, Blender cavity shader GPL. İkisi de
  bu projeye KOPYALANAMAZ; yalnızca tarif seviyesinde yeniden yazılabilir.
  Luft 2006 (akademik yöntem) ve matplotlib formülleri sorunsuz.

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

# Ölçüm düzeneğinin sınırı (önemli)

Chrome'un `--use-file-for-fake-video-capture` ile beslediği tek kareli y4m
GERÇEKTEN sabit değil: yakalanan karenin parmak izi bir koşu içinde %1.3
kayıyor (50895 -> 51594, monoton). Bunun sonucu zincirleme:
ham p2/p98 sınırları %4-24 oynuyor -> normalize çıktı kayıyor -> ölçüm gürültüsü.
Bu yüzden FARKLI KOŞULAR ARASINDA görsel karşılaştırma yapılamaz; A/B için tek
oturumda kontrolü yerinde değiştirmek gerekir (`localStorage.clear()` +
slider'a `input` olayı göndererek). `depthDiagnostics()` artık normLo/normHi,
rawLo/rawHi ve captureChecksum döndürüyor; kararsızlık şüphesinde önce bunlara
bak.

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
