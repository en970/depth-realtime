# Şu an

Görev 1-3 ve 8'in ilk hâli bitti: ölçüm düzeneği, taban çizgisi, kamera rehberli
kenar korumalı yükseltme ve Kalite modu (518 px + tam rehber) yayında değil ama
yerelde çalışıyor. Kalite araştırması (workflow wf_ae9c86ae-7be) hâlâ çalışıyor;
sonucu gelince yüzey gölgelendirme ve büyük model kararları verilecek.

# Kullanıcının açık şikayetleri (2026-08-11)

1. "arka planı anlamıyor, arkamı siyah yapıyor" — ÇÖZÜLDÜ: uyarlanabilir ton
   eğrisi (kontrast sınırlı histogram eşitleme) eklendi, "Distance contrast"
   kaydırıcısı, varsayılan 0.70. Uzak alan detayı +%77.
2. "FPS çok yavaş" — KISMEN. Ölçüldü: WebGPU'da fp16 19.4 fps, q4f16 12.4,
   q8 0.8 — yani mevcut varsayılan (fp16) zaten en hızlısı, dtype tarafında
   kazanç YOK. Normalizasyon tek geçişe indirildi. Kullanıcının makinesinde
   neden yavaş olduğu HÂLÂ BİLİNMİYOR: hangi backend'e düştüğü (WebGPU mu WASM
   mı) ve kaç fps aldığı sorulmalı; arayüzde Runtime bölümünde yazıyor.
3. "daha kompakt" — AÇIK. İlk yükleme ~24-55 MB.
4. "default side by side" — Reset düğmesi eklendi; varsayılan zaten split,
   kullanıcının localStorage'ında stacked kayıtlıydı.

Repo artık PUBLIC.

# Sıradaki adım

DA3-small kuantalama. Araştırma bulgusu aşağıda; özet: DA3-small hesap maliyeti
nötr, uzun menzilde belirgin daha iyi (ETH3D δ1 86.5 -> 98.6), Apache-2.0, ama
depoda sadece fp32 var (105.3 MB). Kuantalanmadan geçiş indirmeyi masaüstünde
2.1x, mobilde 5.5x büyütür — kullanıcının "kompakt olsun" isteğine ters.

Adımlar:
1. Kaggle/Colab notebook ile fp16 + q8 + q4f16 export (GPU gerekmiyor).
   DİKKAT: DA3 kafasındaki Exp düğümü fp32 kalmalı (op_block_list=["Exp"]),
   yoksa taşma. Notebook kodu araştırma bulgusunda.
2. Çıktıları kullanıcının HF hesabına, transformers.js'in beklediği dosya
   adlarıyla koy (fp16 -> model_fp16.onnx, q8 -> model_quantized.onnx,
   q4f16 -> model_q4f16.onnx).
3. Kodda üç değişiklik: MODEL_ID, toTensor rank 5 ([1,1,3,H,W]), ve normalise()
   içinde ters çevirme (DA3 büyük = UZAK).
4. DA3'e geçilirse ton eğrisi KAPATILMALI (ölçüldü: DA3'te dL* 50.2 -> 41.1).
5. `--vary` ile DA2/DA3 karşılaştırması yapılamaz (model değişimi oturum
   gerektirir); iki ayrı koşuda, aynı sahnede, aynı anda ölçülmeli.

Kullanıcı onayı gerekiyor: HF hesabına model yüklemek onun hesabında kalıcı bir
değişiklik. Kaggle token'ı da yenilenmeli (eski token sohbette ifşa oldu).

Sonra görev 5: eş derinlik kontur çizgileri. Compose shader'ında, LUT aramasından
hemen önce; `fract(t * bandCount)` üzerinden `fwidth` ile antialiaslı ince
çizgi. Açılıp kapanabilir olmalı (View grubunda switch).

Sonra görev 6 (salınımlı 3B) ve 7 (nokta bulutu). Salınım için araştırma notu:
renklendirilmiş derinlik haritasını kaydırmak İŞE YARAMIYOR (dokusu yok);
parallax ancak KAMERA görüntüsünü derinlikle çarpıtınca güçlü. Backward
(inverse) warp kullanılırsa disocclusion deliği hiç oluşmuyor, bedeli gizli
yüzeylerin gerilmesi. Genlik <= %1.5 genişlik, pointer/gyro ile sürülmesi
otomatik salınımdan daha güçlü.

# Tamamlananlar

- `tests/quality.mjs` yazıldı: sabit y4m girdiyle kenar hizalama, yüksek frekans
  ve zamansal kayma ölçüyor; `--save` / `--compare` ile kayıt tutuyor
- `--vary` eklendi: tek oturumda çoklu varyant, ileri+geri sıra ile sürüklenme
  dengelemesi. Koşular arası karşılaştırma GÜVENİLMEZ (aynı ayarla edge 3.9-7.5);
  karar verilecek her karşılaştırma `--vary` ile yapılmalı.
  Kullanım: `SCENES=demo20 node tests/quality.mjs --vary tone=0,0.4,0.7,1`
  Değiştirilebilir ayarlar: tone, structure, guide, smooth, res.
- Taban çizgisi `.continuum/quality/baseline.json`: edgeAlignment 5.891,
  highFrequency 24.15, temporalDrift 7.158, 9.6 fps (ölçüm modunda res=350 sabit,
  adaptive kapalı, üç sahne ortalaması)
- Kamera rehberli kenar korumalı yükseltme (5x5 cross-bilateral) eklendi:
  `guide` slider'ı, varsayılan 0.85; sigma URL'den (`sigma=`) ayarlanabilir.
  Ölçüm: edgeAlignment 5.78 -> 6.38 (+%10), fps 11.75 -> 11.4
- Kalite modu eklendi: 518 px girdi + tam rehber + adaptif kapalı, ~5 fps
- Uyarlanabilir ton eğrisi eklendi (kontrast sınırlı histogram eşitleme,
  worker'da kuantalama öncesi, kareler arası EMA 0.05). Uzak alan detayı
  6.30 -> 11.15 (+%77), yayılım 12.25 -> 22.52 (+%84).
- quality.mjs'e farGradient ve farSpread metrikleri eklendi: mevcut metrikler
  yakın özneye baskın olduğu için arka plan sorununu HİÇ göremiyordu.
- MİMARİ: iki geçişe ayrıldı. Compose geçişi (rehberli yükseltme + kabartma
  gölgeleme + renk) çıkarım başına bir kez, display geçişi her karede sadece
  iki hazır kareyi karıştırıyor. Ölçüm (structure=0.6, iki sahne):
  fps 12.15 -> 21.75, highFrequency 34.4 -> 41.0, drift 4.24 -> 2.74.
  Gölgeleme AÇIKKEN fps, gölgeleme KAPALI taban çizgisinden (18.55) bile yüksek.
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

# Araştırma bulgusu: DA3-small (2026-08-11, workflow wf_e8e81111-fb3)

Ajan iki modeli de yerel onnxruntime ile aynı görüntülerde çalıştırdı:
- Arka planın siyah olması MODELİN değil NORMALİZASYONUN suçu. Ölçüm: mevcut
  lineer p2/p98 ile arka planda dL*=23.5 (kötü sahnede 9.7, 256 tonun 31'i).
  Log-disparity dL*=48.1 (+%105), DA3-small dL*=50.2 (+%114). Yani DA3'ün arka
  plan kazancının neredeyse tamamı model değiştirmeden alınabilir — ki alındı.
- DA3-small (onnx-community/depth-anything-v3-small, Apache-2.0) hesap maliyeti
  NÖTR: 224 px'te DA2'nin 0.88x'i (daha hızlı), 322'de 1.01x, 518'de 1.12x.
  ETH3D δ1 86.5 -> 98.6 (uzun menzil), ama NYU 97.9 -> 97.4 (yakın menzil hafif
  geriler). Bedava confidence haritası çıkarıyor.
- ENGEL: depoda SADECE fp32 var (105.3 MB). Mevcut indirmeler 49.6 MB (fp16) /
  19.1 MB (q4f16). Kuantalama yapılmadan geçiş şikayet 2 ve 3'ü kötüleştirir.
- DA3 ÇIKTISI DERİNLİK (büyük = UZAK), DA2'nin TERSİ. Geçilirse normalise()
  içinde ters çevirme şart, ve log/ton eğrisi DA3'te UYGULANMAMALI (ölçüldü:
  dL* 50.2 -> 41.1, yani kötüleşiyor).
- DA3 girdisi RANK 5: [B, num_images, 3, H, W]. toTensor() rank 4 üretiyor.
- Kuantalama Kaggle'da GPU bile gerektirmiyor; ajan tam notebook kodu verdi
  (onnxconverter_common float16, op_block_list=["Exp"] — DA3 kafasındaki Exp
  fp32 kalmalı, taşma koruması). Ölçülen q8 kalite kaybı: r=0.995.

# Ölçülmüş bulgular

- AYAR TARAMALARI (eşleştirilmiş, tek oturum, demo20, ileri+geri ortalaması):
  tone:      0 -> edge 5.12 far 6.35 drift 8.16
             0.4 -> edge -%20, far +%70
             0.7 -> edge -%15, far +%73, drift 5.34 (EN DÜŞÜK)  <- varsayılan
             1.0 -> edge -%31, far +%109, drift 14.30 (kararsız)
    Sonuç: 0.7 optimum. 1.0 arka planı daha çok açıyor ama titreme üçe katlanıyor
    ve ön plan kenarları belirgin bozuluyor.
  structure: 0 -> highFreq 12.8
             0.3 -> +%54
             0.6 -> +%76  <- varsayılan
             1.0 -> +%120, edge +%32
    Sonuç: yapı miktarı tekdüze artıyor, fps sabit. edge metriği bu taramada
    gürültülü (monoton değil), o yüzden 0.6 -> 1.0 kararı ölçümle verilemedi;
    görsel değerlendirme gerekiyor.
- WebGPU'da dtype hızı (ölçüldü, tek sahne): fp16 19.4 fps, q4f16 12.4 fps,
  q8 0.8 fps. Araştırmanın "q8 2x hızlı" bulgusu CPU/WASM içindir, WebGPU'da
  TERSİ geçerli. Varsayılan fp16 doğru.

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

# Denenip reddedilenler

- Ton histogramında adım (stride 3): maliyeti üçte bire indiriyor ama örneklem
  gürültüsü doğrudan eğriye giriyor; genel parlaklık kayması neredeyse iki
  katına çıktı. Her piksel taranıyor.

- Gölgelemeyi alanın kendi çözünürlüğünde (350x196) hesaplayıp ekranda büyütmek:
  fps 21.6 verdi ama gölgelemenin keskinliğini yok etti (highFrequency 34.4 ->
  7.3). Gölgeleme, rehberli yükseltmenin ürettiği keskin kenarlardan
  hesaplanmalı; kazanç çözünürlüğü düşürmekten değil, SIKLIĞI düşürmekten gelir.
- Gölgeleme terimlerini rehberli değerden (t) hesaplamak: drift %22 iyileşti ama
  edgeAlignment %8 düştü. Ana hedef kenar hizalaması olduğu için ham örnek
  korundu.
- Render hedefi olan dokuyu sampler birimine bağlı bırakmak: geri besleme
  döngüsü, çizim tanımsız. Pratikte sessizce düz bir kazanç üretti (gölgeleme
  hiç görünmedi). Compose öncesi ilgili birimler `null`'a bağlanıyor.
- İki geçiş de aynı vertex aşamasını kullanamaz: compose FBO'ya yazarken v'yi
  çeviriyor, display tekrar çevirirse görüntü baş aşağı olur. tests/renderer.mjs
  bunu yakaladı.

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

2026-08-11 (ölçüm düzeneği eşleştirilmiş karşılaştırmaya geçirildi)
