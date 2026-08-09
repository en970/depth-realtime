# Hedef

Derinlik haritasında gözle görülür şekilde daha fazla ayrıntı ve daha güçlü bir
üç boyut algısı olacak. Parmak aralığı, sandalye kenarı, kablo gibi ince yapılar
derinlik haritasında ayırt edilebilecek; kenarlar nesne siluetlerine oturacak;
bakan kişi yüzeylerin kıvrımını sezgisel olarak kavrayacak. Varsayılan görünüm
akıcı kalacak (25-30 fps), bunun yanında isteğe bağlı bir Kalite modu bulunacak.
Her iyileştirme, öznel izlenimle değil ölçülen bir metrikle doğrulanacak.

# Görevler

- [ ] 1. Ölçüm düzeneği: sabit test görüntüsüyle tekrarlanabilir koşu
      (Chrome'a y4m besleme), kenar hizalama ve yüksek frekans metrikleri,
      zamansal kararlılık ölçümü, tests/quality.mjs
- [ ] 2. Referans çıktı: mevcut hâlin metrikleri kaydedilir (taban çizgisi)
- [ ] 3. Kenar korumalı yükseltme: kamera luma'sını rehber alan cross-bilateral
      veya guided filter, WebGL2 tek geçiş. En yüksek getirili ayrıntı adımı.
- [ ] 4. Yüzey gölgelendirme: derinlikten normal, yapay ışıkla gölgeleme;
      renk skalasının bilimsel okunabilirliğini bozmadan
- [ ] 5. Eş derinlik kontur çizgileri (açılıp kapanabilir)
- [ ] 6. Salınımlı 3B (wiggle) görünüm modu
- [ ] 7. 3B nokta bulutu görünümü
- [ ] 8. Kalite modu: yüksek girdi çözünürlüğü + büyük model
      (Depth Anything V1 Large veya dengi, Apache-2.0), isteğe bağlı indirme
- [ ] 9. Her adımda: build + tests/renderer.mjs + tests/smoke.mjs + quality.mjs,
      sonra commit ve deploy; canlıda doğrulama

# Bitti sayılma ölçütü

- `npm run build` temiz
- `node tests/renderer.mjs`, `node tests/smoke.mjs`, `node tests/quality.mjs`
  hepsi PASS
- quality.mjs'in kenar hizalama metriği taban çizgisine göre belirgin artmış,
  zamansal kararlılık kötüleşmemiş
- Varsayılan görünüm hâlâ >= 25 fps; Kalite modu ayrı ve isteğe bağlı
- Canlı sitede (https://en970.github.io/depth-realtime/) doğrulanmış

# Sınırlar

- Lisans: proje MIT. Yalnızca Apache-2.0 / MIT / BSD / CC0 model ve kod.
  CC-BY-NC kesinlikle kullanılmaz (Depth Anything V2 Base/Large bu yüzden yasak).
- Görselleştirme WebGL2 kalır; WebGPU yalnızca çıkarım için (iOS 18-25'te yok).
- GitHub Pages: sunucu yok, COOP/COEP yok, çok iş parçacıklı WASM yok.
- Video elementi hiçbir zaman display:none olmaz (WebKit kare üretmeyi durdurur).
- Git kimliği daima `en970 <enesozile@gmail.com>`; commit'te kimlik override
  edilmez. `oze05607@gmail.com` ve `enesoz970` asla kullanılmaz.
- Kullanıcının kararları: Kalite modu ayrı olacak (varsayılan akıcı kalır);
  dört algı tekniğinin dördü de istendi; büyük model yalnızca Kalite modunda.
- Beklemede: geçmiş yeniden yazıldı ama `git push --force-with-lease origin main`
  kullanıcı onayı bekliyor. Onaysız force push denenmez.
