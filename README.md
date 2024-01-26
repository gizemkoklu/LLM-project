Bu proje Create a Large Language Model from Scratch with Python – Tutorial (https://youtu.be/UU1WVnMk4E8?si=Fgj6Tle9XatWg2fq) ile çalışarak hazırlanmış içeriği türkçe açıklamalarla anlatılmıştır.

Proje ilk olarak pyTorch anlatımı ile başlamaktadır.

Daha sonra Bigram Tutorial'ı içermektedir.

# Transformatör Mimarisi Yapısı

file:///home/gizem/Pictures/Screenshots/Screenshot%20from%202024-01-25%2014-50-33.png (Attention Is All You Need - https://arxiv.org/pdf/1706.03762.pdf)

1- Giriş Gömme (Input Embedding): İlk adım, metni kelime seviyesinde temsil etmektir. Her kelimenin gömme vektörü, genellikle öğrenilebilir parametrelerle başlar. Bu vektörler, kelimenin dil içindeki anlamını temsil eder.

2- Pozisyonel Kodlama (Positional Encoding): Giriş metni, sırasal bir yapıya sahip olduğu için transformatör modeli, kelime sırasının model tarafından öğrenilememesi nedeniyle pozisyonel kodlama ekler. Pozisyonel kodlama, her kelimeye bir pozisyon bilgisi ekleyerek sıralı yapının model tarafından dikkate alınmasını sağlar.

3-Kodlayıcı (Encoder) Katmanları: Giriş gömme ve pozisyonel kodlama sonrasında, metin kodlayıcı katmanlarına geçilir. Bu katmanlar, giriş metnindeki bilgileri hiyerarşik olarak işler. Her bir kodlayıcı katmanı şu adımları içerir:

     1-Çok Başlı Dikkat (Multi-Head Attention): Her kelimenin diğer kelimelere olan dikkatini hesaplar. Bu, her kelimenin çevresindeki bağlamı anlamasına yardımcı olur.

     2-Katman Normalizasyon (Layer Normalization): Her başlıkta hesaplanan dikkat matrisinin çıktısını normalize eder.

     3-İleri Besleme Ağları (Feed-Forward Networks): Her kelimenin temsilini daha karmaşık özelliklere dönüştürmek için kullanılır.

     4-Toplamsal Bağlam (Residual Connection): Bu adım, her katmanın çıktısına girişe olan doğrudan bağlantıyı sağlar, bu da ağın daha iyi öğrenmesine yardımcı olur.

4- Toplam Çıkış:Kodlayıcı katmanlarından geçtikten sonra, elde edilen çıktılar bir dizi vektör temsilini içerir. Bu vektör temsilleri, giriş metni üzerinde yüksek düzeyde bilgi taşıyan bir temsilasyon oluşturur.

5-Çıkış Katmanları: Bu aşamada, modelin spesifik görevine bağlı olarak çıkış katmanları eklenir. Örneğin, çeviri görevinde hedef dilin kelimelerini oluşturmak için kullanılır.

6-Kayıp Fonksiyonu (Loss Function) ve Geri Yayılım (Backpropagation): Modelin tahminleri ile gerçek etiketler arasındaki kaybı ölçen bir kayıp fonksiyonu kullanılır. Ardından, geri yayılım algoritması kullanılarak bu kayıp geriye doğru iletilir ve modelin parametreleri, gradyan inişi kullanılarak güncellenir.
