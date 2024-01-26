Bu proje Create a Large Language Model from Scratch with Python – Tutorial (https://youtu.be/UU1WVnMk4E8?si=Fgj6Tle9XatWg2fq) ile çalışarak hazırlanmış içeriği türkçe açıklamalarla anlatılmıştır.

Proje ilk olarak pyTorch anlatımı ile başlamaktadır.

Daha sonra Bigram Tutorial'ı içermektedir.

# Transformatör Mimarisi Yapısı 
![Transformer - model architecture](https://github.com/gizemkoklu/LLM-project/assets/93999489/ce0c6caa-4c46-4fc7-b576-7d62c0e9d9ef)
(Attention Is All You Need - https://arxiv.org/pdf/1706.03762.pdf)

1- Giriş Gömme (Input Embedding): İlk adım, metni kelime seviyesinde temsil etmektir. Her kelimenin gömme vektörü, genellikle öğrenilebilir parametrelerle başlar. Bu vektörler, kelimenin dil içindeki anlamını temsil eder.

2- Pozisyonel Kodlama (Positional Encoding): Giriş metni, sırasal bir yapıya sahip olduğu için transformatör modeli, kelime sırasının model tarafından öğrenilememesi nedeniyle pozisyonel kodlama ekler. Pozisyonel kodlama, her kelimeye bir pozisyon bilgisi ekleyerek sıralı yapının model tarafından dikkate alınmasını sağlar.

3-Kodlayıcı (Encoder) Katmanları: Giriş gömme ve pozisyonel kodlama sonrasında, metin kodlayıcı katmanlarına geçilir. 

Bu katmanlar, giriş metnindeki bilgileri hiyerarşik olarak işler. Her bir kodlayıcı katmanı şu adımları içerir:

   *Çok Başlı Dikkat (Multi-Head Attention): Her kelimenin diğer kelimelere olan dikkatini hesaplar. Bu, her kelimenin çevresindeki bağlamı anlamasına yardımcı olur.

   *Katman Normalizasyon (Layer Normalization): Her başlıkta hesaplanan dikkat matrisinin çıktısını normalize eder.

   *İleri Besleme Ağları (Feed-Forward Networks): Her kelimenin temsilini daha karmaşık özelliklere dönüştürmek için kullanılır.

   *Toplamsal Bağlam (Residual Connection): Bu adım, her katmanın çıktısına girişe olan doğrudan bağlantıyı sağlar, bu da ağın daha iyi öğrenmesine yardımcı olur.

4- Toplam Çıkış:Kodlayıcı katmanlarından geçtikten sonra, elde edilen çıktılar bir dizi vektör temsilini içerir. Bu vektör temsilleri, giriş metni üzerinde yüksek düzeyde bilgi taşıyan bir temsilasyon oluşturur.

5-Çıkış Katmanları: Bu aşamada, modelin spesifik görevine bağlı olarak çıkış katmanları eklenir. Örneğin, çeviri görevinde hedef dilin kelimelerini oluşturmak için kullanılır.

6-Kayıp Fonksiyonu (Loss Function) ve Geri Yayılım (Backpropagation): Modelin tahminleri ile gerçek etiketler arasındaki kaybı ölçen bir kayıp fonksiyonu kullanılır. Ardından, geri yayılım algoritması kullanılarak bu kayıp geriye doğru iletilir ve modelin parametreleri, gradyan inişi kullanılarak güncellenir.


FEED FORWARD NETWORKS İleri besleme ağları (FFN), transformatör modelinin her kodlayıcı katmanında bulunan bir alt ağdır. Bu ağ, her kelimenin temsilini daha karmaşık özelliklere dönüştürmek ve daha genel bir temsil elde etmek için kullanılır. İleri besleme ağları, genellikle iki katmanlı tam bağlı (fully connected) yapılardan oluşur.

İleri besleme ağlarının adımları şu şekildedir:

1-İlk Tam Bağlı Katman (First Fully Connected Layer): Çok başlı dikkat katmanından gelen çıktılar, ilk tam bağlı katmana beslenir. Bu katmanın her bir çıkışı, belirli bir özellik veya konsepti temsil eder. İlk katman genellikle modelin giriş boyutundan (kelime gömme boyutu ve dikkat başlığı sayısı) daha büyük bir boyuta sahiptir.

2-Aktivasyon Fonksiyonu: İlk tam bağlı katmanın çıkışları genellikle bir aktivasyon fonksiyonundan geçer. Yaygın olarak kullanılan aktivasyon fonksiyonları arasında ReLU (Rectified Linear Unit) bulunur. Bu adım, ağın öğrenme kapasitesini artırır ve daha karmaşık özellikleri öğrenmesine yardımcı olur.

3-İkinci Tam Bağlı Katman (Second Fully Connected Layer): Aktivasyon fonksiyonundan geçen çıktılar, ikinci tam bağlı katmana beslenir. Bu katman genellikle ilk katmana göre daha küçük bir boyuta sahiptir ve çıkışlar, giriş metnindeki her kelimenin temsilini daha yoğun bir şekilde ifade eder.

4-Katman Normalizasyon (Layer Normalization): İleri besleme ağlarının çıkışları genellikle katman normalizasyonuna tabi tutulur. Bu adım, çıkışları normalize ederek eğitimi stabil tutar ve daha hızlı bir öğrenme süreci sağlar.

5- Toplamsal Bağlam (Residual Connection): İleri besleme ağlarının çıkışı, toplamsal bağlam (residual connection) ile birleştirilir. Bu, ağın daha iyi öğrenmesine ve daha etkili bir şekilde gradyanın geriye yayılmasına yardımcı olur.

Feed Forward Networks, her kelimenin temsilini zenginleştirerek daha karmaşık özellikleri öğrenmeye yardımcı olur. Bu adım, transformatör modelinin dil yapısını daha iyi anlamasına ve dil işleme görevlerinde yüksek performans elde etmesine katkıda bulunur.



"Çok Başlı Dikkat" (Multi-Head Attention) adımı, transformatör modelinin temel yapı taşlarından biridir. Bu mekanizma, belirli bir kelimenin temsilini hesaplamak için aynı anda birden fazla dikkat başlığını kullanır. Bu, modelin farklı özelliklere odaklanmasını sağlar ve dil içindeki uzak bağlantıları daha iyi modellemesine yardımcı olur.

Çok Başlı Dikkat adımının ayrıntıları şu şekildedir:

1- Giriş Hazırlığı: Çok Başlı Dikkat, her bir kelimenin giriş vektörünü alır. Bu vektörler, genellikle kelime gömme (word embedding) katmanı tarafından sağlanır ve kelimenin dil içindeki anlamını temsil eder.

2- Dikkat Başlıklarının Hazırlanması:Çok Başlı Dikkat, belirli bir konsepte odaklanmak üzere bir dizi dikkat başlığı (attention head) kullanır. Her başlık, öğrenilebilir ağırlıklardan oluşur ve farklı özelliklere odaklanma yeteneğini temsil eder.

3- Dikkat Pese ve Skalar Çarpım: Her dikkat başlığı, önce bir ağırlık matrisi (query matrisi) ile çarpılır. Bu, her kelimenin dikkat dağılımını öğrenmeye yardımcı olur. Daha sonra, bu dikkat dağılımı, diğer bir ağırlık matrisi (key matrisi) ile çarpılır. Son adım olarak, bu çarpımın skalar çarpımı alınır.

4-Dikkat Skorlarının Ağırlıklı Toplamı: Her bir dikkat başlığından elde edilen skorlar, dikkat ağırlıkları kullanılarak ağırlıklı bir şekilde toplanır. Bu adım, her kelimenin dikkat başlıkları arasındaki ilişkiyi öğrenmesini sağlar.

5-Dikkat Başlıklarının Birleştirilmesi: Tüm dikkat başlıklarından elde edilen toplam dikkat çıktısı, birleştirilir. Bu adım, modelin farklı özelliklere odaklanmasını sağlar ve genel bir temsil oluşturur.

6- Projece Edilme: Elde edilen birleştirilmiş dikkat çıktısı, öğrenilebilir bir projeksiyon matrisi ile çarpılarak daha yüksek bir boyuta yansıtılır.

7- Toplamsal Bağlam (Residual Connection) ve Normalizasyon: Bu adımda, toplamsal bağlam (residual connection) ve katman normalizasyonu uygulanarak çıkış stabilize edilir ve ağın öğrenmesine katkıda bulunur.

Çok Başlı Dikkat mekanizması, bir kelimenin temsilini hesaplamak için birden fazla bakış açısını birleştirerek modelin daha kapsamlı ve esnek bir dil anlayışı geliştirmesine yardımcı olur.



"Skalalanmış Nokta-Çarpım Dikkat" (Scaled Dot-Product Attention) mekanizması, transformatör modelinde dikkat (attention) hesaplamak için kullanılan temel bir bileşendir. Bu mekanizma, bir kelimenin diğer kelimelere olan önem derecesini belirlemek için nokta-çarpım işlemi kullanır ve ağırlıkların daha stabil olması için skalalanır. 

İşte bu mekanizmanın adımlarının detayları:

1- Sorgu, Anahtar ve Değer Vektörleri Oluşturma: Öncelikle, her bir kelimenin temsilini ifade eden üç tane vektör oluşturulur: sorgu (query), anahtar (key) ve değer (value) vektörleri. Bu vektörler, öğrenilebilir ağırlık matrisleriyle çarpılarak elde edilir.

2- Nokta-Çarpım Dikkat Skorlarının Hesaplanması: Sorgu, anahtar vektörleri arasındaki nokta-çarpım işlemi yapılır. Bu işlem, sorgunun bir kelimenin ne kadar önemli olduğunu belirlemesine yardımcı olur. Her bir çıkış, sorgu vektörü ile ilgili anahtar vektörleri arasındaki benzerliği temsil eder.

Özellikle, sorgu ve anahtar vektörleri arasındaki nokta-çarpım işlemi sonrasında, skorlar skalalanır. Bu skalalama işlemi, büyük sayıların softmax fonksiyonu içinde daha stabil bir şekilde işlenmesine yardımcı olur. Skalalama genellikle 1 / √d_k formülü ile yapılır, burada d_k kelimenin temsil boyutunu ifade eder.

3- Softmax ve Dikkat Ağırlıklarının Hesaplanması: Elde edilen skorlar, softmax fonksiyonu içinden geçirilir. Bu, her skoru 0 ile 1 arasında bir olasılık değerine dönüştürür. Softmax işlemi, dikkat ağırlıklarını elde etmek için kullanılır. Bu ağırlıklar, sorgunun hangi kelimelere odaklanacağını belirler.

4- Değer Vektörleri ile Ağırlıkların Ağırlı Ortalaması: Elde edilen dikkat ağırlıkları, değer vektörleri ile ağırlıklı bir şekilde çarpılır ve toplanır. Bu adım, sorgunun odaklandığı kelimelerin toplamsal temsilini oluşturur.

5- Çıkış Vektörü: Son olarak, elde edilen ağırlıklı toplam, önceki adımlardan gelen çıkışa eklenir. Bu adım, toplamsal bağlam (residual connection) adımını temsil eder.


Bu işlemler, her kelimenin diğer kelimelere olan dikkatini hesaplayan ve bu dikkat kullanılarak daha zengin bir temsil oluşturan Skalalanmış Nokta-Çarpım Dikkat mekanizmasını açıklar. Bu mekanizma, transformatör modelinin dil içindeki bağlantıları anlamasına ve farklı özelliklere odaklanmasına yardımcı olur.







