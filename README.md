Bu proje Create a Large Language Model from Scratch with Python – Tutorial (https://youtu.be/UU1WVnMk4E8?si=Fgj6Tle9XatWg2fq) ile çalışarak hazırlanmış içeriği türkçe açıklamalarla anlatılmıştır.

Proje ilk olarak PyTorch anlatımı ile başlamaktadır.

Daha sonra Bigram Tutorial'ı içermektedir.

# Transformer Mimari Yapısı 
![Transformer - model architecture](https://github.com/gizemkoklu/LLM-project/assets/93999489/ce0c6caa-4c46-4fc7-b576-7d62c0e9d9ef)
(Attention Is All You Need - https://arxiv.org/pdf/1706.03762.pdf)

1- `Giriş Gömme (Input Embedding)`: İlk adım, metni kelime seviyesinde temsil etmektir. Her kelimenin gömme vektörü, genellikle öğrenilebilir parametrelerle başlar. Bu vektörler, kelimenin dil içindeki anlamını temsil eder.

2- `Pozisyonel Kodlama (Positional Encoding)`: Giriş metni, sırasal bir yapıya sahip olduğu için transformatör modeli, kelime sırasının model tarafından öğrenilememesi nedeniyle pozisyonel kodlama ekler. Pozisyonel kodlama, her kelimeye bir pozisyon bilgisi ekleyerek sıralı yapının model tarafından dikkate alınmasını sağlar.

3-`Kodlayıcı (Encoder) Katmanları`: Giriş gömme ve pozisyonel kodlama sonrasında, metin kodlayıcı katmanlarına geçilir. 

Bu katmanlar, giriş metnindeki bilgileri hiyerarşik olarak işler. Her bir kodlayıcı katmanı şu adımları içerir:

   *Çok Başlı Dikkat (Multi-Head Attention): Her kelimenin diğer kelimelere olan dikkatini hesaplar. Bu, her kelimenin çevresindeki bağlamı anlamasına yardımcı olur.

   *Katman Normalizasyon (Layer Normalization): Her başlıkta hesaplanan dikkat matrisinin çıktısını normalize eder.

   *İleri Besleme Ağları (Feed-Forward Networks): Her kelimenin temsilini daha karmaşık özelliklere dönüştürmek için kullanılır.

   *Toplamsal Bağlam (Residual Connection): Bu adım, her katmanın çıktısına girişe olan doğrudan bağlantıyı sağlar, bu da ağın daha iyi öğrenmesine yardımcı olur.

4- `Toplam Çıkış`:Kodlayıcı katmanlarından geçtikten sonra, elde edilen çıktılar bir dizi vektör temsilini içerir. Bu vektör temsilleri, giriş metni üzerinde yüksek düzeyde bilgi taşıyan bir temsilasyon oluşturur.

5-`Çıkış Katmanları`: Bu aşamada, modelin spesifik görevine bağlı olarak çıkış katmanları eklenir. Örneğin, çeviri görevinde hedef dilin kelimelerini oluşturmak için kullanılır.

6-`Kayıp Fonksiyonu (Loss Function) ve Geri Yayılım (Backpropagation)`: Modelin tahminleri ile gerçek etiketler arasındaki kaybı ölçen bir kayıp fonksiyonu kullanılır. Ardından, geri yayılım algoritması kullanılarak bu kayıp geriye doğru iletilir ve modelin parametreleri, gradyan inişi kullanılarak güncellenir.


## FEED FORWARD NETWORKS 
#### İleri besleme ağları (FFN), transformatör modelinin her kodlayıcı katmanında bulunan bir alt ağdır. Bu ağ, her kelimenin temsilini daha karmaşık özelliklere dönüştürmek ve daha genel bir temsil elde etmek için kullanılır. İleri besleme ağları, genellikle iki katmanlı tam bağlı (fully connected) yapılardan oluşur.

İleri besleme ağlarının adımları şu şekildedir:

1-`İlk Tam Bağlı Katman (First Fully Connected Layer)`: Çok başlı dikkat katmanından gelen çıktılar, ilk tam bağlı katmana beslenir. Bu katmanın her bir çıkışı, belirli bir özellik veya konsepti temsil eder. İlk katman genellikle modelin giriş boyutundan (kelime gömme boyutu ve dikkat başlığı sayısı) daha büyük bir boyuta sahiptir.

2-`Aktivasyon Fonksiyonu`: İlk tam bağlı katmanın çıkışları genellikle bir aktivasyon fonksiyonundan geçer. Yaygın olarak kullanılan aktivasyon fonksiyonları arasında ReLU (Rectified Linear Unit) bulunur. Bu adım, ağın öğrenme kapasitesini artırır ve daha karmaşık özellikleri öğrenmesine yardımcı olur.

3-`İkinci Tam Bağlı Katman (Second Fully Connected Layer)`: Aktivasyon fonksiyonundan geçen çıktılar, ikinci tam bağlı katmana beslenir. Bu katman genellikle ilk katmana göre daha küçük bir boyuta sahiptir ve çıkışlar, giriş metnindeki her kelimenin temsilini daha yoğun bir şekilde ifade eder.

4-`Katman Normalizasyon (Layer Normalization)`: İleri besleme ağlarının çıkışları genellikle katman normalizasyonuna tabi tutulur. Bu adım, çıkışları normalize ederek eğitimi stabil tutar ve daha hızlı bir öğrenme süreci sağlar.

5- `Toplamsal Bağlam (Residual Connection)`: İleri besleme ağlarının çıkışı, toplamsal bağlam (residual connection) ile birleştirilir. Bu, ağın daha iyi öğrenmesine ve daha etkili bir şekilde gradyanın geriye yayılmasına yardımcı olur.

Feed Forward Networks, her kelimenin temsilini zenginleştirerek daha karmaşık özellikleri öğrenmeye yardımcı olur. Bu adım, transformatör modelinin dil yapısını daha iyi anlamasına ve dil işleme görevlerinde yüksek performans elde etmesine katkıda bulunur.



## Multi-Head Attention
#### "Çok Başlı Dikkat" adımı, transformatör modelinin temel yapı taşlarından biridir. Bu mekanizma, belirli bir kelimenin temsilini hesaplamak için aynı anda birden fazla dikkat başlığını kullanır. Bu, modelin farklı özelliklere odaklanmasını sağlar ve dil içindeki uzak bağlantıları daha iyi modellemesine yardımcı olur.
![multi-head attention](https://github.com/gizemkoklu/LLM-project/assets/93999489/5b1de6b3-1e0a-4e91-a80c-06e50b2e2ff6)


Çok Başlı Dikkat adımının ayrıntıları şu şekildedir:

1- `Giriş Hazırlığı`: Çok Başlı Dikkat, her bir kelimenin giriş vektörünü alır. Bu vektörler, genellikle kelime gömme (word embedding) katmanı tarafından sağlanır ve kelimenin dil içindeki anlamını temsil eder.

2-` Dikkat Başlıklarının Hazırlanması`:Çok Başlı Dikkat, belirli bir konsepte odaklanmak üzere bir dizi dikkat başlığı (attention head) kullanır. Her başlık, öğrenilebilir ağırlıklardan oluşur ve farklı özelliklere odaklanma yeteneğini temsil eder.

3- `Dikkat Pese ve Skalar Çarpım`: Her dikkat başlığı, önce bir ağırlık matrisi (query matrisi) ile çarpılır. Bu, her kelimenin dikkat dağılımını öğrenmeye yardımcı olur. Daha sonra, bu dikkat dağılımı, diğer bir ağırlık matrisi (key matrisi) ile çarpılır. Son adım olarak, bu çarpımın skalar çarpımı alınır.

4-`Dikkat Skorlarının Ağırlıklı Toplamı`: Her bir dikkat başlığından elde edilen skorlar, dikkat ağırlıkları kullanılarak ağırlıklı bir şekilde toplanır. Bu adım, her kelimenin dikkat başlıkları arasındaki ilişkiyi öğrenmesini sağlar.

5-`Dikkat Başlıklarının Birleştirilmesi`: Tüm dikkat başlıklarından elde edilen toplam dikkat çıktısı, birleştirilir. Bu adım, modelin farklı özelliklere odaklanmasını sağlar ve genel bir temsil oluşturur.

6- `Projece Edilme`: Elde edilen birleştirilmiş dikkat çıktısı, öğrenilebilir bir projeksiyon matrisi ile çarpılarak daha yüksek bir boyuta yansıtılır.

7-`Toplamsal Bağlam (Residual Connection) ve Normalizasyon`: Bu adımda, toplamsal bağlam (residual connection) ve katman normalizasyonu uygulanarak çıkış stabilize edilir ve ağın öğrenmesine katkıda bulunur.

Çok Başlı Dikkat mekanizması, bir kelimenin temsilini hesaplamak için birden fazla bakış açısını birleştirerek modelin daha kapsamlı ve esnek bir dil anlayışı geliştirmesine yardımcı olur.



## Scaled Dot-Product Attention
#### Skalalanmış Nokta-Çarpım Dikkat mekanizması, transformatör modelinde dikkat (attention) hesaplamak için kullanılan temel bir bileşendir. Bu mekanizma, bir kelimenin diğer kelimelere olan önem derecesini belirlemek için nokta-çarpım işlemi kullanır ve ağırlıkların daha stabil olması için skalalanır. 

![scaled dot-product attention](https://github.com/gizemkoklu/LLM-project/assets/93999489/022c462c-eaa5-4492-aa42-29694cb6e58f)


İşte bu mekanizmanın adımlarının detayları:

1- `Sorgu, Anahtar ve Değer Vektörleri Oluşturma`: Öncelikle, her bir kelimenin temsilini ifade eden üç tane vektör oluşturulur: sorgu (query), anahtar (key) ve değer (value) vektörleri. Bu vektörler, öğrenilebilir ağırlık matrisleriyle çarpılarak elde edilir.

2- `Nokta-Çarpım Dikkat Skorlarının Hesaplanması`: Sorgu, anahtar vektörleri arasındaki nokta-çarpım işlemi yapılır. Bu işlem, sorgunun bir kelimenin ne kadar önemli olduğunu belirlemesine yardımcı olur. Her bir çıkış, sorgu vektörü ile ilgili anahtar vektörleri arasındaki benzerliği temsil eder.

Özellikle, sorgu ve anahtar vektörleri arasındaki nokta-çarpım işlemi sonrasında, skorlar skalalanır. Bu skalalama işlemi, büyük sayıların softmax fonksiyonu içinde daha stabil bir şekilde işlenmesine yardımcı olur. Skalalama genellikle 1 / √d_k formülü ile yapılır, burada d_k kelimenin temsil boyutunu ifade eder.

3- `Softmax ve Dikkat Ağırlıklarının Hesaplanması`: Elde edilen skorlar, softmax fonksiyonu içinden geçirilir. Bu, her skoru 0 ile 1 arasında bir olasılık değerine dönüştürür. Softmax işlemi, dikkat ağırlıklarını elde etmek için kullanılır. Bu ağırlıklar, sorgunun hangi kelimelere odaklanacağını belirler.

4- `Değer Vektörleri ile Ağırlıkların Ağırlı Ortalaması`: Elde edilen dikkat ağırlıkları, değer vektörleri ile ağırlıklı bir şekilde çarpılır ve toplanır. Bu adım, sorgunun odaklandığı kelimelerin toplamsal temsilini oluşturur.

5- `Çıkış Vektörü`: Son olarak, elde edilen ağırlıklı toplam, önceki adımlardan gelen çıkışa eklenir. Bu adım, toplamsal bağlam (residual connection) adımını temsil eder.


Bu işlemler, her kelimenin diğer kelimelere olan dikkatini hesaplayan ve bu dikkat kullanılarak daha zengin bir temsil oluşturan Skalalanmış Nokta-Çarpım Dikkat mekanizmasını açıklar. Bu mekanizma, transformatör modelinin dil içindeki bağlantıları anlamasına ve farklı özelliklere odaklanmasına yardımcı olur.




# GPT MİMARİSİ


![transformer vs gpt](https://github.com/gizemkoklu/LLM-project/assets/93999489/67c3ecf9-516d-4dd9-9ec3-eda55167967f)


GPT (Generative Pre-trained Transformer), OpenAI tarafından geliştirilen bir yapay zeka dil modelidir. GPT mimarisinin en ince ayrıntısına inmek için, "GPT-3" modelini örnek alalım.

* `Transformer Mimarisi`: GPT, "Transformer" adlı bir dil modeli mimarisini kullanır. Bu mimari, özellikle dil işleme görevlerinde yüksek performans göstermesiyle bilinir. Dikkat mekanizması, öğrenme sürecinde belirli kısımlara odaklanma yeteneği ile dikkat çeker.

* `Çoklu Kafa Dikkat Mekanizması (Multi-Head Attention)`: GPT, dikkat mekanizması kullanır. Bu, her bir giriş kelimesinin çıktıda hangi kısımlara daha fazla vurgu yapılması gerektiğini belirlemek için kullanılır.
Multi-Head Attention, dikkat mekanizmasını birden çok alt-mekanizma ile genişletir.

* `Ön-eğitim (Pre-training)`: GPT, büyük miktarlarda metin verisi üzerinde "ön-eğitim" adı verilen bir süreçten geçer. Bu süreçte model, genel dil anlamını ve bağlamını öğrenir.
Ön-eğitim aşamasında, modelin birçok dil görevini anlayabilmesi için çeşitli dil görevleri üzerinde eğitildiği birçok veri kümesi kullanılır.

* `Dönüşlü (Autoregressive) Model`: GPT, dönüşlü bir dil modelidir. Yani, bir cümleyi oluştururken önceki kelimelerin bilgisini kullanır. Bu, dildeki bağlamsal ilişkileri daha iyi anlamasına yardımcı olur.

* `Katmanlar (Layers)`: GPT modelleri, birbirine bağlı birçok katmandan oluşur. GPT-3 gibi büyük modeller, genellikle yüzlerce milyon parametreye sahip çok sayıda katman içerir.

* `Özelleştirilebilir Boyutlar (Configurable Dimensions)`: GPT'nin mimarisi, giriş boyutu, model boyutu ve diğer özellikleri özelleştirmeye olanak tanır. Bu, farklı büyüklükteki modellerin oluşturulabilmesine ve çeşitli görevlerde kullanılabilmesine olanak tanır.

* `Transfer Öğrenme (Transfer Learning)`: GPT'nin temel gücü, ön-eğitim sırasında genel dil anlayışını öğrenmesi ve daha sonra bu öğrenilen bilgiyi çeşitli görevlerde transfer edebilmesidir.

* `Kendi Düzenlemeli (Self-Attention) Mekanizması`: Transformer mimarisi, kendi düzenlemeli mekanizma içerir. Bu mekanizma, bir kelimenin anlamını belirlerken diğer kelimelerin katkısını dikkate alır. Bu, uzun mesafeli bağlantıları modellemenin daha etkili olmasına olanak tanır.

* `Kümeleme (Layer Normalization) ve Hesaplama Verimliliği`: GPT'nin her katmanında kümeleme (layer normalization) kullanılır. Bu, eğitim sürecini stabilize etmeye ve daha hızlı öğrenmeye yardımcı olabilir. Ayrıca, paralel hesaplama ve eğitim sürecinin daha hızlı gerçekleştirilmesi için özel bir dikkat mekanizması vardır.

* `Ön-eğitim (Pre-training) Aşamaları`: GPT, geniş bir dil veri kümesi üzerinde ön-eğitimden geçer. Bu, modelin dilin genel yapısını, kelime ilişkilerini ve dünya genelindeki dil kalıplarını öğrenmesine olanak tanır.

* `Geliştirilmiş Özyinelemeli (Recursive) Yapı`: GPT, geliştirilmiş bir özyinelemeli yapı kullanır. Bu, modelin belirli görevlere uygun özel bilgileri öğrenmesini sağlar.

* `Adaptif Öğrenme Oranı (Adaptive Learning Rate)`: Eğitim sırasında, öğrenme oranı adaptif bir şekilde ayarlanabilir. Bu, eğitim başlangıcında hızlı öğrenme ve daha sonra stabilizasyon için yavaşlama sağlar.

* `Dil Görevlerinde Çok Yönlü Kullanım`: GPT, metin tabanlı çeşitli görevlerde kullanılabilir. Ön-eğitim sırasında genel dil anlamını öğrenen model, daha sonra belirli görevlere uyarlanabilir.

* `GPT-3 ve Büyük Boyutlar`: GPT-3 gibi büyük modeller, yüz milyonlarca veya milyarlarca parametreye sahiptir. Bu büyük boyutlar, genel dil anlayışının daha karmaşık ve zengin olmasına olanak tanır.



### GPT mimarisinin bazı özneleri ve tanımları:

* `Tokenization (Belirteçleme)`: GPT, giriş metni parçalara ayırmak için bir belirteçleme işlemi kullanır.

* `Embedding (Gömme)`: Belirteçler, kelime gömme katmanına geçirilir, bu katman kelimeleri öğrenilebilir vektörlerle temsil eder.

* `Transformer Encoder`: GPT, bir dizi Transformer Encoder katmanını içerir. Bu katmanlar, giriş metninin bağlamını anlamak için dikkat mekanizmasını kullanır.

* `Attention Mechanism (Dikkat Mekanizması)`: Dikkat mekanizması, her bir belirtecin diğer belirtecilere olan önemini belirler. Bu, uzun mesafeli bağlantıları modellemeye olanak tanır.

* `Layer Normalization`:  Her Transformer katmanının çıkışı, normalizasyon işleminden geçirilir. Bu, eğitim sürecini stabilize eder.

* `Positional Encoding (Pozisyonel Kodlama)`: GPT, belirteçlerin sırasını modellemek için pozisyonel kodlamayı kullanır. Bu, belirteçlerin metindeki konumunu temsil eder.

* `Layer-wise Feedforward Networks`: Her Transformer katmanının içinde, bir dizi layer-wise beslemeli (feedforward) ağ bulunur.

* `Multi-Head Attention`: Dikkat mekanizması, birden çok kafa (head) ile genişletilir, her bir kafa farklı özelliklere odaklanabilir.

* `Position-wise Feedforward Networks`: GPT'nin katmanlarında, her belirteç için ayrı ayrı uygulanan bir pozisyon bazlı beslemeli ağ vardır.

* `Layer Output`: Her bir Transformer katmanının çıkışı, bir sonraki katmana veya çıkışa giden giriş olabilir.

* `Layer Stacking`: GPT, birbiri üzerine istiflenmiş birden çok Transformer katmanını içerir.

* `Pre-training`: GPT, geniş bir dil veri kümesi üzerinde ön-eğitim aşamasından geçer. Bu, genel dil anlayışını kazanmasını sağlar.

* `Fine-tuning`: Ön-eğitimden sonra, GPT özel görevlere adapte edilebilir. Bu, fine-tuning aşamasını içerir.
