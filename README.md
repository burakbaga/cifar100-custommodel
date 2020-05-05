
![](RackMultipart20200505-4-1jmeo6k_html_9d4fcb5f96283c84.png)

_Şekil 1 Sınıflar_

Proje kapsamında cifar100 veri setinde bulunan 6 sınıf ile bir derin sinir ağı tasarlayıp bu ağı eğitmemiz isteniyor. Cifar verilerinden istenilen sınıfları alabilmek için cifar2png kütüphanesini kullandım. ([https://github.com/knjcode/cifar2png](https://github.com/knjcode/cifar2png))

![](RackMultipart20200505-4-1jmeo6k_html_2b7bfcb20fa9ed29.png)

_Şekil 2 Kullanılan Bazı Parametreler_

Projede **3 adım** bulunmaktadır.

**1.Augmentation ve Dropout Kullanılmayan Model İle Eğitim**

Elimizde bulunan verileri tasarladığımız bir **Convolutional Neural Network** (Evrişimsel Sinir Ağı) ile eğiteceğiz. Burada tasarladığımız modelde dropout katmanları bulunmayacak ve elimizde bulanan veri üzerinde augmentation işlemleri gerçekleştirilmeyecektir.

B ![](RackMultipart20200505-4-1jmeo6k_html_aca5a3f713a50761.png) unu yaparken öncelikle custon\_dnn adında bir python dosyası oluşturdum. Oluşturulan bu dosyaya bir kütüphanede diyebiliriz. Burada bulunan build\_model isimli metot kullanılarak gerekli parametrelerde (add\_dropout,num\_of\_class,input\_shape) verilerek modelimizi build etmiş oluruz.

_Şekil 3 Dropout kullanılmayan model_

Yukarıda belirtilen modelde 6 CNN 2 Dense katman bulunmaktadır. CNN katmanlarında nöron sayıları her gizli katmanda arttırılmıştır. En son çıkış dense katmanında sınıf sayısı kadar nöron bulunmaktadır. Bunun sebebi kullandığımız loss (categorical\_crossentropy) ile alakalıdır. (6 çıkış her biri için oran veriyor diyebiliriz.) (sparse\_categorical\_crossentopy kullanmış olsaydık tek çıkış verirdik.)

Eğitim aşamasına geçildiğinde yine custom\_dnn python dosyamızda bulunan başka bir metot olan data\_fit metodunu çağırıyoruz. Bu metot da bulunan parametreler (data\_augmentation, train\_path, test\_path, width,height,batch\_size,epoch,model) bunlardır.

Elimizde bulunan veriyi göndermek için daha önceden cifar2png kullanarak indirdiğimiz cifar datalarında seçtiğimiz 6 sınıf yolunu (train\_path ve val\_path) ve augmentation yapılacak mı bunu belirtiyoruz.

![](RackMultipart20200505-4-1jmeo6k_html_c86ddfaba0bc42f6.png)

_Şekil 4 Eğitimin yapılması_

![](RackMultipart20200505-4-1jmeo6k_html_cc92e8e2f2687738.png)

_Şekil 5 Eğitimin başlatılması_

Şekil 4&#39;te gösterilen kod parçasında augmentation yapılmayacak şekilde datagen ayarlanmıştır jupyter tarafında gönderilen train\_path ve validation\_path te bulanan veriler okunmuş ve scale edilmiştir. Model compile edilmiş ve fit\_generator komutu ile eğitimi başlatılır. Eğitim sonucunda bilmek istediğimiz accuracy ve loss bilgileri history değişkenine yazılır. Bu değişken işlemler bittikten sonra return edilir.

![](RackMultipart20200505-4-1jmeo6k_html_6c561463e28abd40.png)

_Şekil 6 Historye yazılan bilgilerin okunması_

![](RackMultipart20200505-4-1jmeo6k_html_7b3752fa6ea411bd.png)

_Şekil 7 Grafik Çizdirme_

History den okunan bilgiler doğrultusunda grafik çizdirmek için plot\_graph adında bir python dosyası oluşturulmuştur. Bu python dosyası içerinde bulanan plot isimli metot da historyden okunan (acc,val\_accuracy,loss,val\_loss) bilgiler gönderildiğinde grafiklerimiz çizilecek ve save edilecektir.

**1.1 Grafiklerin Yorumlanması**

![](RackMultipart20200505-4-1jmeo6k_html_da7c8e99f7bf57a9.png)

_Şekil 8 1.Model Train ve Validation Accuracy_

Yukarı da ki grafikte görüldüğü üzere validation accuracy 6. Epochtan sonra neredeyse sabit kalmıştır. Ancak train accuracy yükselmeye devam etmiştir. 6 epochtan sonra öğrenme kesilmiştir. Model ezberlemeye başlamıştır. Bu ezberleme durumuna overfit deriz. Modelin elimizde bulanan veriye ve sınıf sayısına göre oldukça karmaşık olması bu sonucu almamızda etkili olmuştur. Eğitim bir süre daha devam ettirilmesi durumda model 1.0 başarı göstermeye devam edecek ve zaman içerisinde validation accuracy düşmeye başlayabilir

. ![](RackMultipart20200505-4-1jmeo6k_html_ebb172f291e1ac03.png)

_Şekil 9 1. Model Train ve Validation Loss_

Loss grafiğini incelediğimizde train datasıyla loss değerinin sürekli düştüğü gözlenmektedir. Hatta 20. Epochta loss değeri 0 olmuştur. Bu overfit olmayan bir modelde karşılaşacağımız bir sonuç değildir. Bunu yanında validation loss değeri düşme eğilimi göstersede bir noktadan sonra durağan olmayan sonuçlar vermiştir. Sonuncu epochta pik noktasına ulaşmıştır. Buda eğitimlerde beklediğimiz bir durum değildir.

**2.Dropout Kullanılan Model İle Eğitim**

![](RackMultipart20200505-4-1jmeo6k_html_a24d849fe1156d1a.png)Burada bizden istenilen 1 adımda oluşturduğumuz modele dropout katmanları ekleyerek overfit problemini aşmayı denememizdir. Burada da yukarıda belirtilen kodlar kullanışmışır. Ancak burada custom\_dnn içerisinde bulunan build\_model metodunda bulnan add\_dropout true olarak gönderilmiştir ve dropoutlu model import edilmiştir.

![](RackMultipart20200505-4-1jmeo6k_html_a7894f933912e907.gif)

_Şekil 10 Dropout modelin import edilmesi_

Bu modelde de yukarıda belirtilen şekilde data\_fit çalıştırılmıtır. Gerekli parametreler gönderilmiş ve eğitim başlatılmıştır. Yine eğitim bilgileri history nesnesinde tutulmaktadır.

**2.1 Grafiklerin Yorumlanması**

![](RackMultipart20200505-4-1jmeo6k_html_acf000ba99d75b57.png)

_Şekil 11 Dropout Kullanılan Model Accuracy_

Yukarıda bulunan grafikte görüleceği üzere dropout kullanılan modelde train ve validation accuracy değerleri birbirlerine yakın sonuçlar almışlardır. 1. Modelde karşılaştıımız overfit probleminde ki gibi 1 başarı değil daha makul bir sonuç olan 0.9 başarı alınmıştır. Validation da overfit olmadığımızı doğrular şekilde 0.85 dolaylarında bir başarı elde etmiştir. Buradan çıkaracağımız sonuç overfit problemi ile başa çıkmada en sık başvurabilceğimiz yollaradan birinin dropout katmanı kullanmak olduğudur.

![](RackMultipart20200505-4-1jmeo6k_html_e9295484d17b3392.png)

_Şekil 12 Dropout kullanılan model Loss_

Dropout kullanılan modelin loss grafiğini incelediğimizde de train lossun epochlar boyunca düştüğü ve validation eğrisinin de bir düşüş gösterdiği görülmektedir. Arada sapmalar yaşanmıştır ama bu kısmen kabul edilebilecek bir sonuçtur. Eğitim yeterince başarılı görünüyor. Dropout oranları arttırarak daha iyi sonuçlar alınabilir.

**3. Sadece Augmentation Kullanılan Model ile Eğitim**

Sadece augmentation (veri zenginleştirme) kullanılan modelde dropout kullanılması istenmiyor add\_dropout = False parametresi ile bunu sağlarız. data\_fit metodumuzda parametre olarak data\_augmentation = True şeklinde göndereceğiz.

![](RackMultipart20200505-4-1jmeo6k_html_64f9f581b618bde.png)

_Şekil 13 Augmentation yapılması ve fit edilme_

Augmentation yapmak için ImageDataGenerator kullanarak ve gerekli parametreler verilerek bir data generator oluşturuyoruz. Augmentation işlemini sadece eğitim aşamasında kullacağımzı için bir de test için augmentation yapılmayan bir datagen oluşturuyoruz. Yine yukarılarda belirttiğimiz gibi grafikleri çizdirmek için history değişkenini döndürüyoruz

**3.1 Grafiklerin Yorumlanması**

![](RackMultipart20200505-4-1jmeo6k_html_58c2ef32c5ce199e.png)

_Şekil 14 Augmentation kullanılan model Accuracy_

Augmentation kullanılan modelde 1. modele göre daha iyi sonuçlar aldığımız söylenebilir. Train ve validation accuracy birlikte artış gösteriyorlar. Ancak 16 epochtan sonra validation accuracy artmayı kesiyor hatta azalamaya başlıyor. Belki eğitimi 16. Epochta durdurarak model overfit olmadan kesebiliriz (Early Stopping). Ancak tam bir ezberleme durumunda söz etmek çok doğru olmayacaktır. Bunun yanında belki augmentation parametreleri ile oynayıp overfit yönelimi biraz daha düşürülebilir.

![](RackMultipart20200505-4-1jmeo6k_html_8deaf8c11a9dd5d0.png)

_Şekil 15 Augmentation kullanılan model loss_

Augmentation kullanılan modelin loss grafiğini incelediğimizde train loss zaman içerisinde düştüğü görebiliriz. Validation loss değeri de zamanla azalmıştır. Ancak train loss eğrisi gibi çok stabil bir grafiği olduğu söylenemez. Bunun sebebi elimizde bulunan verinin azlığında kaynaklı olabileceğine düşünmekteyim. Yine 16 Epochtan sonra modelin durdurulması iyi bir tercih olacaktır. Bu durumun önüne geçmek için öğrenme katsayısının küçültülmesi denenebilir. Bu koşulda epoch sayısınında artırılması gerekir.

**4.Genel Yorum**

Tüm aşamalar grafikler incelendiğinde en başarılı modelin dropout kullanılan model olduğu söylenebilir. Ancak augmentation parametreleri ile oynayarak iyi sonuçlar alınabilir. En doğru hamle dropout kullanılan modelde augmentation veri kullanmak olacaktır. Overfit problemi ile karşılaştığımızda dropout ve augmentation etkili sonuçlar verdiği çıkarımını yapabiliriz.

Derin öğrenme problemlerinde kesin bir çözümden söz edemeyiz. Seçilen optimizasyon algoritması, katmanlar da ki nöron sayısı, katman sayısı, dropout oranı, öğrenme katsayısı, augmentation yapılırken kullanılan (yakınlaştırma, eksenlerde kaydırma, yatay-dikey döndürme) parametreler bunların her biri ile denemeler yapılır optimum sonuca ulaşılabilir. Grid Search kullanarak da denemeler yapılabilir.
