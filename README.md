![](https://github.com/burakbaga/cifar100-custommodel/blob/master/classes.png)
Proje kapsamında cifar100 veri setinde bulunan 6 sınıf ile bir derin sinir ağı tasarlayıp bu ağı eğitmemiz isteniyor. Cifar verilerinden istenilen sınıfları alabilmek için cifar2png kütüphanesini kullandım. (https://github.com/knjcode/cifar2png)

# 1.Augmentation ve Dropout Kullanılmayan Model İle Eğitim

Elimizde bulunan verileri tasarladığımız bir Convolutional Neural Network (Evrişimsel Sinir Ağı) ile eğiteceğiz. Burada tasarladığımız modelde dropout katmanları bulunmayacak ve elimizde bulanan veri üzerinde augmentation işlemleri gerçekleştirilmeyecektir.

Bunu yaparken öncelikle custon_dnn adında bir python dosyası oluşturdum. Oluşturulan bu dosyaya bir kütüphanede diyebiliriz. Burada bulunan build_model isimli metot kullanılarak gerekli parametrelerde (add_dropout,num_of_class,input_shape) verilerek modelimizi build etmiş oluruz. 

6 CNN 2 Dense katman bulunmaktadır. CNN katmanlarında nöron sayıları her gizli katmanda arttırılmıştır. En son çıkış dense katmanında sınıf sayısı kadar nöron bulunmaktadır. Bunun sebebi kullandığımız loss (categorical_crossentropy) ile alakalıdır.  (6 çıkış her biri için oran veriyor diyebiliriz.) (sparse_categorical_crossentopy kullanmış olsaydık tek çıkış verirdik.)

Eğitim aşamasına geçildiğinde yine custom_dnn python dosyamızda bulunan başka bir metot olan data_fit metodunu çağırıyoruz. Bu metot da bulunan parametreler (data_augmentation, train_path, test_path, width,height,batch_size,epoch,model) bunlardır. 

Elimizde bulunan veriyi göndermek için daha önceden cifar2png kullanarak indirdiğimiz cifar datalarında seçtiğimiz 6 sınıf yolunu (train_path ve val_path) ve augmentation yapılacak mı bunu belirtiyoruz.

History den okunan bilgiler doğrultusunda grafik çizdirmek için plot_graph adında bir python dosyası oluşturulmuştur. Bu python dosyası içerinde bulanan plot isimli metot da historyden okunan (acc,val_accuracy,loss,val_loss) bilgiler gönderildiğinde grafiklerimiz çizilecek ve save edilecektir.

## 1.1 Grafiklerin Yorumlanması 

![](https://github.com/burakbaga/cifar100-custommodel/blob/master/grafikler/1.K%C4%B1s%C4%B1m%20(Basit%20Model)_accuracy.png)
Yukarı da ki grafikte görüldüğü üzere validation accuracy 6. Epochtan sonra neredeyse sabit kalmıştır. Ancak train accuracy yükselmeye devam etmiştir. 6 epochtan sonra öğrenme kesilmiştir. Model ezberlemeye başlamıştır. Bu ezberleme durumuna overfit deriz. Modelin elimizde bulanan veriye ve sınıf sayısına göre oldukça karmaşık olması bu sonucu almamızda etkili olmuştur. Eğitim bir süre daha devam ettirilmesi durumda model 1.0 başarı göstermeye devam edecek ve zaman içerisinde validation accuracy düşmeye başlayabilir

![](https://github.com/burakbaga/cifar100-custommodel/blob/master/grafikler/1.K%C4%B1s%C4%B1m%20(Basit%20Model)_loss.png)
Loss grafiğini incelediğimizde train datasıyla loss değerinin sürekli düştüğü gözlenmektedir. Hatta 20. Epochta loss değeri 0 olmuştur. Bu overfit olmayan bir modelde karşılaşacağımız bir sonuç değildir. Bunu yanında validation loss değeri düşme eğilimi göstersede bir noktadan sonra durağan olmayan sonuçlar vermiştir. Sonuncu epochta pik noktasına ulaşmıştır. Buda eğitimlerde beklediğimiz bir durum değildir.

# 2.Dropout Kullanılan Model İle Eğitim 

Burada bizden istenilen 1 adımda oluşturduğumuz modele dropout katmanları ekleyerek overfit problemini aşmayı denememizdir. Burada da yukarıda belirtilen kodlar kullanışmışır. Ancak burada custom_dnn içerisinde bulunan build_model metodunda bulnan add_dropout true olarak gönderilmiştir ve dropoutlu model import edilmiştir. 

## 2.1 Grafiklerin Yorumlanması
![](https://github.com/burakbaga/cifar100-custommodel/blob/master/grafikler/2.K%C4%B1s%C4%B1m%20(Dropout%20Kullan%C4%B1lan)_accuracy.png)

Yukarıda bulunan grafikte görüleceği üzere dropout kullanılan modelde train ve validation accuracy değerleri birbirlerine yakın sonuçlar almışlardır. 1. Modelde karşılaştıımız overfit probleminde ki gibi 1 başarı değil daha makul bir sonuç olan 0.9 başarı alınmıştır. Validation da overfit olmadığımızı doğrular şekilde 0.85 dolaylarında bir başarı elde etmiştir. Buradan çıkaracağımız sonuç overfit problemi ile başa çıkmada en sık başvurabilceğimiz yollaradan birinin dropout katmanı kullanmak olduğudur.


![](https://github.com/burakbaga/cifar100-custommodel/blob/master/grafikler/2.K%C4%B1s%C4%B1m%20(Dropout%20Kullan%C4%B1lan)_loss.png)

Dropout kullanılan modelin loss grafiğini incelediğimizde de train lossun epochlar boyunca düştüğü ve validation eğrisinin de bir düşüş gösterdiği görülmektedir. Arada sapmalar yaşanmıştır ama bu kısmen kabul edilebilecek bir sonuçtur. Eğitim yeterince başarılı görünüyor. Dropout oranları arttırarak daha iyi sonuçlar alınabilir.

# 3. Sadece Augmentation Kullanılan Model ile Eğitim

Sadece augmentation (veri zenginleştirme) kullanılan modelde dropout kullanılması istenmiyor add_dropout = False parametresi ile bunu sağlarız. data_fit metodumuzda parametre olarak data_augmentation = True şeklinde göndereceğiz.
Augmentation yapmak için ImageDataGenerator kullanarak ve gerekli parametreler verilerek bir data generator oluşturuyoruz. Augmentation işlemini sadece eğitim aşamasında kullacağımzı için bir de test için augmentation yapılmayan bir datagen oluşturuyoruz. Yine yukarılarda belirttiğimiz gibi grafikleri çizdirmek için history değişkenini döndürüyoruz

## 3.1 Grafiklerin Yorumlanması
![](https://github.com/burakbaga/cifar100-custommodel/blob/master/grafikler/3.K%C4%B1s%C4%B1m%20(Augmentation%20Kullan%C4%B1lan)_accuracy.png)
Augmentation kullanılan modelde 1. modele göre daha iyi sonuçlar aldığımız söylenebilir. Train ve validation accuracy birlikte artış gösteriyorlar. Ancak 16 epochtan sonra validation accuracy artmayı kesiyor hatta azalamaya başlıyor. Belki eğitimi 16. Epochta durdurarak model overfit olmadan kesebiliriz (Early Stopping). Ancak tam bir ezberleme durumunda söz etmek çok doğru olmayacaktır. Bunun yanında belki augmentation parametreleri ile oynayıp overfit yönelimi biraz daha düşürülebilir. 

![](https://github.com/burakbaga/cifar100-custommodel/blob/master/grafikler/3.K%C4%B1s%C4%B1m%20(Augmentation%20Kullan%C4%B1lan)_loss.png)
Augmentation kullanılan modelin loss grafiğini incelediğimizde train loss zaman içerisinde düştüğü görebiliriz. Validation loss değeri de zamanla azalmıştır. Ancak train loss eğrisi gibi çok stabil bir grafiği olduğu söylenemez. Bunun sebebi elimizde bulunan verinin azlığında kaynaklı olabileceğine düşünmekteyim. Yine 16 Epochtan sonra modelin durdurulması iyi bir tercih olacaktır. Bu durumun önüne geçmek için öğrenme katsayısının küçültülmesi denenebilir. Bu koşulda epoch sayısınında artırılması gerekir.

# 4.Genel Yorum

Tüm aşamalar grafikler incelendiğinde en başarılı modelin dropout kullanılan model olduğu söylenebilir. Ancak augmentation parametreleri ile oynayarak iyi sonuçlar alınabilir. En doğru hamle dropout kullanılan modelde augmentation veri kullanmak olacaktır. Overfit problemi ile karşılaştığımızda dropout ve augmentation etkili sonuçlar verdiği çıkarımını yapabiliriz.


Derin öğrenme problemlerinde kesin bir çözümden söz edemeyiz. Seçilen optimizasyon algoritması, katmanlar da ki nöron sayısı, katman sayısı, dropout oranı, öğrenme katsayısı, augmentation yapılırken kullanılan (yakınlaştırma, eksenlerde kaydırma, yatay-dikey döndürme) parametreler bunların her biri ile denemeler yapılır optimum sonuca ulaşılabilir. Grid Search kullanarak da denemeler yapılabilir. 


