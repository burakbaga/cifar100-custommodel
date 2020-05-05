
![](https://github.com/burakbaga/cifar100-custommodel/blob/master/classes.png)
Proje kapsamında cifar100 veri setinde bulunan 6 sınıf ile bir derin sinir ağı tasarlayıp bu ağı eğitmemiz isteniyor. Cifar verilerinden istenilen sınıfları alabilmek için cifar2png kütüphanesini kullandım. (https://github.com/knjcode/cifar2png)

1.Augmentation ve Dropout Kullanılmayan Model İle Eğitim

Elimizde bulunan verileri tasarladığımız bir Convolutional Neural Network (Evrişimsel Sinir Ağı) ile eğiteceğiz. Burada tasarladığımız modelde dropout katmanları bulunmayacak ve elimizde bulanan veri üzerinde augmentation işlemleri gerçekleştirilmeyecektir.

Bunu yaparken öncelikle custon_dnn adında bir python dosyası oluşturdum. Oluşturulan bu dosyaya bir kütüphanede diyebiliriz. Burada bulunan build_model isimli metot kullanılarak gerekli parametrelerde (add_dropout,num_of_class,input_shape) verilerek modelimizi build etmiş oluruz. 

6 CNN 2 Dense katman bulunmaktadır. CNN katmanlarında nöron sayıları her gizli katmanda arttırılmıştır. En son çıkış dense katmanında sınıf sayısı kadar nöron bulunmaktadır. Bunun sebebi kullandığımız loss (categorical_crossentropy) ile alakalıdır.  (6 çıkış her biri için oran veriyor diyebiliriz.) (sparse_categorical_crossentopy kullanmış olsaydık tek çıkış verirdik.)

Eğitim aşamasına geçildiğinde yine custom_dnn python dosyamızda bulunan başka bir metot olan data_fit metodunu çağırıyoruz. Bu metot da bulunan parametreler (data_augmentation, train_path, test_path, width,height,batch_size,epoch,model) bunlardır. 

Elimizde bulunan veriyi göndermek için daha önceden cifar2png kullanarak indirdiğimiz cifar datalarında seçtiğimiz 6 sınıf yolunu (train_path ve val_path) ve augmentation yapılacak mı bunu belirtiyoruz.

History den okunan bilgiler doğrultusunda grafik çizdirmek için plot_graph adında bir python dosyası oluşturulmuştur. Bu python dosyası içerinde bulanan plot isimli metot da historyden okunan (acc,val_accuracy,loss,val_loss) bilgiler gönderildiğinde grafiklerimiz çizilecek ve save edilecektir.

1.1 Grafiklerin Yorumlanması 

![Şekil 8 1.Model Train ve Validation Accuracy](https://github.com/burakbaga/cifar100-custommodel/blob/master/grafikler/1.K%C4%B1s%C4%B1m%20(Basit%20Model)_accuracy.png)

Yukarı da ki grafikte görüldüğü üzere validation accuracy 6. Epochtan sonra neredeyse sabit kalmıştır. Ancak train accuracy yükselmeye devam etmiştir. 6 epochtan sonra öğrenme kesilmiştir. Model ezberlemeye başlamıştır. Bu ezberleme durumuna overfit deriz. Modelin elimizde bulanan veriye ve sınıf sayısına göre oldukça karmaşık olması bu sonucu almamızda etkili olmuştur. Eğitim bir süre daha devam ettirilmesi durumda model 1.0 başarı göstermeye devam edecek ve zaman içerisinde validation accuracy düşmeye başlayabilir

















