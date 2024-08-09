# Entity Bazlı Duygu Analizi
## 1.Giriş
### Takım Hakkında:

Bu proje, HİTİTDDİ1 tarafından geliştirilen Entity Bazlı Duygu Analizi adlı bir uygulamadır. Ekibimiz, müşteri geri bildirimlerini daha iyi anlamak için bu projeyi oluşturmuştur.

### Takım Üyeleri
- **Danışman**: Dr. Öğr. Üyesi Emre DENİZ
- **Makine Öğrenimi Mühendisi**: Harun Emre Kıran
- **Doğal Dil İşleme Uzmanı ve Takım Kaptanı**: Hikmet Sıla Ulukan
- **Veri Mühendisi**: Beyzanur Demir
- **Yazılım Geliştirici**: Muhammet Karadeniz
  
### Projenin Amacı:

Bu projenin amacı Türkçe metinlerdeki belirli entity'leri (varlıklar) tespit etmek ve bu entity'lerin duygu durumlarını analiz etmektir. Entity bazlı duygu analizi müşteri geri bildirimlerini daha iyi anlamak, hizmet ve ürün kalitesini artırmak ve genel müşteri memnuniyetini sağlamak için kullanılabilir.

## 2.Kurulum:
### Gerekli Python Paketlerinin Kurulması:
Projenin çalıştırılabilmesi için gerekli Python paketlerini kurmak için aşağıdaki komutu kullanın:
```bash
pip install -r requirements.txt
````
## 3.Modellern Eğitilmesi:
### NER Modeli
NER modelini eğitmek için ner_model/ dizinine gidin ve aşağıdaki komutu çalıştırın:
```bash
python ner_model_training.py
```
### Duygu Analizi Modeli
Duygu analizi modelini eğitmek için sentiment_model/ dizinine gidin ve aşağıdaki komutu çalıştırın:
```bash
python sentiment_model_training.py
```

## 4.Projenin Önemi

Bu proje, müşteri geri bildirimlerinden değerli içgörüler çıkarmak için NLP tekniklerini kullanır. Firmalar, bu analizler sayesinde ürün ve hizmetlerini iyileştirerek müşteri memnuniyetini artırabilirler. Ayrıca, sosyal medya ve çağrı merkezi uygulamalarında kullanılabilir.

## 5.Proje Sonuçları

Model performansı aşağıdaki gibidir:

                  precision    recall  f1-score   support

     0              0.62      0.55      0.59       166
     1              0.60      0.66      0.63       317
     2              0.59      0.58      0.58       304

    accuracy                            0.60       787
    macro avg       0.61      0.60      0.60       787
    weighted avg    0.60      0.60      0.60       787 
## 6.Görseller
<img src="https://github.com/user-attachments/assets/9fec1da7-1c79-4522-b26d-02140ad71054" alt="resim2" width="650"/>
<img src="https://github.com/user-attachments/assets/d63b41cd-32a8-4451-94a0-9c1addc8e63c" alt="resim3" width="650"/>
<img src="https://github.com/user-attachments/assets/44913dcc-65ae-4ca5-bccd-09cf05e7b00d" alt="resim4" width="650"/>
<img src="https://github.com/user-attachments/assets/2a530c26-305e-4ff9-9fd4-792aef03b631" alt="resim5" width="650"/>
<img src="https://github.com/user-attachments/assets/332857a6-1f50-4697-92b2-d00f7fa4625b" alt="resim6" width="650"/>




