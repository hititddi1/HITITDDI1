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
                 precision    recall    f1-score    support

    negative        0.88       1.00      0.93         14
     neutral        0.67       1.00      0.80          4
    positive        1.00       0.83      0.90         23

    accuracy                             0.90         41
    macro avg        0.85       0.94     0.88         41
    weighted avg     0.92       0.90     0.90         41
