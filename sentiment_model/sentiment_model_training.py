# %%
import json
import re
import pandas as pd
import torch
import spacy
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

# Türkçe dil modeli yüklemesi (spacy)
nlp = spacy.blank("tr")
nlp.add_pipe('sentencizer')

# GPU kullanımı için kontrol
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# %%
# Veriyi yükleme
data_path = "filtered_vaka3.json"
with open(data_path, 'r', encoding='utf-8') as file:
    data = json.load(file)

# Metin temizleme fonksiyonu
def preprocess_text(text, entity_list):
    words = text.split()
    processed_words = []
    for word in words:
        if any(entity.lower() == word.lower() for entity in entity_list):
            processed_words.append(word)  # entity'i olduğu gibi bırak
        else:
            processed_words.append(word.lower())  # diğer kelimeleri küçük harfe çevir
    return " ".join(processed_words)

# Veri çerçevesine dönüştürme
rows = []
for item in data:
    comment = item['comment']
    entity_list = item['entity_list']
    processed_comment = preprocess_text(comment, entity_list)
    for entity in entity_list:
        sentiment = next((result['sentiment'] for result in item['results'] if result['entity'] == entity), None)
        if sentiment is not None:
            rows.append({
                'comment': comment,
                'entity': entity,
                'cleaned_comment': processed_comment,
                'sentiment': sentiment
            })

df = pd.DataFrame(rows)
print(df.head())

# Label encoding
label_encoder = LabelEncoder()
df['label'] = label_encoder.fit_transform(df['sentiment'])
print(df[['comment', 'entity', 'cleaned_comment', 'sentiment', 'label']].head())


# %%
df

# %%
# Her entity için cümleleri ayırma
def split_sentences(comment, entity_list):
    doc = nlp(comment)
    sentences = [sent.text.strip() for sent in doc.sents]
    entity_sentences = {entity: [] for entity in entity_list}
    for sentence in sentences:
        for entity in entity_list:
            if entity.lower() in sentence.lower():
                entity_sentences[entity].append(sentence)
    return entity_sentences

df['entity_sentences'] = df.apply(lambda x: split_sentences(x['cleaned_comment'], [x['entity']]), axis=1)
print(df.head())


# %%
with open("entity_sentences.txt", "w") as f:
  for sentence in df["entity_sentences"]:
      # Eğer eleman dict ise, istediğiniz formatta string'e çevirin
      if isinstance(sentence, dict):
          # Örnek olarak, key-value çiftlerini birleştirerek string oluşturalım
          sentence = ", ".join([f"{k}: {v}" for k, v in sentence.items()])
      # String'e çevirdikten sonra dosyaya yazın
      f.write(sentence + "\n")

# %%
df

# %%
# Entity ve cümleleri birleştirip veri seti oluşturma
dataset = []
for idx, row in df.iterrows():
    for entity, sentences in row['entity_sentences'].items():
        combined_sentence = " ".join(sentences)
        dataset.append({
            'text': combined_sentence,
            'entity': entity,
            'label': row['label']
        })

dataset_df = pd.DataFrame(dataset)
print(dataset_df.head())

# Label encoding
label_encoder = LabelEncoder()
dataset_df['label'] = label_encoder.fit_transform(dataset_df['label'])
print(dataset_df[['text', 'entity', 'label']].head())


# %%
dataset_df

# %%
# Eğitim, doğrulama ve test setlerine ayırma
train_texts, temp_texts, train_labels, temp_labels = train_test_split(dataset_df['text'], dataset_df['label'], test_size=0.3, random_state=42)
val_texts, test_texts, val_labels, test_labels = train_test_split(temp_texts, temp_labels, test_size=0.5, random_state=42)

# BERT tokenizer kullanarak veri setini dönüştürme
tokenizer = BertTokenizer.from_pretrained('dbmdz/bert-base-turkish-uncased')

train_encodings = tokenizer(list(train_texts), truncation=True, padding=True)
val_encodings = tokenizer(list(val_texts), truncation=True, padding=True)
test_encodings = tokenizer(list(test_texts), truncation=True, padding=True)

class EntityDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

train_dataset = EntityDataset(train_encodings, list(train_labels))
val_dataset = EntityDataset(val_encodings, list(val_labels))
test_dataset = EntityDataset(test_encodings, list(test_labels))


# %%
import torch
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split

# Eğitim ve test setlerine ayırma
train_texts, val_texts, train_labels, val_labels = train_test_split(dataset_df['text'], dataset_df['label'], test_size=0.2, random_state=42)

# BERT tokenizer kullanarak veri setini dönüştürme
tokenizer = BertTokenizer.from_pretrained('dbmdz/bert-base-turkish-uncased')

train_encodings = tokenizer(list(train_texts), truncation=True, padding=True)
val_encodings = tokenizer(list(val_texts), truncation=True, padding=True)

class EntityDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

train_dataset = EntityDataset(train_encodings, list(train_labels))
val_dataset = EntityDataset(val_encodings, list(val_labels))

# Model tanımlama
model = BertForSequenceClassification.from_pretrained('dbmdz/bert-base-turkish-uncased', num_labels=len(label_encoder.classes_))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Modify the save_pretrained method to ensure contiguous tensors
def contiguous_save_pretrained(self, *args, **kwargs):
    for param in self.parameters():
        param.data = param.data.contiguous()
    return self._old_save_pretrained(*args, **kwargs)

model._old_save_pretrained = model.save_pretrained
model.save_pretrained = contiguous_save_pretrained.__get__(model)

# Eğitim argümanları
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=20,  # Epoch sayısını artırdık
    per_device_train_batch_size=64,  # Batch size'ı artırdık
    per_device_eval_batch_size=64,  # Batch size'ı artırdık
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
    save_strategy="epoch",  # Save at the end of each epoch
)

# Trainer tanımlama
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)

# Modeli eğitme
trainer.train()

# Modeli kaydetme
def contiguous_save_pretrained(model, output_dir):
    for param in model.parameters():
        param.data = param.data.contiguous()
    model.save_pretrained(output_dir)

# Trainer'dan sonra model parametrelerini contiguous hale getirme
for param in model.parameters():
    param.data = param.data.contiguous()

contiguous_save_pretrained(model, "entity_sentiment_model")
tokenizer.save_pretrained("entity_sentiment_tokenizer")


# %%
# Tahminleri yapma
val_preds = trainer.predict(val_dataset)
y_true = val_labels
y_pred = val_preds.predictions.argmax(-1)

# Label decoding
y_true = label_encoder.inverse_transform(y_true)
y_pred = label_encoder.inverse_transform(y_pred)

# Classification report ve confusion matrix
report = classification_report(y_true, y_pred)
print(report)

conf_matrix = confusion_matrix(y_true, y_pred)

# Confusion matrix görselleştirme
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()



