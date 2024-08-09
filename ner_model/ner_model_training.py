import torch
from transformers import BertTokenizerFast, BertForTokenClassification, Trainer, TrainingArguments
from datasets import Dataset, load_metric

# Tokenizer ve modelin yüklenmesi
model_name = "dbmdz/bert-base-turkish-cased"
tokenizer = BertTokenizerFast.from_pretrained(model_name)
model = BertForTokenClassification.from_pretrained(model_name, num_labels=3)  # O, B-ORG, I-ORG

# Etiketlerin tanımlanması
label_list = ["O", "B-ORG", "I-ORG"]
label_to_id = {label: i for i, label in enumerate(label_list)}
id_to_label = {i: label for label, i in label_to_id.items()}

# Verileri okuma ve formatlama fonksiyonu
def read_data(file_paths):
    data = []
    for file_path in file_paths:
        tokens, labels = [], []
        with open(file_path, 'r', encoding='utf-8') as file:
            lines = file.readlines()
        
        for line in lines:
            parts = line.strip().split()
            if len(parts) == 0:  # Boş satır, cümle sonu
                if tokens:
                    data.append({"tokens": tokens, "labels": labels})
                    tokens, labels = [], []
            elif len(parts) == 2:  # Token ve etiket içeren satır
                token, label = parts
                tokens.append(token)
                labels.append(label_to_id[label])
            else:
                print(f"Skipping malformed line: {line.strip()}")
        
        if tokens:  # Son cümleyi ekle
            data.append({"tokens": tokens, "labels": labels})
    return data

# Mevcut ve yeni veri dosyaları
file_paths = [
    'formatted_data.txt', 
    'nerLabeled.txt'
]

# Veriyi okuma ve birleştirme
data = read_data(file_paths)
dataset = Dataset.from_list(data)

# Tokenize ve etiketleri hizalama fonksiyonu
def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(examples['tokens'], truncation=True, padding='max_length', max_length=512, is_split_into_words=True)
    all_labels = []
    for i, label in enumerate(examples['labels']):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        label_ids = []
        previous_word_id = None
        for word_id in word_ids:
            if word_id is None:
                label_ids.append(-100)
            elif word_id != previous_word_id:
                label_ids.append(label[word_id])
            else:
                label_ids.append(-100)
            previous_word_id = word_id
        all_labels.append(label_ids)
    tokenized_inputs["labels"] = all_labels
    return tokenized_inputs

# Verinin tokenizasyonu
tokenized_dataset = dataset.map(tokenize_and_align_labels, batched=True, remove_columns=["tokens", "labels"])

# Veriyi eğitim ve test setlerine ayırma
split_dataset = tokenized_dataset.train_test_split(test_size=0.1)

# Değerlendirme metriğinin yüklenmesi
metric = load_metric("seqeval", trust_remote_code=True)

# Metriği hesaplama fonksiyonu
def compute_metrics(p):
    predictions, labels = p
    predictions = torch.from_numpy(predictions)
    predictions = torch.argmax(predictions, dim=-1)

    true_predictions = []
    true_labels = []

    for prediction, label in zip(predictions, labels):
        pred_labels = []
        true_label_ids = []
        for pred, lab in zip(prediction, label):
            if lab != -100:
                pred_labels.append(id_to_label.get(pred.item(), "O"))
                true_label_ids.append(id_to_label.get(lab.item(), "O"))
        true_predictions.append(pred_labels)
        true_labels.append(true_label_ids)

    results = metric.compute(predictions=true_predictions, references=true_labels)
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }

# Eğitim argümanları
training_args = TrainingArguments(
    "test-ner",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=5,
    weight_decay=0.01,
)

# Tüm tensörlerin contiguous olmasını sağlama
class CustomTrainer(Trainer):
    def save_model(self, output_dir: str = None, _internal_call: bool = False):
        self.model.save_pretrained(output_dir, state_dict={k: v.contiguous() for k, v in self.model.state_dict().items()})
        self.tokenizer.save_pretrained(output_dir)

# Trainer'ın başlatılması
trainer = CustomTrainer(
    model=model,
    args=training_args,
    train_dataset=split_dataset["train"],
    eval_dataset=split_dataset["test"],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

# Eğitim
trainer.train()

# Modeli ve tokenizer'ı kaydetme
model_save_path = "fine-tuned-bert-ner-optimized"
trainer.save_model(model_save_path)
tokenizer.save_pretrained(model_save_path)