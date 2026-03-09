import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import os
import re
import platform
from sklearn.metrics import classification_report, accuracy_score, f1_score
from sklearn.utils.class_weight import compute_class_weight
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, EarlyStoppingCallback
from datasets import Dataset

# --- Configuration ---
MODEL_NAME = "airesearch/wangchanberta-base-att-spm-uncased"
OUTPUT_DIR = "../wangchanberta_models"
MAX_LEN = 256  # 🌟 เพิ่มความยาวเป็น 256 เพื่อเก็บรายละเอียดรีวิวให้ครบ
EPOCHS = 5     # 🌟 เพิ่มรอบเป็น 5 (เราจะใช้ Early Stopping ช่วยหยุดถ้ามันเริ่มตัน)

# 1. การทำความสะอาดข้อความเบื้องต้น
def clean_text(text):
    if not isinstance(text, str): return ""
    text = text.strip()
    text = re.sub(r'[\n\t\r]', ' ', text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'\s+', ' ', text)
    # 🌟 เทคนิคใหม่: ลดตัวอักษรซ้ำๆ เช่น "ดีมากกกกกก" -> "ดีมากก"
    text = re.sub(r'(.)\1{2,}', r'\1\1', text)
    return text

def map_stars_to_label(star):
    if star <= 1: return 0  # 0 = Negative (1, 2 stars are represented as 0, 1)
    elif star == 2: return 1 # 1 = Neutral (3 stars are represented as 2)
    else: return 2          # 2 = Positive (4, 5 stars are represented as 3, 4)

# 2. โหลดและเตรียมข้อมูล
print("🚀 [Step 1/5] Loading and Cleaning Data...")
df_train = pd.read_csv('train.csv')
df_test = pd.read_csv('test.csv')

review_col, star_col = df_train.columns[0], df_train.columns[1]

df_train = df_train.dropna(subset=[review_col, star_col])
df_test = df_test.dropna(subset=[review_col, star_col])

df_train['text'] = df_train[review_col].apply(clean_text)
df_train['label'] = df_train[star_col].apply(map_stars_to_label)
df_test['text'] = df_test[review_col].apply(clean_text)
df_test['label'] = df_test[star_col].apply(map_stars_to_label)

# 🌟 คำนวณ Class Weights เพื่อแก้ปัญหา Imbalanced Data (ดาวเยอะ vs ดาวน้อย)
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(df_train['label']),
    y=df_train['label']
)
print(f"⚖️ Class Weights สำหรับ (Neg, Neu, Pos): {class_weights}")

train_dataset = Dataset.from_pandas(df_train[['text', 'label']])
test_dataset = Dataset.from_pandas(df_test[['text', 'label']])

# 3. โหลด Tokenizer
print(f"⚙️ [Step 2/5] Loading Tokenizer ({MODEL_NAME})...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)

def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=MAX_LEN)

print("⏳ กำลัง Tokenize ข้อมูล...")
tokenized_train = train_dataset.map(tokenize_function, batched=True)
tokenized_test = test_dataset.map(tokenize_function, batched=True)

# 4. โหลด Model & ตรวจสอบ GPU
print("🧠 [Step 3/5] Loading Deep Learning Model...")
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=3)

if not torch.cuda.is_available():
    print("❌ ERROR: ไม่พบ GPU!")
    exit()

device = "cuda"
model.to(device)
class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)
print(f"💻 อุปกรณ์ที่ใช้ประมวลผล: {device.upper()} (PyTorch Version: {torch.__version__}) 🚀")

# 🌟 ปรับปรุงการวัดผลให้เพิ่ม F1-Score (ตัวชี้วัดที่แท้จริงสำหรับข้อมูลไม่สมดุล)
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average='macro')
    return {"accuracy": acc, "f1_macro": f1}

# 🌟 สร้าง Custom Trainer เพื่อบังคับใช้ Class Weights ตอนคำนวณ Loss
class ImbalancedTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        # ถ่วงน้ำหนักคลาสที่มีข้อมูลน้อยให้สำคัญขึ้น
        loss_fct = nn.CrossEntropyLoss(weight=class_weights_tensor)
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss

# 🌟 อัปเดตพารามิเตอร์การ Train
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    eval_strategy="epoch",       
    save_strategy="epoch",
    learning_rate=2e-5,          
    per_device_train_batch_size=16, 
    per_device_eval_batch_size=16,
    num_train_epochs=EPOCHS,
    weight_decay=0.01,
    warmup_ratio=0.1, # 🌟 เพิ่ม Warmup ให้โมเดลตั้งตัวได้ดีขึ้นในช่วงแรก
    load_best_model_at_end=True,
    metric_for_best_model="f1_macro", # 🌟 เลือกโมเดลที่ F1-Macro ดีที่สุด ไม่ใช่แค่ Accuracy
    fp16=True, 
    dataloader_pin_memory=True, 
)

trainer = ImbalancedTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_test,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)] # 🌟 หยุดก่อนถ้าโมเดลเริ่มตัน (2 epochs ไม่ดีขึ้น=หยุด)
)

# 5. เริ่ม Train Model
print("🔥 [Step 4/5] Training Deep Learning Model with Class Weights...")
trainer.train()

# 6. วัดผลลัพธ์สุดท้าย
print("📊 [Step 5/5] Final Evaluation...")
predictions = trainer.predict(tokenized_test)
y_pred = predictions.predictions.argmax(-1)
y_true = tokenized_test['label']

target_names = ['Negative (0-2)', 'Neutral (3)', 'Positive (4-5)']
print("\n" + "="*50)
print(f"✅ Final Accuracy: {accuracy_score(y_true, y_pred):.2%}")
print(f"✅ Final F1-Macro: {f1_score(y_true, y_pred, average='macro'):.2%}")
print("="*50)
print(classification_report(y_true, y_pred, target_names=target_names))

# 7. เซฟโมเดล
trainer.save_model(f"{OUTPUT_DIR}/best_wangchanberta")
tokenizer.save_pretrained(f"{OUTPUT_DIR}/best_wangchanberta")
print(f"✨ Deep Learning Model saved to {OUTPUT_DIR}/best_wangchanberta")