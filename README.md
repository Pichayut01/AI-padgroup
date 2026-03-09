# AI Padgroup - Wongnai Review Sentiment Analysis

โปรเจกต์นี้เป็นการพัฒนาระบบวิเคราะห์ความรู้สึก (Sentiment Analysis) จากข้อความรีวิวของ Wongnai โดยแบ่งออกเป็นสองส่วนหลักคือ การจัดการข้อมูล (Data Preparation) ด้วย Node.js + DuckDB และการเทรนโมเดล (Model Training) ด้วย Python + PyTorch โดยใช้โมเดล Pre-trained ภาษาไทย `wangchanberta-base-att-spm-uncased`

## โครงสร้างโปรเจกต์ (Project Structure)

- `getdata.js`: สคริปต์ Node.js สำหรับดาวน์โหลดชุดข้อมูลรีวิว Wongnai รูปแบบ Parquet จาก Hugging Face
- `splitdata.js`: สคริปต์ Node.js สำหรับแบ่งข้อมูลดิบออกเป็น `train.csv` (80%) และ `test.csv` (20%) โดยใช้ DuckDB แบบ In-memory
- `dataset_split/train.py`: สคริปต์ Python สำหรับทำความสะอาดข้อความ เตรียมข้อมูลสำหรับเทรน และทำการ Fine-tune โมเดล WangchanBERTa ด้วย PyTorch
- `convertcsv.js`: สคริปต์เสริมสำหรับการแปลงไฟล์ข้อมูล (ถ้ามี)
- `models/` และ `wangchanberta_models/`: โฟลเดอร์สำหรับเก็บไฟล์โมเดลที่ผ่านการเทรนแล้ว (ถูกละเว้นใน Git)

## การติดตั้งและการใช้งาน (Installation & Usage)

### ส่วนที่ 1: การเตรียมข้อมูล (Node.js)

1. ติดตั้ง Dependencies ของ Node.js
   ```bash
   npm install
   ```
2. รันสคริปต์เพื่อดาวน์โหลดข้อมูล
   ```bash
   node getdata.js
   ```
3. แบ่งข้อมูลเป็น Train และ Test ชุดข้อมูลจะถูกสร้างในโฟลเดอร์ `dataset_split/`
   ```bash
   node splitdata.js
   ```

### ส่วนที่ 2: การเทรนโมเดล (Python)

1. เข้าไปยังโฟลเดอร์ `dataset_split/`
   ```bash
   cd dataset_split
   ```
2. สร้างและเปิดใช้งาน Virtual Environment (ตัวอย่างสำหรับ Windows)
   ```bash
   python -m venv venv_dl
   .\venv_dl\Scripts\activate
   ```
3. ติดตั้ง Dependencies สำหรับโมเดล
   ```bash
   pip install pandas numpy torch scikit-learn transformers datasets
   ```
4. เริ่มเทรนโมเดล
   ```bash
   python train.py
   ```
   *หมายเหตุ: สคริปต์จะทำการเทรนโมเดลและเซฟไว้ที่โฟลเดอร์ `../wangchanberta_models/best_wangchanberta` อัตโนมัติ*

## เทคนิคที่ใช้ในการพัฒนา

- **Text Cleaning:** ทำความสะอาดข้อความ ลบอักขระซ้ำ ลดความยาวเกินจำเป็น
- **Class Weights:** จัดการปัญหา Imbalanced Data (ดาวน้อย vs ดาวเยอะ) ให้อยู่ในสัดส่วนที่สมดุลขึ้น
- **Early Stopping:** ช่วยให้หยุดเทรนโมเดลได้ทันท่วงที หากความแม่นยำไม่พัฒนาขึ้นใน Epoch ถัดๆ ไป
- **Metrics:** เน้นการประเมินผลลัพธ์ผ่าน F1-Macro เพื่อให้สอดคล้องกับคลาสที่มีข้อมูลจำนวนน้อย
