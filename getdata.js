const fs = require('fs');
const path = require('path');
const axios = require('axios');

async function downloadFile(url, outputPath) {
  const writer = fs.createWriteStream(outputPath);

  try {
    const response = await axios({
      url,
      method: 'GET',
      responseType: 'stream',
    });

    response.data.pipe(writer);

    return new Promise((resolve, reject) => {
      writer.on('finish', resolve);
      writer.on('error', reject);
    });
  } catch (error) {
    console.error(`เกิดข้อผิดพลาดในการดาวน์โหลด: ${error.message}`);
  }
}

async function main() {
  // เปลี่ยน URL เป็น Path ที่มีอยู่จริง (ตรวจสอบจากหน้าเว็บ Files and versions)
  // ตัวอย่างไฟล์ Train ส่วนที่ 1:
  const fileUrl = 'https://huggingface.co/datasets/Wongnai/wongnai_reviews/resolve/main/data/train-00000-of-00001.parquet';
  const fileName = 'wongnai_train.parquet';
  
  const downloadPath = path.resolve(__dirname, fileName);

  console.log(`กำลังเริ่มดาวน์โหลด: ${fileName}...`);
  await downloadFile(fileUrl, downloadPath);
  console.log(`ดาวน์โหลดสำเร็จ!`);
}

main();