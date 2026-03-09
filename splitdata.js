const duckdb = require('duckdb');
const fs = require('fs');
const path = require('path');

// 1. ตั้งค่าพาธและชื่อโฟลเดอร์
const db = new duckdb.Database(':memory:');
const inputParquet = 'wongnai_train.parquet';
const outputFolder = path.join(__dirname, 'dataset_split');
const trainPath = path.join(outputFolder, 'train.csv');
const testPath = path.join(outputFolder, 'test.csv');

async function runSplit() {
    // 2. สร้าง Folder ถ้ายังไม่มี
    if (!fs.existsSync(outputFolder)) {
        fs.mkdirSync(outputFolder);
        console.log(`สร้างโฟลเดอร์: ${outputFolder}`);
    }

    console.log('--- กำลังเริ่มแบ่งข้อมูล 80:20 ---');

    // 3. ใช้ SQL ในการสุ่มและแบ่งข้อมูล
    // เราจะใช้ hash หรือ random() เพื่อแยกกลุ่มข้อมูล
    const splitQuery = `
        -- สร้างตารางชั่วคราวพร้อมเพิ่มเลขสุ่ม
        CREATE TABLE raw_data AS SELECT *, random() as split_key FROM read_parquet('${inputParquet}');

        -- Export ข้อมูล 80% ไปยัง train.csv
        COPY (SELECT * EXCLUDE split_key FROM raw_data WHERE split_key <= 0.8) 
        TO '${trainPath.replace(/\\/g, '/')}' (HEADER, DELIMITER ',');

        -- Export ข้อมูล 20% ไปยัง test.csv
        COPY (SELECT * EXCLUDE split_key FROM raw_data WHERE split_key > 0.8) 
        TO '${testPath.replace(/\\/g, '/')}' (HEADER, DELIMITER ',');
    `;

    db.exec(splitQuery, (err) => {
        if (err) {
            console.error('เกิดข้อผิดพลาด:', err);
        } else {
            console.log('---------------------------');
            console.log('แบ่งข้อมูลเสร็จสมบูรณ์!');
            console.log(`- Train data (80%): ${trainPath}`);
            console.log(`- Test data (20%): ${testPath}`);
        }
    });
}

runSplit();