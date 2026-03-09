const duckdb = require('duckdb');
const db = new duckdb.Database(':memory:'); // ใช้ Memory ชั่วคราว

const inputPath = 'wongnai_train.parquet';
const outputPath = 'wongnai_reviews.csv';

console.log('--- กำลังเริ่มแปลงไฟล์ด้วย DuckDB ---');

// ใช้คำสั่ง SQL ของ DuckDB ในการ Copy ข้อมูลออกเป็น CSV
db.all(`COPY (SELECT * FROM read_parquet('${inputPath}')) TO '${outputPath}' (HEADER, DELIMITER ',');`, (err, res) => {
    if (err) {
        console.error('เกิดข้อผิดพลาด:', err);
    } else {
        console.log('---------------------------');
        console.log('เสร็จเรียบร้อย!');
        console.log(`ไฟล์ CSV ถูกสร้างแล้วที่: ${outputPath}`);
    }
});