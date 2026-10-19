# Lab Data Preparation

Aplikasi Streamlit untuk simulasi data preparation dan pemodelan sederhana. Dibuat sebagai alat bantu pembelajaran Program Data Analytics.

## Cara menjalankan

```bash
pip install -r requirements.txt
streamlit run app.py
```

Browser akan terbuka di http://localhost:8501.

## Menu

| Menu | Isi |
|---|---|
| 1 · Data & Tipe | cuplikan data, tipe kolom, konversi teks angka (format Indonesia atau internasional), placeholder menjadi NaN, hapus duplikat, konversi tanggal dan ekstraksi fitur, hapus kolom |
| 2 · Missing Value | jumlah dan persentase kosong, peta nilai kosong, korelasi pola kosong, pola kombinasi, cek ketergantungan pada kategori |
| 3 · Imputasi | median, mean, konstanta, KNN, interpolasi, forward/backward fill, median per kelompok, modus; pratinjau dampak sebelum diterapkan; hapus baris atau kolom |
| 4 · EDA & Kualitas | skor kualitas, profil per kolom dengan catatan otomatis, statistik deskriptif, distribusi, korelasi, scatter, perbandingan kelompok |
| 5 · Outlier | IQR, Z-score, Modified Z-score (MAD), Isolation Forest; capping, hapus baris, ganti NaN, transformasi log |
| 6 · Pemodelan | pilih target, prediktor, dan algoritma; regresi (Linear, Ridge, Decision Tree, Random Forest, KNN) dan klasifikasi (Logistic, Decision Tree, Random Forest, KNN, Naive Bayes); cross-validation; metrik, grafik, importansi fitur, simulasi prediksi |
| 7 · Riwayat & Unduh | riwayat langkah, perbandingan sebelum dan sesudah, unduh data bersih |

## Catatan

- Semua langkah di menu 1 sampai 5 mengubah **data kerja**. Tombol "Kembalikan ke data asli" di sidebar membatalkan semuanya.
- Di menu Pemodelan, imputasi, penskalaan, dan one-hot encoding dijalankan ulang **di dalam pipeline** yang hanya di-fit pada data latih, sehingga tidak terjadi kebocoran data.
- Dua data contoh bawaan dan `contoh_upload_pencairan_kotor.csv` seluruhnya **fiktif**.
- Skenario demonstrasi yang disarankan: pada data contoh "Harga properti", latih regresi lebih dulu, lalu hapus outlier `harga_juta` di menu Outlier dan latih ulang. Pada pengujian, R² Linear Regression naik dari sekitar 0,45 menjadi 0,90.
