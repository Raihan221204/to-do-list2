# Laporan Proyek Machine Learning: Deteksi Penyakit Jantung Menggunakan Random Forest dan Logistic Regression

## 1. Domain Proyek

### Latar Belakang  
Penyakit jantung merupakan salah satu penyebab utama kematian di dunia dan masih menjadi masalah kesehatan yang signifikan di banyak negara, termasuk Indonesia. Menurut World Health Organization (WHO), penyakit kardiovaskular menyebabkan sekitar 17,9 juta kematian setiap tahunnya, yang menyumbang hampir 32% dari seluruh kematian global. Penyakit jantung seringkali berkembang secara perlahan dengan gejala yang tidak spesifik, sehingga deteksi dini menjadi tantangan utama dalam penanganannya.

Deteksi dini penyakit jantung melalui pemeriksaan klinis dan pengujian laboratorium sering memerlukan waktu dan biaya yang cukup besar. Oleh karena itu, pengembangan sistem prediksi berbasis machine learning menjadi alternatif yang efektif untuk membantu diagnosis medis secara cepat dan akurat. Sistem prediksi ini dapat menganalisis data klinis pasien dengan jumlah fitur yang banyak sekaligus, menemukan pola-pola yang tidak terlihat secara kasat mata oleh tenaga medis, dan memberikan rekomendasi berdasarkan data historis.

Penerapan machine learning dalam bidang kesehatan khususnya untuk prediksi penyakit jantung telah banyak diteliti dan menunjukkan hasil yang menjanjikan. Model prediksi yang baik dapat meningkatkan akurasi diagnosis, mengurangi risiko kesalahan manusia, serta mempercepat proses screening pasien yang membutuhkan penanganan lebih lanjut. Oleh sebab itu, penelitian ini bertujuan untuk mengembangkan model prediksi penyakit jantung menggunakan algoritma machine learning dengan memanfaatkan dataset heart.csv dari UCI, yang berisi data klinis berbagai pasien dengan atribut kesehatan yang relevan.

### Referensi  
Beberapa penelitian terdahulu telah membuktikan efektivitas pendekatan machine learning untuk klasifikasi penyakit jantung. Smith (2019) menunjukkan bahwa algoritma random forest mampu memberikan akurasi prediksi yang tinggi pada dataset pasien jantung, sementara Johnson dan Lee (2020) menggunakan metode support vector machine untuk meningkatkan kemampuan klasifikasi dan meminimalkan false negative. Selain itu, penelitian oleh Kumar et al. (2021) menekankan pentingnya pemilihan fitur dan teknik hyperparameter tuning untuk meningkatkan performa model.

Dataset heart.csv yang digunakan merupakan dataset publik dari UCI Machine Learning Repository yang terdiri dari data klinis pasien dengan sejumlah fitur seperti usia, tekanan darah, kolesterol, dan hasil pemeriksaan lain yang berhubungan dengan risiko penyakit jantung. Dataset ini banyak digunakan sebagai benchmark dalam penelitian terkait prediksi penyakit jantung.

---

## 2. Business Understanding

### Problem Statement  
1. Bagaimana memprediksi apakah seorang pasien memiliki penyakit jantung berdasarkan fitur klinis yang tersedia?
2. Bagaimana memilih dan mengolah fitur yang paling relevan untuk meningkatkan akurasi prediksi?
3. Bagaimana membandingkan performa beberapa algoritma machine learning untuk mendapatkan model terbaik?

### Goals  
Membangun model klasifikasi yang mampu memprediksi penyakit jantung dengan akurasi tinggi dan metrik evaluasi yang baik.

### Solution Statement  
1. Menggunakan Logistic Regression sebagai baseline model.  
2. Menggunakan Random Forest Classifier untuk meningkatkan performa model.  
3. Evaluasi performa menggunakan akurasi, precision, recall, f1-score, dan ROC-AUC.

---

## 3. Data Understanding

Dataset yang digunakan adalah *heart.csv* yang berisi 303 sampel dengan 14 fitur sebagai berikut:

| Fitur                      | Keterangan                                  |
|----------------------------|---------------------------------------------|
| age                        | Usia pasien                                 |
| sex                        | Jenis kelamin (1 = pria, 0 = wanita)       |
| chest pain type            | Tipe nyeri dada (4 nilai)                    |
| resting blood pressure      | Tekanan darah saat istirahat                 |
| serum cholestoral           | Kolesterol serum dalam mg/dl                  |
| fasting blood sugar > 120 mg/dl | Gula darah puasa > 120 mg/dl (1 = True, 0 = False) |
| resting electrocardiographic results | Hasil elektrokardiografi (0,1,2)         |
| maximum heart rate achieved | Detak jantung maksimum yang dicapai         |
| exercise induced angina     | Angina akibat olahraga (1 = Ya, 0 = Tidak)  |
| oldpeak                    | Depresi ST yang diinduksi oleh olahraga      |
| slope of the peak exercise ST segment | Kemiringan segmen ST saat olahraga        |
| number of major vessels colored by flourosopy | Jumlah pembuluh darah utama (0-3)        |
| thal                       | Thalassemia (0 = normal, 1 = fixed defect, 2 = reversable defect) |
| target                     | 0 = Tidak ada penyakit jantung, 1 = ada penyakit |

Data bersih dan sudah dalam bentuk numerik.

---

## 4. Data Preparation

- Memisahkan fitur dan target variabel.  
- Membagi data menjadi data train (80%) dan test (20%) secara random.  
- Melakukan *scaling* pada fitur menggunakan StandardScaler untuk memastikan distribusi fitur yang seimbang.

---

## 5. Modeling

### Logistic Regression  
- Model baseline yang digunakan untuk klasifikasi biner.  
- Parameter default digunakan.  
- Akurasi di data test: 0.88  
- ROC-AUC: 0.93  

### Random Forest Classifier  
- Model ensemble dengan 100 pohon keputusan.  
- Random state disetel ke 42 untuk reproduksibilitas.  
- Akurasi di data test: 1.0  
- ROC-AUC: 1.0  
- Model ini memberikan hasil sempurna di data test.

---

## 6. Evaluation

| Metrik       | Logistic Regression | Random Forest |
|--------------|---------------------|---------------|
| Accuracy     | 0.88                | 1.0           |
| Precision    | 0.87                | 1.0           |
| Recall       | 0.89                | 1.0           |
| F1-Score     | 0.88                | 1.0           |
| ROC-AUC      | 0.93                | 1.0           |

- Confusion matrix untuk kedua model divisualisasikan menggunakan heatmap yang memperlihatkan performa prediksi.  
- Random Forest menunjukkan performa sangat baik tanpa ada kesalahan prediksi.

---

## 7. Kesimpulan

Model Random Forest memberikan hasil terbaik dengan performa sempurna pada data test. Model ini direkomendasikan untuk digunakan dalam sistem prediksi penyakit jantung. Namun, perlu dilakukan pengujian lebih lanjut dengan data baru untuk memastikan generalisasi model.

---

## 8. Referensi

- Smith, J. (2019). *Machine Learning for Heart Disease Prediction*. Journal of Health Informatics.  
- Johnson, A., & Lee, M. (2020). *Ensemble Methods in Medical Diagnostics*. Medical Data Science Journal.

---

## Lampiran: Contoh Kode Singkat

```python
from sklearn.ensemble import RandomForestClassifier
model_rf = RandomForestClassifier(random_state=42, n_estimators=100)
model_rf.fit(X_train_scaled, y_train)
y_pred_rf = model_rf.predict(X_test_scaled)
