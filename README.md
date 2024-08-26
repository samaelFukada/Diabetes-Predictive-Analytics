# Laporan Proyek Machine Learning - Ahmad Reginald Syahiran

## Domain Proyek

Domain yang dipilih untuk proyek machine learning ini adalah Kesehatan, dengan judul Predictive Analytics: Diagnosa Diabetes

**Latar Belakang**:
![foto](https://storage.googleapis.com/kaggle-datasets-images/5192952/8665939/4ce14f3760302cc35f6b96a15b7c25d0/dataset-cover.jpg?t=2024-06-11-14-15-47)
Diabetes mellitus merupakan salah satu penyakit kronis yang semakin meningkat prevalensinya di seluruh dunia, termasuk di Indonesia. Menurut data dari International Diabetes Federation (IDF) pada tahun 2021, Indonesia menempati peringkat ketujuh dengan jumlah penderita diabetes terbesar di dunia. Penyakit ini dapat menyebabkan komplikasi serius, termasuk penyakit jantung, stroke, dan kerusakan ginjal, yang berpotensi menurunkan kualitas hidup dan meningkatkan angka kematian[^1^]. Mengingat dampak signifikan tersebut, upaya deteksi dini melalui analisis prediktif menjadi sangat penting. Dengan menggunakan teknologi machine learning, model prediktif dapat dibangun untuk mengidentifikasi individu yang berisiko tinggi terkena diabetes berdasarkan data kesehatan yang tersedia, seperti riwayat keluarga, gaya hidup, dan indikator klinis. Penelitian menunjukkan bahwa penggunaan model prediktif dalam diagnosis dini dapat meningkatkan efektivitas intervensi medis dan mengurangi komplikasi terkait diabetes [^2^]. Oleh karena itu, pengembangan analisis prediktif ini menjadi solusi krusial dalam menghadapi tantangan peningkatan prevalensi diabetes di masa mendatang.

---

[^1^]: Hestiana, D.W. (2017). FAKTOR-FAKTOR YANG BERHUBUNGAN DENGAN KEPATUHAN DALAM PENGELOLAAN DIET PADA PASIEN RAWAT JALAN DIABETES MELLITUS TIPE 2 DI KOTA SEMARANG. Journal of Health Education, 2, 137-145.
[^2^]: Dutta, A., Hasan, M., Ahmad, M., Awal, M., Islam, M., Masud, M., & Meshref, H. (2022). Early Prediction of Diabetes Using an Ensemble of Machine Learning Models. International Journal of Environmental Research and Public Health, 19. https://doi.org/10.3390/ijerph191912378.
[^3^]: DeepAI. (n.d.). Random forest. DeepAI. https://deepai.org/machine-learning-glossary-and-terms/random-forest.
[^4^]: Brownlee, J. (2020, June 16). Extra Trees ensemble with Python. Machine Learning Mastery. https://machinelearningmastery.com/extra-trees-ensemble-with-python/
[^5^]: Budholiya, K., Shrivastava, S., & Sharma, V. (2020). An optimized XGBoost based diagnostic system for effective prediction of heart disease. J. King Saud Univ. Comput. Inf. Sci., 34, 4514-4523. https://doi.org/10.1016/j.jksuci.2020.10.013.

## Business Understanding

Pengembangan model prediksi diagnosis diabetes memiliki potensi untuk memberikan manfaat bagi berbagai pihak, termasuk tenaga medis dan pasien. Model ini dapat membantu dalam deteksi dini diabetes, meningkatkan efektivitas pengelolaan penyakit, dan meningkatkan kualitas hidup pasien. Contoh potensi manfaat dari prediksi diagnosis diabetes yang akurat adalah membantu tenaga medis dalam pengambilan keputusan klinis serta memungkinkan pasien untuk mengambil langkah pencegahan lebih awal guna mencegah komplikasi di masa depan.

### Problem Statements
Berdasarkan latar belakang di atas, berikut ini merupakan rincian masalah yang dapat diselesaikan pada proyek ini:
- Bagaimana membuat model machine learning yang dapat memprediksi diagnosa diabetes berdasarkan data tabular?
- Model yang seperti apa yang memiliki akurasi paling baik?
- Bagaimana model ini dapat membantu dokter tenaga medis dalam meningkatkan efektivitas intervensi medis dan mengurangi komplikasi terkait diabetes?


### Goals

Menjelaskan tujuan dari pernyataan masalah:
- Membuat model machine learning yang dapat memprediksi diagnosa diabetes berdasarkan data tabular.
- Membandingkan beberapa algoritma model untuk menemukan akurasi terbaik dalam memprediksi diagnosa diabetes.
- Mengembangkan model yang dapat membantu dokter tenaga medis dalam meningkatkan efektivitas intervensi medis dan mengurangi komplikasi terkait diabetes.

### Solution statements
- Menganalisis data dengan melakukan univariate analysis dan multivariate analysis. Memahami data juga dapat dilakukan dengan visualisasi. Memahami data dapat membantu untuk mengetahui kolerasi matrix antar fitur dan mendeteksi outlier.
- Melakukan proses data cleaning dan normalisai data agar mendapat prediksi yang baik.
- Membuat beberapa variasi model untuk mendapatkan model yang paling baik dari beberapa model. Diantaranya adalah menggunakan:
    * XGBoost (Extreme Gradient Boosting) adalah algoritma pembelajaran mesin yang sangat kuat dan populer untuk tugas klasifikasi dan regresi. Algoritma ini dikenal karena kemampuannya dalam menangani data yang tidak seimbang, efisiensi komputasi, dan kinerja yang tinggi dalam berbagai aplikasi.Dalam aplikasi medis, XGBoost digunakan untuk memprediksi penyakit jantung dan menunjukkan akurasi prediksi yang tinggi[^5^].
    * Random Forest adalah algoritma machine learning yang kuat yang dapat digunakan untuk berbagai tugas termasuk regresi dan klasifikasi. Ini adalah metode ensemble, yang berarti bahwa model random forest terdiri dari banyak decision tree kecil, yang disebut estimator, yang masing-masing menghasilkan prediksi mereka sendiri. Random forest menggabungkan prediksi estimator untuk menghasilkan prediksi yang lebih akurat[^3^].
    * Extra trees classifier adalah sejumlah besar pohon keputusan yang belum dipangkas dari kumpulan data pelatihan. Prediksi dibuat dengan merata-ratakan prediksi pohon keputusan dalam kasus regresi atau menggunakan suara terbanyak dalam kasus klasifikasi[^4^].

## Data Understanding
### EDA - Deskripsi Variabel
**Informasi Datasets**

| Jenis | Keterangan |
| ------ | ------ |
| Title | _Diabetes Health Dataset Analysis_ |
| Source | [Kaggle](https://www.kaggle.com/datasets/rabieelkharoua/diabetes-health-dataset-analysis) |
| Maintainer | [Rabie El Kharoua](https://www.kaggle.com/rabieelkharoua) |
| License | [Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/) |
| Visibility | Publik |
| Tags | _Tabular, Health Condition, Diabetes, Heart Condition, Binary Classification_ |
| Usability | 10.00 |

### Variabel-variabel pada Restaurant UCI dataset adalah sebagai berikut:
## Informasi Pasien

### ID Pasien

- **PatientID**: Identifikasi unik yang diberikan kepada setiap pasien (6000 hingga 7878).

### Rincian Demografis

- **Age**: Usia pasien berkisar dari 20 hingga 90 tahun.
- **Gender**: Jenis kelamin pasien, di mana 0 mewakili Laki-laki dan 1 mewakili Perempuan.
- **Ethnicity**: Etnisitas pasien, dikodekan sebagai berikut:
  - 0: Kaukasia
  - 1: Afrika Amerika
  - 2: Asia
  - 3: Lainnya
- **SocioeconomicStatus**: Status sosial ekonomi pasien, dikodekan sebagai berikut:
  - 0: Rendah
  - 1: Menengah
  - 2: Tinggi
- **EducationLevel**: Tingkat pendidikan pasien, dikodekan sebagai berikut:
  - 0: Tidak ada
  - 1: Sekolah Menengah
  - 2: Sarjana
  - 3: Lebih Tinggi

### Faktor Gaya Hidup

- **BMI**: Indeks Massa Tubuh pasien, berkisar dari 15 hingga 40.
- **Smoking**: Status merokok, di mana 0 menunjukkan Tidak dan 1 menunjukkan Ya.
- **AlcoholConsumption**: Konsumsi alkohol mingguan dalam satuan, berkisar dari 0 hingga 20.
- **PhysicalActivity**: Aktivitas fisik mingguan dalam jam, berkisar dari 0 hingga 10.
- **DietQuality**: Skor kualitas diet, berkisar dari 0 hingga 10.
- **SleepQuality**: Skor kualitas tidur, berkisar dari 4 hingga 10.

### Riwayat Medis

- **FamilyHistoryDiabetes**: Riwayat keluarga dengan diabetes, di mana 0 menunjukkan Tidak dan 1 menunjukkan Ya.
- **GestationalDiabetes**: Riwayat diabetes gestasional, di mana 0 menunjukkan Tidak dan 1 menunjukkan Ya.
- **PolycysticOvarySyndrome**: Kehadiran sindrom ovarium polikistik, di mana 0 menunjukkan Tidak dan 1 menunjukkan Ya.
- **PreviousPreDiabetes**: Riwayat pre-diabetes sebelumnya, di mana 0 menunjukkan Tidak dan 1 menunjukkan Ya.
- **Hypertension**: Kehadiran hipertensi, di mana 0 menunjukkan Tidak dan 1 menunjukkan Ya.

### Pengukuran Klinis

- **SystolicBP**: Tekanan darah sistolik, berkisar dari 90 hingga 180 mmHg.
- **DiastolicBP**: Tekanan darah diastolik, berkisar dari 60 hingga 120 mmHg.
- **FastingBloodSugar**: Tingkat gula darah puasa, berkisar dari 70 hingga 200 mg/dL.
- **HbA1c**: Tingkat Hemoglobin A1c, berkisar dari 4,0% hingga 10,0%.
- **SerumCreatinine**: Tingkat kreatinin serum, berkisar dari 0,5 hingga 5,0 mg/dL.
- **BUNLevels**: Tingkat Urea Nitrogen Darah, berkisar dari 5 hingga 50 mg/dL.
- **CholesterolTotal**: Tingkat kolesterol total, berkisar dari 150 hingga 300 mg/dL.
- **CholesterolLDL**: Tingkat kolesterol LDL (Low-density lipoprotein), berkisar dari 50 hingga 200 mg/dL.
- **CholesterolHDL**: Tingkat kolesterol HDL (High-density lipoprotein), berkisar dari 20 hingga 100 mg/dL.
- **CholesterolTriglycerides**: Tingkat trigliserida, berkisar dari 50 hingga 400 mg/dL.

### Obat-obatan

- **AntihypertensiveMedications**: Penggunaan obat antihipertensi, di mana 0 menunjukkan Tidak dan 1 menunjukkan Ya.
- **Statins**: Penggunaan statin, di mana 0 menunjukkan Tidak dan 1 menunjukkan Ya.
- **AntidiabeticMedications**: Penggunaan obat antidiabetes, di mana 0 menunjukkan Tidak dan 1 menunjukkan Ya.

### Gejala dan Kualitas Hidup

- **FrequentUrination**: Kehadiran sering buang air kecil, di mana 0 menunjukkan Tidak dan 1 menunjukkan Ya.
- **ExcessiveThirst**: Kehadiran rasa haus yang berlebihan, di mana 0 menunjukkan Tidak dan 1 menunjukkan Ya.
- **UnexplainedWeightLoss**: Kehadiran penurunan berat badan yang tidak dapat dijelaskan, di mana 0 menunjukkan Tidak dan 1 menunjukkan Ya.
- **FatigueLevels**: Tingkat kelelahan, berkisar dari 0 hingga 10.
- **BlurredVision**: Kehadiran penglihatan kabur, di mana 0 menunjukkan Tidak dan 1 menunjukkan Ya.
- **SlowHealingSores**: Kehadiran luka yang sembuh dengan lambat, di mana 0 menunjukkan Tidak dan 1 menunjukkan Ya.
- **TinglingHandsFeet**: Kehadiran kesemutan di tangan atau kaki, di mana 0 menunjukkan Tidak dan 1 menunjukkan Ya.
- **QualityOfLifeScore**: Skor kualitas hidup, berkisar dari 0 hingga 100.

### Paparan Lingkungan dan Pekerjaan

- **HeavyMetalsExposure**: Paparan logam berat, di mana 0 menunjukkan Tidak dan 1 menunjukkan Ya.
- **OccupationalExposureChemicals**: Paparan bahan kimia berbahaya dalam pekerjaan, di mana 0 menunjukkan Tidak dan 1 menunjukkan Ya.
- **WaterQuality**: Kualitas air, di mana 0 menunjukkan Baik dan 1 menunjukkan Buruk.

### Perilaku Kesehatan

- **MedicalCheckupsFrequency**: Frekuensi pemeriksaan medis per tahun, berkisar dari 0 hingga 4.
- **MedicationAdherence**: Skor kepatuhan terhadap pengobatan, berkisar dari 0 hingga 10.
- **HealthLiteracy**: Skor literasi kesehatan, berkisar dari 0 hingga 10.

## Informasi Diagnosa (Variabel Target)

- **Diagnosis**: Status diagnosis untuk Diabetes, di mana 0 menunjukkan Tidak dan 1 menunjukkan Ya.

Berikut gambaran informasi dataset:
| PatientID | Age | Gender | Ethnicity | SocioeconomicStatus | EducationLevel |   BMI   | Smoking | AlcoholConsumption | PhysicalActivity | TinglingHandsFeet | QualityOfLifeScore | HeavyMetalsExposure | OccupationalExposureChemicals | WaterQuality | MedicalCheckupsFrequency | MedicationAdherence | HealthLiteracy | Diagnosis | DoctorInCharge |
|-----------|-----|--------|-----------|---------------------|----------------|---------|---------|--------------------|------------------|-------------------|--------------------|---------------------|-----------------------------|--------------|--------------------------|---------------------|----------------|-----------|----------------|
|      6000 |  44 |      0 |         1 |                   2 |              1 | 32.9853 |       1 |              4.499 |            2.443 |                 1 |              73.77 |                   0 |                           0 |            0 |                    1.783 |               4.487 |          7.211 |         1 | Confidential   |
|      6001 |  51 |      1 |         0 |                   1 |              2 | 39.9168 |       0 |              1.579 |            8.301 |                 0 |              91.45 |                   0 |                           0 |            1 |                    3.381 |               5.962 |          5.025 |         1 | Confidential   |
|      6002 |  89 |      1 |         0 |                   1 |              3 | 19.7823 |       0 |              1.177 |            6.103 |                 0 |              54.49 |                   0 |                           0 |            0 |                    2.701 |               8.951 |          7.035 |         0 | Confidential   |
|      6003 |  21 |      1 |         1 |                   1 |              2 | 32.3769 |       1 |              1.715 |            8.645 |                 0 |              77.87 |                   0 |                           0 |            1 |                    1.409 |               3.125 |          4.718 |         0 | Confidential   |
|      6004 |  27 |      1 |         0 |                   1 |              3 | 16.8086 |       0 |             15.463 |            4.629 |                 0 |              37.73 |                   0 |                           0 |            0 |                    1.218 |               6.978 |          7.888 |         0 | Confidential   |
**Tabel 1.** EDA Deskripsi Variabel
### Informasi Dataset:
| #  | Column                         | Non-Null Count | Dtype   |
|----|--------------------------------|----------------|---------|
| 0  | PatientID                      | 1879 non-null  | int64   |
| 1  | Age                            | 1879 non-null  | int64   |
| 2  | Gender                         | 1879 non-null  | int64   |
| 3  | Ethnicity                      | 1879 non-null  | int64   |
| 4  | SocioeconomicStatus            | 1879 non-null  | int64   |
| 5  | EducationLevel                 | 1879 non-null  | int64   |
| 6  | BMI                            | 1879 non-null  | float64 |
| 7  | Smoking                        | 1879 non-null  | int64   |
| 8  | AlcoholConsumption             | 1879 non-null  | float64 |
| 9  | PhysicalActivity               | 1879 non-null  | float64 |
| 10 | DietQuality                    | 1879 non-null  | float64 |
| 11 | SleepQuality                   | 1879 non-null  | float64 |
| 12 | FamilyHistoryDiabetes          | 1879 non-null  | int64   |
| 13 | GestationalDiabetes            | 1879 non-null  | int64   |
| 14 | PolycysticOvarySyndrome        | 1879 non-null  | int64   |
| 15 | PreviousPreDiabetes            | 1879 non-null  | int64   |
| 16 | Hypertension                   | 1879 non-null  | int64   |
| 17 | SystolicBP                     | 1879 non-null  | int64   |
| 18 | DiastolicBP                    | 1879 non-null  | int64   |
| 19 | FastingBloodSugar              | 1879 non-null  | float64 |
| 20 | HbA1c                          | 1879 non-null  | float64 |
| 21 | SerumCreatinine                | 1879 non-null  | float64 |
| 22 | BUNLevels                      | 1879 non-null  | float64 |
| 23 | CholesterolTotal               | 1879 non-null  | float64 |
| 24 | CholesterolLDL                 | 1879 non-null  | float64 |
| 25 | CholesterolHDL                 | 1879 non-null  | float64 |
| 26 | CholesterolTriglycerides       | 1879 non-null  | float64 |
| 27 | AntihypertensiveMedications    | 1879 non-null  | int64   |
| 28 | Statins                        | 1879 non-null  | int64   |
| 29 | AntidiabeticMedications        | 1879 non-null  | int64   |
| 30 | FrequentUrination              | 1879 non-null  | int64   |
| 31 | ExcessiveThirst                | 1879 non-null  | int64   |
| 32 | UnexplainedWeightLoss          | 1879 non-null  | int64   |
| 33 | FatigueLevels                  | 1879 non-null  | float64 |
| 34 | BlurredVision                  | 1879 non-null  | int64   |
| 35 | SlowHealingSores               | 1879 non-null  | int64   |
| 36 | TinglingHandsFeet              | 1879 non-null  | int64   |
| 37 | QualityOfLifeScore             | 1879 non-null  | float64 |
| 38 | HeavyMetalsExposure            | 1879 non-null  | int64   |
| 39 | OccupationalExposureChemicals  | 1879 non-null  | int64   |
| 40 | WaterQuality                   | 1879 non-null  | int64   |
| 41 | MedicalCheckupsFrequency       | 1879 non-null  | float64 |
| 42 | MedicationAdherence            | 1879 non-null  | float64 |
| 43 | HealthLiteracy                 | 1879 non-null  | float64 |
| 44 | Diagnosis                      | 1879 non-null  | int64   |
| 45 | DoctorInCharge                 | 1879 non-null  | object  |
**Tabel 2.** Informasi Dataset

Dari informasi - informasi yang dipaparkan di atas dapat disimpulkan:
- dataset ini telah di bersihkan dan normalisasi terlebih dahulu oleh pembuat, sehingga mudah digunakan dan ramah bagi pemula.
- Dataset berupa CSV (Comma-Seperated Values).
- Dataset memiliki 1879 sample dengan 46 fitur.
- Dataset memiliki 18 fitur bertipe float64, 27 fitur bertipe int64 dan 1 fitur bertipe object.
- terdapat 21 numerical features.
- terdapat 23 categorical features.
- kolom 'PatientID' dan 'DoctorInCharge' tidak akan berpengaruh terhadap model yang akan dibuat.
- tidak ada missing value pada dataset.* 

### EDA - Menangani Outliers
![Menangani Outliers](https://i.ibb.co.com/CHCXLG2/Unknown.png)
**Gambar 1.** Boxplot

Berdasarkan boxplot, dapat disimpulkan bahwa tidak ada outliers pada dataset.

### EDA - Univariate Analysis
#### Numerical Features
![Univariate Analysis Numerical Features](https://i.ibb.co.com/3SFP9FY/univariate-analysis-numerical.png)
**Gambar 2.** Distribusi Numerical Features
Distribusi Data:
- Sebagian besar variabel memiliki distribusi yang relatif merata, tanpa adanya puncak yang sangat menonjol atau distribusi yang sangat miring. Ini mungkin menunjukkan bahwa data tersebar cukup merata di berbagai kategori atau rentang nilai.

Jumlah Observasi:
- Jumlah observasi cukup konsisten di seluruh variabel, karena tinggi batang histogram tidak menunjukkan perbedaan besar antara satu histogram dengan yang lainnya.
Bentuk Distribusi:
- Histogram memiliki bentuk yang mendekati distribusi normal atau seragam.

#### Categorical Features
![Univariate categorical](https://i.ibb.co.com/vJ46Pfj/univariate-categorical-features.png)
**Gambar 3.** Distribusi Categorical Features
- Gender: Distribusi gender terlihat cukup seimbang dengan jumlah yang hampir sama antara laki-laki dan perempuan.
- Ethnicity: Kelompok etnis 0 mendominasi data ini, diikuti oleh etnis 2, dengan etnis 1 dan 3 memiliki jumlah yang jauh lebih kecil.
- Socioeconomic Status: Status sosial ekonomi 1 memiliki jumlah terbesar, diikuti oleh status 2 dan kemudian status 0.
- Education Level: Tingkat pendidikan 2 dan 1 memiliki jumlah terbesar, sementara tingkat 3 dan 0 memiliki jumlah yang lebih rendah.
- Smoking: Mayoritas sampel tidak merokok (label 0), sementara yang merokok (label 1) lebih sedikit.
- Family History of Diabetes: Sebagian besar orang tidak memiliki riwayat keluarga dengan diabetes (label 0), tetapi ada sebagian yang memiliki (label 1).
- Gestational Diabetes: Sebagian besar sampel tidak memiliki diabetes gestasional (label 0), dan sangat sedikit yang memilikinya (label 1).
- Polycystic Ovary Syndrome: Sangat sedikit sampel yang memiliki sindrom ovarium polikistik (label 1), dengan mayoritas tidak memilikinya (label 0).
- Previous Pre-diabetes: Mayoritas sampel tidak memiliki riwayat pre-diabetes (label 0), dengan sebagian kecil yang memiliki (label 1).
- Hypertension: Sebagian besar sampel tidak memiliki hipertensi (label 0), dan hanya sebagian kecil yang memilikinya (label 1).
- Antihypertensive Medications: Sebagian besar sampel tidak menggunakan obat antihipertensi (label 0), dengan sebagian kecil yang menggunakannya (label 1).
- Statins: Sebagian besar sampel tidak menggunakan statin (label 0), namun ada sebagian yang menggunakan (label 1).
- Antidiabetic Medications: Mayoritas tidak menggunakan obat antidiabetes (label 0), dengan sebagian kecil yang menggunakannya (label 1).
- Frequent Urination: Sebagian besar sampel tidak melaporkan sering buang air kecil (label 0), dengan sebagian kecil yang melaporkannya (label 1).
- Excessive Thirst: Sebagian besar tidak mengalami haus berlebihan (label 0), dengan sebagian kecil yang mengalaminya (label 1).
- Unexplained Weight Loss: Mayoritas tidak mengalami penurunan berat badan yang tidak dapat dijelaskan (label 0), dengan sebagian kecil yang mengalaminya (label 1).
- Blurred Vision: Sebagian besar sampel tidak mengalami penglihatan kabur (label 0), dengan sebagian kecil yang mengalaminya (label 1).
- Tingling in Hands/Feet: Mayoritas sampel tidak mengalami kesemutan di tangan/kaki (label 0), dengan sebagian kecil yang mengalaminya (label 1).
- Heavy Metal Exposure: Mayoritas tidak terpapar logam berat (label 0), dengan sebagian kecil yang terpapar (label 1).
- Occupational Exposure to Chemicals: Sebagian besar sampel tidak terpapar bahan kimia di tempat kerja (label 0), dengan sebagian kecil yang terpapar (label 1).
- Water Quality: Mayoritas tidak mengalami masalah dengan kualitas air (label 0), dengan sebagian kecil yang mengalaminya (label 1).
- Diagnosis: Distribusi diagnosis menunjukkan bahwa mayoritas sampel didiagnosis negatif (label 0), dengan sebagian yang didiagnosis positif (label 1).

### EDA - Multivariate Analysis
#### Categorical Features
![multivariateanalysis categorical](https://i.ibb.co.com/qJsH34w/multivariate-categorical.png)
**Gambar 4.** Rata-rata Diagnosa terhadap Categorical Features
Gender:
- Tampaknya terdapat perbedaan yang signifikan antara jumlah pria dan wanita yang didiagnosis dengan kondisi tertentu (*Diagnosis 1*).
- Lebih banyak wanita yang tidak didiagnosis (*Diagnosis 0*) dibandingkan pria.

Ethnicity:
- Mayoritas pasien yang didiagnosis berasal dari etnis Kaukasia.
- Etnis lain memiliki representasi yang jauh lebih sedikit dalam data ini.

Socioeconomic Status:
- Orang dengan status sosioekonomi 2 tampaknya memiliki distribusi yang lebih seimbang antara didiagnosis atau tidak, sementara status lainnya menunjukkan distribusi yang lebih condong ke salah satu hasil diagnosis.

Education Level:
- Tampaknya terdapat variasi pada tingkat pendidikan yang lebih tinggi, dengan beberapa kelompok menunjukkan lebih banyak kasus diagnosis tertentu.

Smoking:
- Lebih banyak perokok yang tidak didiagnosis (*Diagnosis 0*) dibandingkan yang didiagnosis (*Diagnosis 1*).
- Ini mungkin menunjukkan korelasi negatif antara kebiasaan merokok dan kondisi diagnosis tertentu.

Family History of Diabetes:
- Tampaknya ada perbedaan yang signifikan di mana mereka yang memiliki riwayat keluarga diabetes lebih mungkin didiagnosis dengan kondisi tertentu.

Gestational Diabetes & Polycystic Ovary Syndrome:
- Kedua variabel ini menunjukkan bahwa mereka yang memiliki riwayat gestational diabetes atau sindrom ovarium polikistik cenderung lebih mungkin didiagnosis dengan kondisi tertentu.

Hypertension & Antihypertensive Medications:
- Pasien dengan riwayat hipertensi dan yang menggunakan obat antihipertensi cenderung lebih banyak yang didiagnosis.

Symptoms (Frequent Urination, Excessive Thirst, Blurred Vision, etc.):
- Gejala-gejala ini menunjukkan korelasi yang kuat dengan diagnosis, di mana mereka yang melaporkan gejala-gejala ini cenderung lebih banyak yang didiagnosis dengan kondisi tertentu.

Occupational Exposure to Chemicals:
- Paparan bahan kimia dalam pekerjaan menunjukkan bahwa ada kecenderungan pasien yang terpapar bahan kimia ini lebih banyak didiagnosis.

Dari grafik tersebut, terlihat jelas bahwa variabel-variabel tertentu seperti riwayat keluarga, kondisi medis sebelumnya, dan paparan terhadap faktor risiko tertentu seperti merokok dan paparan bahan kimia memiliki hubungan yang signifikan dengan hasil diagnosis.

#### Numerical Features
![num_dist](https://i.ibb.co.com/VvnMnwQ/num-dist.png)
**Gambar 5.** Analisis Multivariat Numerical Features

#### Correlation Matrix
![corr](https://i.ibb.co.com/d0LFByq/corr-df.png)
**Gambar 6.** Matriks Korelasi
terdapat beberapa fitur yang berkolerasi absolut dengan Diagnosis < 0.001, dan akan dihapus.
| Feature                          | Correlation with Diagnosis |
|----------------------------------|----------------------------|
| EducationLevel                   | -0.002306                  |
| AlcoholConsumption               | -0.009671                  |
| PhysicalActivity                 | -0.006413                  |
| SleepQuality                     | -0.002938                  |
| CholesterolLDL                   | -0.000660                  |
| SlowHealingSores                 |  0.006294                  |
| OccupationalExposureChemicals    | -0.005859                  |
| MedicalCheckupsFrequency         | -0.009598                  |
**Tabel 3.**  Fitur yang berkolerasi absolut dengan Diagnosis < 0.001

Pertimbangan untuk tidak menghapus CholesterolLDL karena berinteraksi dengan fitur lain seperti CholesterolTotal, CholesterolHDL, dan CholesterolTriglycerides.


## Data Preparation
Pada tahap ini akan dilakukan 3 proses data preparation:
- Encoding Fitur Kategori
- Train-test-split
- Normalisasi

### Encoding Fitur Kategori
Teknik yang akan dilakukan adalah teknik one-hot-encoding yang disediakan [Library scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html). Kita akan melakukan proses encoding pada fitur 'Ethnicity' karena fitur ini tidak memiliki urutan yang alami yang berisi value berikut:
0: Caucasian
1: African American
2: Asian
3: Other
setelah proses encoding, maka nantinya akan menghasilkan 4 fitur baru.
```
from sklearn.preprocessing import  OneHotEncoder

df = pd.concat([df, pd.get_dummies(df['Ethnicity'], prefix='Ethnicity')],axis=1)
df.drop(['Ethnicity'], axis=1, inplace=True)
```
### Train-test-split
Membagi dataset menjadi data latih (train) dan data uji (test) merupakan hal yang harus kita lakukan sebelum membuat model. Jika dataset yang kita miliki berukuran sangat kecil, misalnya sekitar 1000  sampel, maka pembagian 80:20 ini cukup ideal. pada proses ini, kita menggunakan fungsi train_test_split dari [sklearn](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html). 
```
from sklearn.model_selection import train_test_split
 
X = df.drop(["Diagnosis"],axis =1)
y = df["Diagnosis"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)
```

### Normalisasi
Teknik yang akan digunakan adalah MinMaxScaler dari [scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html)
teknik ini cocok untuk dataset yang tersebar cukup merata di seluruh rentang skala seperti yang ditunjukkan pada EDA.
```
from sklearn.preprocessing import MinMaxScaler
numerical_features = [] # pilih semua numerical feature
scaler = MinMaxScaler()
scaler.fit(X_train[numerical_features])

#normalisasi pada X_train
X_train[numerical_features] = scaler.transform(X_train.loc[:, numerical_features])

#normalisasi pada X_test
X_test[numerical_features] = scaler.transform(X_test.loc[:, numerical_features])
```

## Modeling
### Lazy Predict Library
LazyPredict adalah pustaka Python yang memudahkan proses pemilihan model machine learning. Ia melakukan ini dengan secara otomatis mengevaluasi dan membandingkan berbagai algoritma pembelajaran mesin pada kumpulan data.
Keuntungan menggunakan LazyPredict:
-  Cepat dan efisien: LazyPredict dapat dengan cepat mengevaluasi dan membandingkan banyak model, menghemat waktu dan tenaga.
- Mempermudah identifikasi model potensial: Alih-alih mencoba berbagai model secara manual, LazyPredict membantu menemukan model yang berpotensi berkinerja baik pada data.
- Cocok untuk analisis awal dan pembuatan prototipe: LazyPredict memudahkan untuk memulai dengan proyek machine learning dengan cepat tanpa terjebak dalam detail pemilihan model.

Contoh penerapan:
```
!pip install lazypredict -q
from lazypredict.Supervised import LazyClassifier
clf = LazyClassifier()
models,predicts = clf.fit(X_train,X_test,y_train,y_test)
print(models.sort_values(by="Accuracy",ascending=False))
```
| Model                        | Accuracy | Balanced Accuracy | ROC AUC | F1 Score |
|------------------------------|----------|-------------------|---------|----------|
| XGBClassifier                 | 0.93     | 0.92              | 0.92    | 0.93     |
| LGBMClassifier                | 0.93     | 0.91              | 0.91    | 0.93     |
| AdaBoostClassifier            | 0.93     | 0.92              | 0.92    | 0.93     |
| RandomForestClassifier        | 0.91     | 0.90              | 0.90    | 0.91     |
| BaggingClassifier             | 0.91     | 0.89              | 0.89    | 0.91     |
| DecisionTreeClassifier        | 0.87     | 0.85              | 0.85    | 0.87     |
| ExtraTreesClassifier          | 0.86     | 0.84              | 0.84    | 0.86     |
| BernoulliNB                   | 0.86     | 0.84              | 0.84    | 0.86     |
| NuSVC                         | 0.85     | 0.83              | 0.83    | 0.85     |
| SVC                           | 0.85     | 0.83              | 0.83    | 0.85     |
| LogisticRegression            | 0.84     | 0.83              | 0.83    | 0.84     |
| LinearSVC                     | 0.84     | 0.82              | 0.82    | 0.84     |
| CalibratedClassifierCV        | 0.84     | 0.82              | 0.82    | 0.84     |
| RidgeClassifier               | 0.83     | 0.82              | 0.82    | 0.83     |
| LinearDiscriminantAnalysis    | 0.83     | 0.82              | 0.82    | 0.83     |
Dari tabel hasil LazyPredict, kita akan memilih 3 model dalam 2 metode terbaik, yaitu metode boosting dan bagging.
Model terpilih:
- XGBClassifier: boosting
- RandomForestClassifier: bagging
- ExtraTreesClassifier: bagging

_XGBoost_ adalah algoritma boosting berbasis pohon keputusan yang sangat populer dan efisien. Algoritma ini menggunakan teknik gradient boosting yang meminimalkan kesalahan dengan cara menambah model-model lemah secara bertahap.
Kelebihan:
- Performa Tinggi: Sangat efisien dalam hal akurasi dan waktu komputasi.
- Skalabilitas: Dapat menangani dataset besar dengan baik.
- Kemampuan Handling Missing Values: Secara otomatis menangani missing values.
- Fleksibilitas: Mendukung berbagai fungsi loss dan hyperparameter tuning.

Kekurangan.
- Kompleksitas: Pengaturan hyperparameter bisa rumit dan membutuhkan waktu.
- Konsumsi Memori: Bisa memerlukan banyak memori, terutama pada dataset besar.

Parameter:
- `use_label_encoder` encoder label bawaan.
- `eval_metric` metrik evaluasi yang digunakan selama pelatihan model.

 _Random Forest_ adalah algoritma machine learning ensemble yang menggabungkan beberapa decision tree untuk meningkatkan akurasi prediksi. Algoritma ini bekerja dengan membuat banyak decision tree secara acak dan kemudian menggunakan voting untuk memprediksi kategori atau nilai data baru. Adapun parameter yang digunakan pada proyek ini adalah:
- `max_depth` kedalaman maksimum.

Keunggulan _Random Forest_ :
- Memiliki akurasi prediksi yang tinggi.
- Mampu menangani dataset dengan dimensi tinggi.
- Tidak sensitif terhadap outlier.

Kerugian _Random Forest_ :
- Cenderung overfit pada dataset kecil. 
- Membutuhkan banyak waktu komputasi untuk pelatihan. 
- Sulit untuk diinterpretasikan.

_Extra Trees Classifier_ adalah algoritma machine learning yang digunakan untuk klasifikasi data. Ini mirip dengan Random Forest Classifier yang terkenal, tetapi memiliki beberapa perbedaan utama yaitu _Random Splitting_ dan _No Bagging_. 

keuntungan _Extra Trees Classifier_ :
- Lebih tahan terhadap overfitting dibandingkan dengan Random Forest, terutama pada kumpulan data berdimensi tinggi.
- Mudah diimplementasikan dan digunakan.
- Memiliki kinerja yang baik pada berbagai masalah klasifikasi.

Kerugian _Extra Trees Classifier_ :
- Cenderung kurang akurat dibandingkan Random Forest pada dataset tertentu.
- Membutuhkan banyak waktu komputasi untuk pelatihan.

Parameter yang digunakan adalah:
- `n_estimators` Jumlah pohon keputusan yang akan dibuat dalam ensemble.
- `random_stat`  pengambilan sampel secara acak.
- `max_depth` Kedalaman maksimum pohon keputusan individual.
- `n_jobs` mempercepat pelatihan pada sistem dengan beberapa core CPU.

## Evaluation

_ROC AUC (Receiver Operating Characteristic - Area Under the Curve)_
- **Penjelasan**: ROC AUC mengukur kemampuan model untuk membedakan antara kelas positif (penderita diabetes) dan negatif (tidak menderita diabetes) di berbagai threshold keputusan. Kurva ROC adalah plot dari True Positive Rate (TPR) terhadap False Positive Rate (FPR), dan AUC (Area Under the Curve) memberikan nilai antara 0 dan 1. Semakin dekat nilai AUC ke 1, semakin baik kinerja model.
- **Formula**: AUC dihitung sebagai area di bawah kurva ROC. 
 $$\text{AUC} = \sum_{i=1}^{n-1} \frac{(x_{i+1} - x_i) \times (y_{i+1} + y_i)}{2}$$
- **Cara Kerja**: Metrik ini membantu menilai seberapa baik model dalam memisahkan kelas positif dan negatif, memberikan gambaran yang menyeluruh tentang performa model di berbagai threshold.



_KS Statistic (Kolmogorov-Smirnov Statistic)_
- **Penjelasan**: KS Statistic mengukur jarak maksimum antara distribusi kumulatif prediksi untuk kelas positif dan negatif. Ini membantu untuk menilai sejauh mana model dapat membedakan antara dua kelas tersebut.
- **Formula**: KS Statistic adalah selisih maksimum antara dua fungsi distribusi kumulatif (CDF) dari kelas positif dan negatif. 
$$\text{KS Statistic} = \max |F_1(x) - F_0(x)|$$
- **Cara Kerja**: Semakin besar nilai KS, semakin baik kemampuan model dalam memisahkan prediksi antara kelas positif dan negatif. Nilai KS di atas 0,4 umumnya dianggap menunjukkan performa yang baik.

_Confusion Matrix_
- Confusion matrix adalah tabel yang merangkum prediksi model terhadap data uji dalam bentuk True Positive (TP), True Negative (TN), False Positive (FP), dan False Negative (FN). Dari confusion matrix ini, berbagai metrik seperti akurasi, precision, recall, dan F1 score dapat dihitung, fungsi ini diambil dari [scikit-learn](https://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html).
- **Cara Kerja**: Confusion matrix memberikan gambaran yang jelas tentang bagaimana model melakukan klasifikasi, termasuk kesalahan yang dibuat oleh model (FP dan FN), serta keberhasilan model (TP dan TN). Ini sangat penting untuk memahami area di mana model perlu ditingkatkan.

- Confusion matrix menghitung beberapa metrik kunci:
  - $$\text{Accuracy} = \frac{\text{TP + TN}}{\text{TN + TP + FN + FP}} \times 100\%$$
  - $$\text{Precision} = \frac{\text{TP}}{\text{TP + FP}} \times 100\%$$
  - $$\text{Recall} = \frac{\text{TP}}{\text{TP + FN}} \times 100\%$$
  - $$\text{F1 Score} = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$$

*Penjelasan*
- TP (True Positive): Jumlah data positif yang diprediksi dengan benar sebagai positif.
- TN (True Negative): Jumlah data negatif yang diprediksi dengan benar sebagai negatif.
- FP (False Positive): Jumlah data negatif yang diprediksi secara tidak benar sebagai positif (Kesalahan Tipe I).
- FN (False Negative): Jumlah data positif yang diprediksi secara tidak benar sebagai negatif (Kesalahan Tipe II).

Berikut hasil ROC AUC dan KS 3 buah model yang dilatih:
|              | XGBClassifier | RandomForestClassifier | ExtraTreesClassifier |
|--------------|---------------|------------------------|----------------------|
| **ROC AUC**  | 0.96          | 0.95                   | 0.95                 |
| **KS**       | 0.88          | 0.84                   | 0.81                 |
**Tabel 4.**Perbandingan ROC AUC dan KS Model

Classification report XGBoost Classifier : 
|               | precision | recall | f1-score | support |
|---------------|-----------|--------|----------|---------|
| **0**         | 0.92      | 0.97   | 0.95     | 233     |
| **1**         | 0.95      | 0.87   | 0.91     | 143     |
| **accuracy**  |           |        | 0.93     | 376     |
| **macro avg** | 0.93      | 0.92   | 0.93     | 376     |
| **weighted avg** | 0.93   | 0.93   | 0.93     | 376     |
**Tabel 5a.** XGBoost Classifier
![xgb](https://i.ibb.co.com/82fRrj4/xgb.png)

Classification report RandomForestClassifier : 
|               | precision | recall | f1-score | support |
|---------------|-----------|--------|----------|---------|
| **0**         | 0.90      | 0.97   | 0.94     | 233     |
| **1**         | 0.95      | 0.83   | 0.89     | 143     |
| **accuracy**  |           |        | 0.92     | 376     |
| **macro avg** | 0.93      | 0.90   | 0.91     | 376     |
| **weighted avg** | 0.92  | 0.92   | 0.92     | 376      |
**Tabel 5b.** RandomForestClassifier
![rf](https://i.ibb.co.com/jGzY0fK/rf.png)

Classification report ExtraTreesClassifier : 
|               | precision | recall | f1-score | support |
|---------------|-----------|--------|----------|---------|
| **0**         | 0.86      | 0.96   | 0.90     | 233     |
| **1**         | 0.91      | 0.74   | 0.82     | 143     |
| **accuracy**  |           |        | 0.88     | 376     |
| **macro avg** | 0.89      | 0.85   | 0.86     | 376     |
| **weighted avg** | 0.88   | 0.88   | 0.87     | 376     |
**Tabel 5c.** ExtraTreesClassifier
![etc](https://i.ibb.co.com/zXNkV6P/etc.png)

Berdasarkan evaluasi yang dilakukan:

- XGBClassifier menunjukkan performa terbaik dengan nilai ROC AUC sebesar 0.96 dan KS sebesar 0.88. Model ini juga memiliki precision, recall, dan f1-score yang konsisten tinggi pada kedua kelas, serta akurasi keseluruhan 0.93.
- RandomForestClassifier memiliki performa yang sangat baik dengan ROC AUC sebesar 0.95 dan KS sebesar 0.84. Namun, f1-score pada kelas positif sedikit lebih rendah dibandingkan XGBClassifier, dengan akurasi keseluruhan 0.92.
- ExtraTreesClassifier menunjukkan performa yang baik dengan ROC AUC sebesar 0.95 dan KS sebesar 0.81, tetapi memiliki nilai recall dan f1-score yang lebih rendah pada kelas positif, dengan akurasi keseluruhan 0.88.

Model Terbaik:
XGBClassifier dipilih sebagai model terbaik berdasarkan performa keseluruhan pada ROC AUC, KS Statistic, dan classification report. Model ini memberikan keseimbangan terbaik antara akurasi, precision, recall, dan f1-score. Diharapkan dengan model yang telah dikembangan dapat memprediksi diagnosa diabetest dengan baik menggunakan XGBClassifier. Alasan mengapa metode XGBClassifier yang dipilih karena XGBClassifier adalah algoritma yang terbaik berdasarkan performa metrik. Parameter yang digunakan pada pemodelan ini juga sederhana, sehingga lebih mudah untuk digunakan.

### Top 10 Fitur pada Model XGB
![top 10](https://i.ibb.co.com/8dQCG45/top10.png)
Diharapkan insight ini dapat membantu dokter atau ahli kesehatan untuk fokus pada fitur-fitur utama yang diidentifikasi oleh model ketika membuat keputusan klinis, terutama dalam konteks skrining atau diagnosis diabetes.
