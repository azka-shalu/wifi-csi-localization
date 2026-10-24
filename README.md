# Indoor Localization berbasis WiFi CSI (CNN, CNN-LSTM, dan Extra Trees)

Proyek ini mengestimasi posisi (x, y) di dalam ruangan dari data **Channel State Information (CSI)** WiFi. Seluruh pipeline, mulai dari preprocessing, pelatihan model, hingga audit kebocoran data, ada di satu notebook: [`PML.ipynb`](PML.ipynb).

## Dataset

- 90 lokasi pada grid 2D (koordinat bilangan bulat), masing-masing direkam oleh **3 antena**, sehingga ada 270 file CSV (`antenna_{antena}_{x}_{y}.csv`).
- Setiap file berisi 60 time step × 30 subcarrier untuk **amplitudo** dan **fase**, plus kolom `timestamp` (~400 Hz).
- Tensor input per lokasi: `(6, 60, 30)` = 3 antena × (amplitudo, fase). Target: `(x, y)`.
- Dataset (`dataset.zip`) **tidak disertakan** di repo ini. Letakkan sendiri di folder kerja sebelum menjalankan notebook.

## Alur Kerja

1. **Pemuatan data**: ekstrak zip, gabungkan 3 antena per lokasi menjadi `(90, 6, 60, 30)`.
2. **Preprocessing**
   - Phase unwrapping dan sanitization (hapus tren linear antar subcarrier).
   - Low-pass Butterworth (cutoff 20 Hz) dengan `sosfiltfilt`.
   - Pembersihan nilai `inf`/`NaN` pada CSI, lalu normalisasi per kanal **hanya dari data latih**.
3. **Model**
   - `CSI_CNN`: CNN 2D + regressor.
   - `CSI_CNN_LSTM`: CNN + LSTM.
   - KNN dan Extra Trees di atas fitur statistik (mean, std, selisih temporal) per kanal dan subcarrier.
4. **Evaluasi**: ablasi antena dan fitur (amplitudo vs fase), uji beberapa seed, spatial holdout, cross-validation, nested CV, dan audit kebocoran data.

Metrik: *Mean Distance Error* (MDE, jarak Euclidean rata-rata antara prediksi dan posisi asli, dalam satuan grid), median error, dan RMSE posisi.

## Hasil Utama

| Model / Eksperimen | MDE |
|---|---|
| Baseline (rata-rata posisi latih) | 3.28 |
| CNN | 3.63 |
| CNN-LSTM | 3.34 |
| KNN (K=11) | 3.48 |
| Extra Trees, semua fitur (test, 14 lokasi baru) | **3.12** |
| Baseline pada test set yang sama | 3.30 |

Ablasi antena (validasi, CNN-LSTM): kombinasi tiga antena paling baik (MDE 3.36), dibandingkan satu antena saja (3.80 sampai 3.91).

## Catatan Jujur tentang Hasil

- Dataset sangat kecil (90 lokasi, tiap lokasi satu sampel), dan test set hanya 14 lokasi yang belum pernah dilihat model.
- Selisih model terhadap baseline kecil. Pada test akhir, Extra Trees unggul sekitar 5.5% (MDE 3.12 vs 3.30), tetapi interval bootstrap 95% untuk selisihnya (-0.39 sampai 0.77) mencakup nol, jadi **peningkatan ini belum signifikan secara statistik**.
- Pada spatial holdout dengan 3 seed, CNN-LSTM hanya mengalahkan baseline pada 1 dari 3 seed.
- Pemilihan fitur per sumbu (axis-specific) tampak membantu di awal, tetapi pada cross-validation yang memperhatikan kebocoran data ia **tidak lebih baik** (hanya 4 dari 15 fold lebih baik). Karena itu model akhir memakai semua fitur.
- Korelasi antara jarak CSI dan jarak fisik ada tetapi lemah (Spearman sekitar 0.23 sampai 0.30).

Kesimpulannya, CSI pada data ini mengandung sinyal spasial, tetapi data yang tersedia belum cukup untuk model yang andal. Menambah jumlah sampel per lokasi dan lokasi itu sendiri kemungkinan lebih berpengaruh daripada mengubah arsitektur.

## Menjalankan

Notebook ditulis untuk Google Colab (memakai `google.colab.files` dan, opsional, Google Drive). Untuk lokal:

```bash
pip install -r requirements.txt
jupyter notebook PML.ipynb
```

Ubah bagian upload/ekstraksi dataset di sel-sel awal agar menunjuk ke `dataset.zip` lokal, dan lewati sel yang memakai `google.colab`. GPU disarankan untuk CNN dan CNN-LSTM tetapi tidak wajib.

## Struktur Repo

```
.
├── PML.ipynb
├── README.md
├── requirements.txt
└── .gitignore
```
