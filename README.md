# Experiment Tracking Project

Proyek ini bertujuan untuk membandingkan dan menerapkan tiga tools utama experiment tracking, yaitu **MLFlow**, **TensorBoard**, dan **DVC** menggunakan model **Random Forest** dan **Neural Network** pada dataset `adult.csv`.

---

## Struktur Folder

```
.
├── artifacts                   # Menyimpan artifact tambahan
├── data
│   ├── processed               # Data setelah preprocessing
│   └── raw                     # Data mentah
├── logs                        # Log TensorBoard
├── mlruns                      # Hasil tracking MLFlow
├── models                      # File model tersimpan
├── src
│   ├── evaluate.py             # Evaluasi model
│   ├── preprocess.py           # Preprocessing data
│   └── train.py                # Script training
├── conda.yaml                  # File environment Conda
├── dvc.lock                    # Lock file dari DVC
├── dvc.yaml                    # Pipeline DVC
├── metrics.csv                 # Metrics akhir model dalam bentuk CSV
├── metrics.json                # Metrics akhir model dalam bentuk JSON
├── metrics_nn.json             # Metrics khusus Neural Network
├── metrics_rf.json             # Metrics khusus Random Forest
├── params.yaml                 # Parameter training
├── requirements.txt            # Dependency Python
└── README.md                   # Dokumentasi
```

---

## Prasyarat

- Python 3.8+
- Conda (opsional, untuk manajemen environment)
- Git

---

## Instalasi

### Clone Repository

```bash
git clone https://github.com/xmriz/Tugas-IF5251-Experiment-Tracking.git
cd Tugas-IF5251-Experiment-Tracking
```

### Setup Environment

Menggunakan conda:

```bash
conda create -n experiment-tracking python=3.10
conda activate experiment-tracking
pip install -r requirements.txt
```

### Inisialisasi DVC

Jika menggunakan remote lokal:

```bash
dvc remote add -d myremote <lokasi_remote_anda>
dvc push
```

---

## Menjalankan Project

### 1. Preprocessing Data

```bash
python src/preprocess.py
```

Data hasil preprocessing disimpan dalam `data/processed/`.

### 2. Training Model

Menggunakan DVC:

```bash
dvc repro
```

Perintah ini akan menjalankan seluruh tahapan berikut:

- preprocess data
- training Random Forest
- training Neural Network
- evaluasi kedua model

### 3. Evaluasi Model

Evaluasi akan otomatis dijalankan pada perintah sebelumnya dan menghasilkan file `metrics.csv` dan `metrics.json`.

Untuk menampilkan metrics:

```bash
dvc metrics show
dvc plots show
```

---

## Menjalankan Project Menggunakan PowerShell

Jika kamu menggunakan **PowerShell** di Windows, pastikan untuk mengizinkan eksekusi skrip terlebih dahulu dengan perintah berikut:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

Kemudian, jalankan skrip berikut untuk menjalankan seluruh tahapan dari awal hingga selesai secara bersih:

```powershell
.\run_all.ps1 -Clean
```

---

## Visualisasi Experiment Tracking

### MLFlow

Menjalankan MLFlow UI:

```bash
mlflow ui
```

Akses di browser: [http://localhost:5000](http://localhost:5000)

### TensorBoard

Menjalankan TensorBoard:

```bash
tensorboard --logdir=logs
```

Akses di browser: [http://localhost:6006](http://localhost:6006)

### DVC

Menampilkan plot DVC:

```bash
dvc plots show metrics.csv --template simple --x model --y accuracy
```

---

## Menjalankan Ulang Semua Eksperimen

Jika ingin mengulang dari awal secara paksa:

```bash
dvc repro --force
```

---

## Tracking Metrics

- **MLFlow**: Tracking metrics, parameter, dan artifact secara otomatis dalam folder `mlruns/`.
- **TensorBoard**: Metrics akurasi dan loss tersimpan di folder `logs/`.
- **DVC**: Metrics akhir tersimpan dalam file CSV dan JSON (`metrics.csv`, `metrics.json`).

---

## Membersihkan Eksperimen

Jika ingin membersihkan hasil eksperimen:

```bash
dvc destroy -f
rm -rf logs/ models/ mlruns/ data/processed/
```

---

## Troubleshooting

Jika mengalami kendala:

- Pastikan versi DVC, MLFlow, dan TensorBoard sesuai dengan requirements.
- Periksa log atau pesan error dari tiap command yang dijalankan.
- Gunakan dokumentasi resmi dari [DVC](https://dvc.org/doc), [MLFlow](https://mlflow.org/docs/latest/index.html), dan [TensorBoard](https://www.tensorflow.org/tensorboard).

---
