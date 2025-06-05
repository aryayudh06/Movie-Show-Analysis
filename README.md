# 🎬 Movie Show Analysis with Apache Airflow

Project ini bertujuan untuk menganalisis data acara film dan streaming menggunakan pipeline **ETL (Extract, Transform, Load)** yang dijalankan melalui **Apache Airflow**. Pipeline ini mengambil data dari sumber, melakukan transformasi, dan menyimpannya untuk analisis lebih lanjut.

## 🛠️ Teknologi yang Digunakan

- Python
- Apache Airflow
- Docker & Docker Compose
- Pandas
- Jupyter Notebook (untuk eksplorasi data)

## 🚀 Cara Menjalankan Project

Pastikan Anda sudah menginstall [Docker](https://www.docker.com/) dan [Docker Compose](https://docs.docker.com/compose/).

### Langkah-langkah:

1. Jalankan Airflow menggunakan Docker Compose:
   ```bash
   docker compose up

2. Buka Airflow UI di browser:
   [text](http://localhost:8080)

3. Login dengan default user (Cek di konfigurasi compose), lalu jalankan DAG bernama:
   streaming_elt_pipeline

4. Setelah selesai, hentikan layanan dengan:
   ```bash
   docker compose down
   ```
   atau
   ```bash
   docker stop <container-name>

Project Directory
```bash
.
├── dags/                   # Folder DAG untuk Airflow
│   └── ELT Scripts
├── docker-compose.yml      # Konfigurasi Docker Compose untuk Airflow
└── README.md

