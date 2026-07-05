# Workflow-CI

Submission Proyek Akhir kelas **Membangun Sistem Machine Learning** (Dicoding) — Kriteria 3: Workflow CI.

Re-training otomatis model klasifikasi arah harga pangan Jawa Timur melalui **MLflow Project**
yang dijalankan **GitHub Actions** pada setiap push ke `main` (atau manual via *workflow_dispatch*).

## Struktur

```
├── .github/workflows/ci.yml   # workflow CI: train -> artefak -> Docker Hub
└── MLProject/
    ├── MLproject               # definisi MLflow Project (entry point + parameter)
    ├── conda.yaml              # environment (python 3.12.7, mlflow 2.19.0)
    ├── modelling.py            # skrip re-training
    └── harga_pangan_jatim_preprocessing/   # dataset siap latih
```

## Alur CI

1. Checkout repository & set up Python 3.12.7
2. Install dependencies (mlflow==2.19.0)
3. `mlflow run MLProject` — re-training model
4. Ambil `run_id` terakhir dari file store `mlruns/`
5. Upload artefak model ke GitHub (Actions artifact)
6. `mlflow models build-docker` — bangun Docker image dari model
7. Login, tag, dan push image ke Docker Hub

## Secrets yang dibutuhkan

| Secret | Isi |
|---|---|
| `DOCKER_USERNAME` | username Docker Hub |
| `DOCKER_PASSWORD` | access token Docker Hub |

## Docker Image

Model tersedia sebagai image publik di Docker Hub dan dapat dijalankan dengan:

```bash
docker run -p 5005:8080 <docker-username>/harga-pangan-jatim:latest
```
