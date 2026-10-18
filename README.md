# Heart Disease Prediction - MLOps Assignment 1

MLOps (AIMLCZG523) Assignment 1 - BITS Pilani

Predicts whether a patient has heart disease (1) or not (0) using the UCI Heart Disease dataset, and serves the model as an API with Docker, Kubernetes (minikube) and Prometheus/Grafana monitoring.

For quick startup commands, service links, and Docker instructions, see [RUN_GUIDE.md](RUN_GUIDE.md).

## Project structure

```
heart-disease-mlops/
├── data/
│   ├── download_data.py      # downloads from UCI (ucimlrepo) + cleans
│   ├── heart_raw.csv
│   └── heart_clean.csv       # cleaned dataset used for training
├── notebooks/eda.ipynb       # EDA with plots
├── src/
│   ├── preprocess.py         # feature lists + sklearn ColumnTransformer
│   ├── eda.py                # saves EDA plots to reports/figures
│   ├── train.py              # trains LogReg + RandomForest, logs to MLflow
│   └── predict.py            # simple inference script
├── app/main.py               # FastAPI app (/predict, /health, /metrics)
├── models/                   # saved best model (model.pkl) + metrics.json
├── tests/                    # pytest unit tests
├── k8s/                      # Kubernetes deployment + service
├── monitoring/               # Prometheus config + Grafana dashboard
├── .github/workflows/ci.yml  # GitHub Actions pipeline
├── Dockerfile
├── docker-compose.yml        # API + Prometheus + Grafana
├── requirements.txt          # everything (training, tests)
├── requirements-api.txt      # only what the Docker image needs
├── screenshots/              # screenshots for the report
└── reports/                  # report + figures
```

## How to run

### 1. Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Data + EDA

```bash
python data/download_data.py
python src/eda.py
```

The notebook `notebooks/eda.ipynb` has the same EDA with comments (`pip install jupyter` to open it).

### 3. Train + MLflow

```bash
python src/train.py
mlflow ui --port 5001        # open http://localhost:5001
```

(port 5000 is used by AirPlay on Mac, so I use 5001)

### 4. Tests and lint

```bash
flake8 .
pytest -v
```

### 5. Run the API locally

```bash
uvicorn app.main:app --reload --port 8000
# open http://localhost:8000/docs
curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d @sample_input.json
```

### 6. Docker

```bash
docker build -t heart-disease-api:v1 .
docker run -d --name heart-api -p 8000:8000 heart-disease-api:v1
curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d @sample_input.json
```

Response:

```json
{ "prediction": 0, "label": "no heart disease", "confidence": 0.8254 }
```

### 7. Kubernetes (minikube)

```bash
minikube start --driver=docker
minikube image load heart-disease-api:v1
kubectl apply -f k8s/
kubectl get pods,svc
minikube service heart-api-service --url     # gives a URL like http://127.0.0.1:xxxxx
```

or run `minikube tunnel` in another terminal so the LoadBalancer gets an external IP.

### 8. Monitoring (Prometheus + Grafana)

```bash
docker compose up -d --build
./send_requests.sh
```

- Prometheus: http://localhost:9090
- Grafana: http://localhost:3300, dashboard "Heart Disease API" (login admin/admin if asked)

## Results

| Model                              | CV ROC-AUC | Test accuracy | Test precision | Test recall | Test ROC-AUC |
| ---------------------------------- | ---------- | ------------- | -------------- | ----------- | ------------ |
| Logistic Regression (C=1)          | 0.904      | 0.885         | 0.839          | 0.929       | **0.968**    |
| Random Forest (200 trees, depth 4) | 0.902      | 0.869         | 0.857          | 0.857       | 0.949        |

Logistic Regression was selected as the final model.

## CI/CD

GitHub Actions (`.github/workflows/ci.yml`) runs on every push to main:

1. flake8 lint
2. train models
3. pytest
4. upload model, plots, test results and mlruns as artifacts
5. build Docker image and test `/health` + `/predict` inside the container

The pipeline fails if lint, training, any test or the container check fails.
