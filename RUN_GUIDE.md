# Step-by-Step Run Guide

Follow these steps from the project root (`heart-disease-mlops`). Run long-lived servers in their own terminal and use a second terminal for commands that need the server. A foreground server stops with `Ctrl+C`.

**GitHub project:** [MLOps-cycle-assignment](https://github.com/2025ae05649-source/MLOps-cycle-assignment)  
**GitHub Actions run:** [Open the passing CI run and its artifacts](https://github.com/2025ae05649-source/MLOps-cycle-assignment/actions/runs/39061488365)

## Before You Start

Open Terminal and move into the project folder. Use your own path if the project is somewhere else:

```bash
cd "/Users/jayakrishna/Downloads/ML-OPS assignement  - latest/heart-disease-mlops"
source .venv/bin/activate
```

The existing `.venv` contains the project dependencies. If it is missing, create it and install the dependencies:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 1. Download the Data

With the environment active and the current directory at the project root, run:

```bash
python data/download_data.py
```

The script downloads the UCI Heart Disease dataset, fills missing `ca` and `thal` values, converts the target to 0/1, and writes:

- `data/heart_raw.csv`: downloaded data
- `data/heart_clean.csv`: cleaned data used for analysis and training

Expected output is about 303 rows and 14 columns, with target counts close to `0: 164` and `1: 139`. The exact output is printed by the script. If the download fails with an SSL certificate error on macOS, make sure `.venv` is active; the script uses `certifi` for certificates.

## 2. Run the Charts

Run the EDA script:

```bash
python src/eda.py
```

It prints the dataset summary and saves charts in `reports/figures/`:

- `class_balance.png`
- `histograms.png`
- `correlation_heatmap.png`
- `age_vs_thalach.png`
- `cp_vs_target.png`

Open `reports/figures/` in the VS Code Explorer and click a PNG to view it. The charts show class balance, feature distributions and correlations, and relationships between patient features and the target.

## 3. Train Models, Then Review MLflow

Start training from the project root:

```bash
python src/train.py
```

Training tunes Logistic Regression and Random Forest using 5-fold cross-validation, prints the best parameters and CV/test metrics, logs both model runs to MLflow, and saves the best model to `models/model.pkl`. It also writes `models/metrics.json` and confusion-matrix/ROC plots under `reports/figures/`. Training exits back to the shell prompt when complete.

After training finishes, start MLflow in the same project folder:

```bash
mlflow ui --port 5001
```

Open the [MLflow UI](http://localhost:5001). In the UI:

1. Select the **heart-disease** experiment.
2. Review the Logistic Regression and Random Forest runs in the run list.
3. Open a run to inspect parameters, CV/test metrics, and the `plots` and `model` artifacts.
4. Select two runs and use **Compare** to compare them.

Port `5001` avoids the commonly occupied macOS port `5000`. When finished reviewing MLflow, return to its terminal and press `Ctrl+C`.

The recorded run selected Logistic Regression (test ROC-AUC about `0.968`); small differences can occur if the data or environment changes.

## 4. Run a Prediction, Test the Project, and Open Docker

Run the saved-model prediction example:

```bash
python src/predict.py
```

It prints the predicted class and confidence, then exits to the shell prompt. A typical result is `Prediction: 0 (No heart disease), confidence: 0.83`.

Run lint and tests from the project root:

```bash
flake8 .
pytest -v
```

No `flake8` output means lint passed. The expected test result is **13 passed**. When pytest completes, it returns to the prompt; there is no test server to leave running.

Before the Docker steps, open **Docker Desktop** and wait until its status shows that Docker is running. You can verify the engine from Terminal with:

```bash
docker info
```

## 5. Try the API Locally, Then Run It in Docker

### 5a. Run without Docker

In a terminal with `.venv` active, start FastAPI:

```bash
uvicorn app.main:app --reload --port 8000
```

Open [FastAPI interactive docs](http://localhost:8000/docs). Expand **POST /predict**, choose **Try it out**, then **Execute**. You can also send the included example from a second terminal:

```bash
curl http://localhost:8000/health
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d @sample_input.json
```

Expected prediction response includes `prediction`, `label`, and `confidence`. Stop the local API with `Ctrl+C` before starting its Docker container, because both use port `8000`.

### 5b. Build and run the API container

Build the image and start the container:

```bash
docker build -t heart-disease-api:v1 .
docker run -d --name heart-api -p 8000:8000 heart-disease-api:v1
```

Check it and make a prediction:

```bash
curl http://localhost:8000/health
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d @sample_input.json
docker logs heart-api
```

Open [Docker API docs](http://localhost:8000/docs) to try the endpoint in the browser. When done, stop and remove the container:

```bash
docker rm -f heart-api
```

## 6. Deploy with Kubernetes (minikube)

Docker Desktop must be running. Start minikube, build the image if you have not already built it in step 5, load it into minikube, and apply the deployment and service:

```bash
minikube start --driver=docker
docker build -t heart-disease-api:v1 .
minikube image load heart-disease-api:v1
kubectl apply -f k8s/
kubectl get pods,svc
```

Wait until both `heart-api` pods show `1/1 Running`. Then expose the service locally:

```bash
minikube service heart-api-service --url
```

This command prints a local URL, for example `http://127.0.0.1:50083`, and stays running. In a second terminal, replace the sample URL below with the one it printed:

```bash
curl http://127.0.0.1:50083/health
curl -X POST http://127.0.0.1:50083/predict \
  -H "Content-Type: application/json" \
  -d @sample_input.json
kubectl logs -l app=heart-api --tail=5
```

For browser docs, append `/docs` to the printed URL. When finished, stop the `minikube service` command with `Ctrl+C`, then remove the deployment and stop minikube:

```bash
kubectl delete -f k8s/
minikube stop
```

## 7. Start Monitoring with Grafana and Prometheus

Make sure the API-only container from step 5 is removed (`docker rm -f heart-api`) and the local API from step 5 is stopped. From the project root, start the API, Prometheus, and Grafana together:

```bash
docker compose up -d --build
```

Open these pages:

- [FastAPI docs](http://localhost:8000/docs)
- [API health check](http://localhost:8000/health)
- [Prometheus](http://localhost:9090)
- [Grafana](http://localhost:3300)

In Prometheus, open **Status → Targets** and confirm the `heart-api` target is **UP**. The raw API metrics are at [http://localhost:8000/metrics/](http://localhost:8000/metrics/).

In Grafana, open the **Heart Disease API** dashboard. Anonymous viewing is enabled; if a login prompt appears, try `admin` / `admin`. To put requests into the dashboard, run this from another terminal in the project root:

```bash
./send_requests.sh
```

It sends sample predictions and a health request to the API at port `8000`. To check container status/logs or stop the stack:

```bash
docker compose ps
docker compose logs -f
docker compose down
```

`docker compose down` stops the API, Prometheus, and Grafana containers.
