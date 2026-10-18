#!/bin/bash
# sends some requests to the API so the Grafana dashboard has data
# usage: ./send_requests.sh [url]   (default http://localhost:8000)
URL=${1:-http://localhost:8000}

for i in $(seq 1 30); do
  curl -s -X POST "$URL/predict" -H "Content-Type: application/json" \
    -d @sample_input.json > /dev/null
  # a sick-looking patient too
  curl -s -X POST "$URL/predict" -H "Content-Type: application/json" \
    -d '{"age":67,"sex":1,"cp":4,"trestbps":160,"chol":286,"fbs":0,"restecg":2,"thalach":108,"exang":1,"oldpeak":1.5,"slope":2,"ca":3,"thal":3}' > /dev/null
  curl -s "$URL/health" > /dev/null
  sleep 0.5
done
# one bad request (missing fields) to show errors
curl -s -X POST "$URL/predict" -H "Content-Type: application/json" -d '{"age":50}' > /dev/null
echo "done"
