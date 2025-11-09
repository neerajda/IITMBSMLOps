#!/bin/bash
SERVICE_IP=$(kubectl get svc iris-service -o jsonpath='{.status.loadBalancer.ingress[0].ip}')
echo "Testing endpoint: http://$SERVICE_IP/predict"

wrk -t4 -c1000 -d30s -s <(cat <<'LUA'
wrk.method = "POST"
wrk.body   = '{"sepal_length":5.1,"sepal_width":3.5,"petal_length":1.4,"petal_width":0.2}'
wrk.headers["Content-Type"] = "application/json"
LUA
) http://$SERVICE_IP/predict
