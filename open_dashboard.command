#!/bin/bash
cd "$(dirname "$0")"
echo "Starting Retail Intelligence Dashboard..."
python3 -m http.server 8080 &
SERVER_PID=$!
sleep 1
open http://localhost:8080/dashboard/retail_intelligence_dashboard.html
wait $SERVER_PID
