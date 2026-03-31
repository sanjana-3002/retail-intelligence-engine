#!/bin/bash
cd "$(dirname "$0")"
echo "Starting Retail Intelligence Dashboard..."
open http://localhost:8080/dashboard/retail_intelligence_dashboard.html &
python3 -m http.server 8080
