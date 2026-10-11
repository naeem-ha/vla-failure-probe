#!/bin/bash
nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader
ps -eo pid,etime,pcpu,rss,cmd --sort=-pcpu | grep -E "eval.py|serve_policy|kit" | grep -v grep | head -5 | cut -c1-160
ls -la ~/spais-demo/work/runs/polaris_foodbussing_test 2>/dev/null
log=$(ls -t ~/.nvidia-omniverse/logs/Kit/*/*/*.log ~/spais-demo/polaris/.venv/lib/python3.11/site-packages/isaacsim/kit/logs/Kit/*/*/*.log 2>/dev/null | head -1)
echo "LOG: $log"
[ -n "$log" ] && grep -E "\[Error\]|Fatal|out of memory|OOM" "$log" | tail -8 && tail -4 "$log" | cut -c1-200
