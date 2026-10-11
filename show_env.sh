#!/bin/bash
f=~/spais-demo/polaris/src/polaris/environments/manager_based_rl_splat_environment.py
grep -n -E "def |sensors\[|rgb|depth|composit|semantic|render\(" "$f" | head -60
