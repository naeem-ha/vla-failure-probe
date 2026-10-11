#!/bin/bash
source ~/spais-demo/work/client_env.sh
find "$VIRTUAL_ENV/lib/python3.8/site-packages" -name '*.so*' -type f -print0 | xargs -0 python clear_execstack.py
timeout 300 python render_test.py 2>&1 | grep -v -i warn | tail -8
