#!/bin/bash
# Block until the policy server is listening on :8000, or report failure from its log.
for i in $(seq 1 360); do
  if (exec 3<>/dev/tcp/127.0.0.1/8000) 2>/dev/null; then echo SERVER_READY; exit 0; fi
  if grep -q -E "Traceback|Error|Killed|RESOURCE_EXHAUSTED" ~/spais-demo/server.log; then
    echo SERVER_FAILED; tail -20 ~/spais-demo/server.log; exit 1
  fi
  sleep 5
done
echo SERVER_TIMEOUT; tail -20 ~/spais-demo/server.log; exit 1
