#!/bin/bash
# Run a probe definition over N episodes per variant, ONE ISAAC PROCESS PER EPISODE (a PhysX blow-up
# or hang in one episode can't take the rest down), then write a CSV summary.
# Usage (inside WSL):
#   bash ~/spais-demo/work/polaris_wsl/probes/run_probe_set.sh <probe.yaml> <episodes> "<variant> [variant...]" [extra run args...]
#   bash run_probe_set.sh food_bussing_battery.yaml 5 "stock hazard_bowl"
# Episode k uses initial condition IC_START+k and seed k. Env vars: IC_START (0), EP_TIMEOUT (s, 420), TAG.
# Uses the WSL env (wsl_env.make_wsl_env) and PolaRiS-Hub-wsl. Needs the policy server on :8000.
probe=$1; n=${2:-5}; variants=${3:-stock}
shift 3 2>/dev/null || shift $#
here=$(cd "$(dirname "$0")" && pwd)
case "$probe" in
  /*) ;;
  *) probe="$here/$probe" ;;
esac
source ~/spais-demo/work/polaris_env.sh
export POLARIS_DATA_PATH=${POLARIS_DATA_PATH_WSL:-$HOME/spais-demo/polaris/PolaRiS-Hub-wsl}
export PATH=$HOME/spais-demo/polaris/.venv/bin:$PATH   # ninja for the splat JIT
mkdir -p ~/spais-demo/logs
pname=$(python3 -c "import yaml,sys; print(yaml.safe_load(open(sys.argv[1]))['name'])" "$probe" 2>/dev/null || basename "$probe" .yaml)
out=~/spais-demo/work/runs/polaris
tag=${TAG:-set}
jsons=(); expected=()
for v in $variants; do
  for k in $(seq 0 $((n - 1))); do
    ic=$(( ${IC_START:-0} + k ))
    name="${pname}_${v}_ic${ic}_s${k}_${tag}"
    log=~/spais-demo/logs/probe_${name}.log
    timeout -k 10 "${EP_TIMEOUT:-420}" uv run python "$here/../probe_polaris.py" run --probe "$probe" \
      --variant "$v" --ic "$ic" --seed "$k" --tag "$tag" "$@" > "$log" 2>&1
    rc=$?
    pkill -9 -f "[p]robe_polaris.py run --probe $probe --variant $v --ic $ic " 2>/dev/null
    jsons+=("$out/$name/$name.json"); expected+=("$v:$ic:$k")
    if [ -f "$out/$name/${name}_summary.txt" ]; then
      echo "[$v ic$ic] rc=$rc $(grep -E '^(unsafe|first_violation_step|rubric_max_progress|steps):' "$out/$name/${name}_summary.txt" | tr '\n' ' ')"
    else
      echo "[$v ic$ic] rc=$rc NO RESULT (crash/timeout) log=$log"
    fi
  done
done
csv="$out/${pname}_${tag}_summary.csv"
uv run python "$here/../probe_polaris.py" aggregate --out-csv "$csv" --jsons "${jsons[@]}" --expected "${expected[@]}" 2>/dev/null | grep -v dev-dependencies
echo "CSV: $csv"
echo PROBE_SET_DONE
