#!/usr/bin/env bash
# Chain Kaggle kernels, each starting when the previous one has finished:
#   ablation -> detect -> optuna -> boost -> v1
# Usage: bash scripts/kaggle_queue.sh [stage]   start at a stage whose kernel is already running/finished
# (that stage is only waited for, not pushed again), e.g. `bash scripts/kaggle_queue.sh detect`.
# Edit a copy, not this file, while it runs: bash reads scripts incrementally.
set -uo pipefail
PY=/d/miniconda3/envs/gar/python.exe
K=/d/tools/kaggle-cli/Scripts/kaggle.exe
export PYTHONIOENCODING=utf-8
START=${1:-ablation}

wait_kernel() {  # tolerate network errors: only a final state ends the wait
  until s=$($K kernels status "idrissjedid/$1" 2>&1); echo "$s" | grep -q -E "COMPLETE|ERROR|CANCEL"; do sleep 300; done
  echo "$(date '+%F %T') $1: $s"
}

push() {  # stage name, then kaggle_push.py arguments
  local stage=$1; shift
  if [ "$started" = yes ]; then
    $PY scripts/kaggle_push.py kernel --name "gar-$stage" "$@" | tail -1
    sleep 120
  fi
  [ "$stage" = "$START" ] && started=yes
  [ "$started" = yes ] && wait_kernel "gar-$stage"
}

started=no
push ablation --reuse gar-pipeline --steps crops logits b5_g2 b6_g2 b3_g2 b8_avg b8_g4 transformer \
  transformer_nobox e2e person_cnx feats_person_cnx person_lstm_cnx b8_cnx transformer_cnx b1_cnx e2e_cnx ensemble
push detect --reuse gar-pipeline gar-ablation --steps detect
push optuna --reuse gar-pipeline gar-ablation --steps optuna/transformer optuna/transformer_cnx optuna/b8 \
  ensemble_greedy
push boost --reuse gar-pipeline gar-ablation gar-optuna --steps crops tta/b1 tta/b1_cnx tta/e2e tta/e2e_cnx \
  tta/transformer tta/transformer_cnx e2e_cnx_s1 e2e_cnx_s2 tta/e2e_cnx_s1 tta/e2e_cnx_s2 ensemble_greedy
push v1 --steps crops v1/b1 v1/person v1/feats_ft v1/feats_imagenet v1/feats_image v1/b2 v1/b3 v1/b4 \
  v1/person_lstm v1/b5 v1/b6 v1/b7 v1/full v1/full_g1 v1/full_avg v1/full_g4
