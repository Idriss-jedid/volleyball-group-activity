#!/usr/bin/env bash
# Smoke test of the post-paper models (needs results_smoke/ from scripts/smoke_all.sh).
set -euo pipefail
PY=${PY:-/d/miniconda3/envs/gar/python.exe}
export GAR_OVERLAY=configs/overlays/smoke.yaml PYTHONUNBUFFERED=1
R=results_smoke
run() { echo "=== $1"; shift; "$@" 2>&1 | grep -v -i warning | tail -2; }
for b in b5_g2 b6_g2; do run $b $PY -m gar.train --config configs/$b.yaml --name $b --smoke; done
run person_cnx       $PY -m gar.train --config configs/person_cnx.yaml --name person_cnx --smoke
run feats_person_cnx $PY -m gar.features.extractor --config configs/person_cnx.yaml --checkpoint $R/person_cnx/best.pt --name person_cnx --workers 2
for b in person_lstm_cnx b8_cnx transformer_cnx; do run $b $PY -m gar.train --config configs/$b.yaml --name $b --smoke; done
GAR_OVERLAY=configs/overlays/smoke_e2e.yaml run e2e $PY -m gar.train --config configs/e2e.yaml --name e2e --smoke
for r in b8 transformer e2e b8_cnx; do
  GAR_OVERLAY=configs/overlays/smoke_e2e.yaml run "logits $r" $PY -m gar.evaluate --run $R/$r --split val
done
run ensemble $PY -m gar.ensemble --results $R --runs b8 transformer e2e
echo "=== smoke_new OK"
