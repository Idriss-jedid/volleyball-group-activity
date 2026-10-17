#!/usr/bin/env bash
# Run every pipeline step on 3 videos, a few batches each: catches bugs before a Kaggle run.
set -euo pipefail
PY=${PY:-/d/miniconda3/envs/gar/python.exe}
export GAR_OVERLAY=configs/overlays/smoke.yaml PYTHONUNBUFFERED=1
R=results_smoke
run() { echo "=== $1"; shift; "$@" 2>&1 | grep -v -i warning | tail -3; }
run person       $PY -m gar.train --config configs/person.yaml --name person --smoke
run feats_person $PY -m gar.features.extractor --checkpoint $R/person/best.pt --name person_ft --workers 2
run b3           $PY -m gar.train --config configs/b3.yaml --name b3 --smoke
run feats_image  $PY -m gar.features.extractor --level image --config configs/b1.yaml \
                   --checkpoint results/b1/best.pt --name b1_image --workers 2
run b4           $PY -m gar.train --config configs/b4.yaml --name b4 --smoke
run person_lstm  $PY -m gar.train --config configs/person_lstm.yaml --name person_lstm --smoke
for b in b6 b5 b7 b8 b3_g2 b8_avg b8_g4 transformer transformer_nobox; do
  run $b $PY -m gar.train --config configs/$b.yaml --name $b --smoke
done
echo "=== smoke OK"
