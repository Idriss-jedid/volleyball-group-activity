#!/usr/bin/env bash
# Smoke test of V1 (paper-faithful AlexNet reproduction) on 3 videos.
set -euo pipefail
PY=${PY:-/d/miniconda3/envs/gar/python.exe}
export GAR_OVERLAY=configs/overlays/smoke.yaml PYTHONUNBUFFERED=1
R=results_smoke
run() { echo "=== $1"; shift; "$@" 2>&1 | grep -v -i warning | tail -2; }
train() { run "v1/$1" $PY -m gar.train --config configs/v1/$1.yaml --name v1/$1 --smoke; }
train b1
train person
run v1/feats_ft       $PY -m gar.features.extractor --config configs/v1/person.yaml --checkpoint $R/v1/person/best.pt --name v1_ft --workers 2
run v1/feats_imagenet $PY -m gar.features.extractor --config configs/v1/person.yaml --name v1_imagenet --workers 2
run v1/feats_image    $PY -m gar.features.extractor --level image --config configs/v1/b1.yaml --checkpoint $R/v1/b1/best.pt --name v1_image --workers 2
for b in b2 b3 b4 person_lstm b5 b6 b7 full full_g1 full_avg full_g4; do train $b; done
echo "=== smoke_v1 OK"
