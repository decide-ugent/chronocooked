#!/bin/bash

PROJECT_NAME="time-perception-ughent"

short_durations=(3 2)
long_duration=(24 32)

for sd in "${short_durations[@]}"; do

  for ld in "${long_duration[@]}"; do


gpulab-cli --debug submit --project $PROJECT_NAME <<EOF
{
  "name": "chronocooked_bisection_ctrnn_sd${sd}_ld${ld}",
  "request": {
    "docker": {
      "image": "pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime",
      "command": "/bin/bash -c 'apt update && apt install -y python3-venv && python3 -m venv venv && source venv/bin/activate && pip install --no-cache-dir stable-baselines3[extra] sb3-contrib gymnasium && python /project_ghent/chronocooked/singleagent/train/gpulab_bisection_ctrnn.py'",
      "environment": {
      "SHORT_DURATION": "${sd}",
      "LONG_DURATION": "${ld}"
      },
      "portMappings": [],
      "storage": [
        {
          "hostPath": "/project_ghent/chronocooked",
          "containerPath": "/project_ghent/chronocooked"
        }
      ]
    },
    "resources": {
      "clusterId": 8,
      "gpus": 0,
      "cpus": 2,
      "cpuMemoryGb": 16
    },
    "scheduling": {},
    "extra": {}
  }
}
EOF
  done
done