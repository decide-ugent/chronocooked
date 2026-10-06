#!/bin/bash

PROJECT_NAME="time-perception-ughent"

#lstm_size=(125 8 256)


#for ls in "${lstm_size[@]}"; do



gpulab-cli --debug submit --project $PROJECT_NAME <<EOF
{
  "name": "chronocooked_cat_multitimer_cnn",
  "request": {
    "docker": {
      "image": "pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime",
      "command": "/bin/bash -c 'apt update && apt install -y python3-venv && python3 -m venv venv && source venv/bin/activate && pip install --no-cache-dir stable-baselines3[extra] sb3-contrib gymnasium && python /project_ghent/chronocooked/singleagent/train/gpulab_cat_multitimer_cl_tp_cnn.py'",
      "environment": {

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


#done