#!/bin/bash

set -x

pip install -e ./mmdetection3d/ --use-deprecated legacy-resolver
pip install -e .
pip install numpy==1.21.0
