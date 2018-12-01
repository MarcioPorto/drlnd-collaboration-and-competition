#!/bin/bash
tensorboard --logdir=./log/ --host=localhost --port=3000 &> /dev/null &
echo "Wait around 10 seconds and open this link at your browser (ignore other outputs):"
