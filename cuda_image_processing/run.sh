#!/bin/bash
mkdir -p data/output logs
./main data/input data/output > logs/execution_log.txt
echo "Done"
