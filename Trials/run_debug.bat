@echo off
cd /d "C:\Users\ericb\Documents\GitHub\PSD_opt"
python Trials\debug_minimal.py > Trials\debug_output.txt 2>&1
type Trials\debug_output.txt
