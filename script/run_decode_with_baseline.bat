@echo off
REM Run decode compile with baseline sheet dumped
REM Assumes the conda env UniNDP is available
call "%~dp0..\..\..\..\Program Files\Anaconda\shell\condabin\conda-hook.ps1"
call conda activate UniNDP
cd /d "%~dp0..\"
python pax\compile_e2e_pax.py --models-csv script\models_llm.csv --model GPT3-2.7B --architecture hbm-pim --batchsize 1 --output-dir debug --dump-baseline
