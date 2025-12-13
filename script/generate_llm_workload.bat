@echo off
REM Thin wrapper around generate_llm_workload.py for Windows CMD.
REM
REM 示例：
REM   generate_llm_workload.bat --model Qwen3-8B --phase decode --tklen 32
REM   generate_llm_workload.bat --model-csv script\models_llm.csv --phase decode
REM   generate_llm_workload.bat --manual-name Qwen3-8B --ndec 36 --hdim 4096 --nheads 32 --dhead 128 --ff-scale 3.0 --gqaheads 8 --phase decode --tklen 32

setlocal ENABLEDELAYEDEXPANSION

REM 进入脚本所在目录，保证相对路径一致
cd /d "%~dp0\.."

python "script\generate_llm_workload.py" %*

endlocal







