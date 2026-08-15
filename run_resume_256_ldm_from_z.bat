@echo off
setlocal
cd /d C:\Terong\DPM_OutpaintForCFRTP
set PYTORCH_ALLOC_CONF=expandable_segments:True
if "%~1"=="" (
  echo Usage: %~nx0 ^<schema-v5-checkpoint.pt^> ^<matching-vae-checkpoint.pt^>
  echo Legacy checkpoints such as step 40000 cannot be resumed.
  exit /b 2
)
if "%~2"=="" (
  echo A matching VAE checkpoint is required.
  exit /b 2
)
set RESUME_CKPT=%~f1
set VAE_CKPT=%~f2
for %%I in ("%RESUME_CKPT%") do set RUN_DIR=%%~dpI
set LOG_DIR=outputs\logs
set RUN_TS=%DATE:~0,4%%DATE:~5,2%%DATE:~8,2%_%TIME:~0,2%%TIME:~3,2%%TIME:~6,2%
set RUN_TS=%RUN_TS: =0%
if not exist "%LOG_DIR%" mkdir "%LOG_DIR%"
echo %RUN_TS%> "%LOG_DIR%\ldm_resume_256_latest_run.txt"
python scripts\outpaint_3d_train_ldm.py --config configs\outpaint_ldm_3d.yaml --vae_checkpoint "%VAE_CKPT%" --output_dir "%RUN_DIR%" --resume_checkpoint "%RESUME_CKPT%" > "%LOG_DIR%\ldm_resume_256_%RUN_TS%.out.log" 2> "%LOG_DIR%\ldm_resume_256_%RUN_TS%.err.log"
endlocal
