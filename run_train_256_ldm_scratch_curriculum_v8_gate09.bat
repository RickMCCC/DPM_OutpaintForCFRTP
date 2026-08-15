@echo off
setlocal
cd /d C:\Terong\DPM_OutpaintForCFRTP
set PYTORCH_ALLOC_CONF=expandable_segments:True
if "%~1"=="" (
  echo Usage: %~nx0 ^<vae-checkpoint.pt^>
  exit /b 2
)
set VAE_CKPT=%~f1
set RUN_TS=%DATE:~0,4%%DATE:~5,2%%DATE:~8,2%_%TIME:~0,2%%TIME:~3,2%%TIME:~6,2%
set RUN_TS=%RUN_TS: =0%
set RUN_DIR=Z:\Outpaint_Checkpoint\DPM_OutpaintForCFRTP\outputs\checkpoints\outpaint_3d\mask64_size256_sourcel2-1.485_ds2_ch8_res2_scratch_curriculum_v8_gate09_%RUN_TS%\ckpt
set LOG_DIR=outputs\logs
if not exist "%RUN_DIR%" mkdir "%RUN_DIR%"
if not exist "%LOG_DIR%" mkdir "%LOG_DIR%"
python scripts\outpaint_3d_train_ldm.py --config configs\outpaint_ldm_3d_scratch_curriculum.yaml --vae_checkpoint "%VAE_CKPT%" --output_dir "%RUN_DIR%" > "%LOG_DIR%\ldm_scratch_curriculum_v8_gate09_%RUN_TS%.out.log" 2> "%LOG_DIR%\ldm_scratch_curriculum_v8_gate09_%RUN_TS%.err.log"
endlocal
