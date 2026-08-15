@echo off
setlocal
cd /d C:\Terong\DPM_OutpaintForCFRTP
set PYTORCH_ALLOC_CONF=expandable_segments:True

set VAE_CKPT=Z:\Outpaint_Checkpoint\DPM_OutpaintForCFRTP\outputs\checkpoints\vae_3d\mask64_size256_sourcel2-1.485_ds2_ch8_res2_klwarm_20260713_184751\ckpt\vae_3d_ckpt_22000.pt
set RUN_TS=%DATE:~0,4%%DATE:~5,2%%DATE:~8,2%_%TIME:~0,2%%TIME:~3,2%%TIME:~6,2%
set RUN_TS=%RUN_TS: =0%
set RUN_TAG=mask64_size256_sourcel2-1.485_ds2_ch8_res2_context_structure_v5_short2k_%RUN_TS%
set RUN_DIR=Z:\Outpaint_Checkpoint\DPM_OutpaintForCFRTP\outputs\checkpoints\outpaint_3d\%RUN_TAG%\ckpt
set LOG_DIR=outputs\logs

if not exist "%VAE_CKPT%" (
  echo Missing VAE checkpoint: %VAE_CKPT%
  exit /b 2
)
if not exist "%RUN_DIR%" mkdir "%RUN_DIR%"
if not exist "%LOG_DIR%" mkdir "%LOG_DIR%"

echo %RUN_TAG%> "%LOG_DIR%\ldm_short_256_latest_run.txt"
python scripts\outpaint_3d_train_ldm.py ^
  --config configs\outpaint_ldm_3d.yaml ^
  --vae_checkpoint "%VAE_CKPT%" ^
  --output_dir "%RUN_DIR%" ^
  --lr_anneal_steps 2001 ^
  --save_interval 2000 ^
  --eval_num_samples 2 ^
  > "%LOG_DIR%\ldm_short_256_%RUN_TS%.out.log" ^
  2> "%LOG_DIR%\ldm_short_256_%RUN_TS%.err.log"

endlocal
