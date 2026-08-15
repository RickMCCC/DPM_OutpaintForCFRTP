@echo off
setlocal
cd /d C:\Terong\DPM_OutpaintForCFRTP
set PYTORCH_ALLOC_CONF=expandable_segments:True
if "%~3"=="" (
  echo Usage: %~nx0 ^<schema-v6-checkpoint.pt^> ^<vae-checkpoint.pt^> ^<schema-v4-backbone.pt^>
  exit /b 2
)
set RESUME_CKPT=%~f1
set VAE_CKPT=%~f2
set BACKBONE_CKPT=%~f3
set RUN_DIR=%~dp1
if "%RUN_DIR:~-1%"=="\" set RUN_DIR=%RUN_DIR:~0,-1%
set RUN_TS=%DATE:~0,4%%DATE:~5,2%%DATE:~8,2%_%TIME:~0,2%%TIME:~3,2%%TIME:~6,2%
set RUN_TS=%RUN_TS: =0%
set LOG_DIR=outputs\logs
if not exist "%LOG_DIR%" mkdir "%LOG_DIR%"
python scripts\outpaint_3d_train_ldm.py --config configs\outpaint_ldm_3d.yaml --resume_checkpoint "%RESUME_CKPT%" --vae_checkpoint "%VAE_CKPT%" --backbone_checkpoint "%BACKBONE_CKPT%" --output_dir "%RUN_DIR%" --training_steps 100001 --lr_anneal_steps 0 --lr 2e-5 --min_lr 2e-6 --lr_warmup_steps 0 --backbone_lr 2e-6 --freeze_backbone_steps 2000 --batch_size 3 --control_gate_scale 0.3 > "%LOG_DIR%\ldm_structure_warmstart_v6_resume100001_%RUN_TS%.out.log" 2> "%LOG_DIR%\ldm_structure_warmstart_v6_resume100001_%RUN_TS%.err.log"
endlocal
