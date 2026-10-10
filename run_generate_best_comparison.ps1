param(
    [string]$Checkpoint = 'Z:\Outpaint_Checkpoint\DPM_OutpaintForCFRTP\outputs\checkpoints\outpaint_3d\mask64_size256_sourcel2-1.485_ds2_ch8_res2_simple_scratch_v9_gate09_20260808_201046\ckpt\ldm_ckpt_138000.pt',

    [double]$ControlGateScale = 1.4,

    [int]$ReferenceSamples = 2,

    [int]$Repeats = 3,

    [string]$VaeCheckpoint = '',

    [string]$OutputDir = ''
)

$ErrorActionPreference = 'Stop'
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$Checkpoint = (Resolve-Path -LiteralPath $Checkpoint).Path
$CheckpointDir = Split-Path -Parent $Checkpoint
$SourceConfigPath = Join-Path $CheckpointDir 'config_used.json'
if (-not (Test-Path -LiteralPath $SourceConfigPath)) {
    throw "Missing config_used.json beside checkpoint: $Checkpoint"
}

$SourceConfig = Get-Content -LiteralPath $SourceConfigPath -Raw | ConvertFrom-Json
if ([string]::IsNullOrWhiteSpace($VaeCheckpoint)) {
    $VaeCheckpoint = [string]$SourceConfig.vae_checkpoint
}
$VaeCheckpoint = (Resolve-Path -LiteralPath $VaeCheckpoint).Path

if ([string]::IsNullOrWhiteSpace($OutputDir)) {
    $Timestamp = Get-Date -Format 'yyyyMMdd_HHmmss'
    $GateToken = ('{0:0.00}' -f $ControlGateScale).Replace('.', 'p')
    $OutputDir = Join-Path $ProjectRoot "outputs\comparison\best_step138000_gate${GateToken}_$Timestamp"
}
$OutputDir = [IO.Path]::GetFullPath($OutputDir)
New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null

$OutLog = Join-Path $OutputDir 'generation.out.log'
$ErrLog = Join-Path $OutputDir 'generation.err.log'
$PythonArguments = @(
    "`"$ProjectRoot\scripts\outpaint_3d_train_ldm.py`"",
    '--config', "`"$ProjectRoot\configs\outpaint_ldm_3d_simple_scratch.yaml`"",
    '--vae_checkpoint', "`"$VaeCheckpoint`"",
    '--resume_checkpoint', "`"$Checkpoint`"",
    '--control_gate_scale', [string]$ControlGateScale,
    '--eval_only', 'true',
    '--validation_use_ema', 'false',
    '--validation_save_volume_pt', 'true',
    '--validation_organized_output', 'true',
    '--validation_repeats', [string]$Repeats,
    '--eval_num_samples', [string]$ReferenceSamples,
    '--output_dir', "`"$OutputDir`""
)

Write-Host "Generating $ReferenceSamples reference samples x $Repeats repeats"
Write-Host "Checkpoint: $Checkpoint"
Write-Host "Inference control gate: $ControlGateScale"
Write-Host "Output: $OutputDir"

$PythonProcess = Start-Process `
    -FilePath (Get-Command python).Source `
    -ArgumentList $PythonArguments `
    -Wait `
    -PassThru `
    -WindowStyle Hidden `
    -RedirectStandardOutput $OutLog `
    -RedirectStandardError $ErrLog

if ($PythonProcess.ExitCode -ne 0) {
    throw "Comparison generation failed. See $ErrLog"
}

$RunManifest = [ordered]@{
    schema_version = 1
    created_at = (Get-Date).ToString('o')
    checkpoint = $Checkpoint
    checkpoint_step = 138000
    trained_control_gate_scale = [double]$SourceConfig.control_gate_scale
    inference_control_gate_scale = $ControlGateScale
    vae_checkpoint = $VaeCheckpoint
    reference_samples = $ReferenceSamples
    repeats_per_sample = $Repeats
    output_dir = $OutputDir
    status = 'complete'
}
$RunManifest | ConvertTo-Json -Depth 10 | Set-Content -LiteralPath (Join-Path $OutputDir 'run_manifest.json') -Encoding utf8
Write-Host "Comparison generation complete: $OutputDir"
