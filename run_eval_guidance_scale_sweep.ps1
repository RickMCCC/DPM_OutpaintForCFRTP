param(
    [Parameter(Mandatory = $true)]
    [string]$Checkpoint,

    [double[]]$Scales = @(0.0, 0.3, 0.6, 0.9, 1.2, 1.5, 1.8),

    [string]$VaeCheckpoint = '',

    [string]$OutputRoot = 'Z:\Outpaint_Checkpoint\DPM_OutpaintForCFRTP\outputs\guidance_sweeps',

    [int]$EvalNumSamples = 2,

    [switch]$UseEma
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

$Timestamp = Get-Date -Format 'yyyyMMdd_HHmmss'
$SweepName = "inference_gate_scale_$Timestamp"
$SweepRoot = Join-Path $OutputRoot $SweepName
New-Item -ItemType Directory -Force -Path $SweepRoot | Out-Null
$CheckpointStep = if ([IO.Path]::GetFileName($Checkpoint) -match 'ldm_ckpt_(\d+)\.pt') {
    [int]$Matches[1]
}
else {
    0
}

$Manifest = [ordered]@{
    schema_version = 1
    created_at = (Get-Date).ToString('o')
    checkpoint = $Checkpoint
    checkpoint_step = $CheckpointStep
    trained_control_gate_scale = [double]$SourceConfig.control_gate_scale
    vae_checkpoint = $VaeCheckpoint
    validation_seed = [int]$SourceConfig.validation_seed
    eval_num_samples = $EvalNumSamples
    validation_use_ema = [bool]$UseEma
    scales = @($Scales)
    runs = @()
}

foreach ($Scale in $Scales) {
    $ScaleToken = ('{0:0.00}' -f $Scale).Replace('.', 'p')
    $RunDir = Join-Path $SweepRoot "gate_$ScaleToken"
    $CheckpointOutputDir = Join-Path $RunDir 'ckpt'
    New-Item -ItemType Directory -Force -Path $RunDir | Out-Null
    $OutLog = Join-Path $RunDir 'eval.out.log'
    $ErrLog = Join-Path $RunDir 'eval.err.log'
    $UseEmaText = if ($UseEma) { 'true' } else { 'false' }

    Write-Host "Evaluating control_gate_scale=$Scale -> $RunDir"
    $PythonArguments = @(
        "`"$ProjectRoot\scripts\outpaint_3d_train_ldm.py`"",
        '--config', "`"$ProjectRoot\configs\outpaint_ldm_3d_simple_scratch.yaml`"",
        '--vae_checkpoint', "`"$VaeCheckpoint`"",
        '--resume_checkpoint', "`"$Checkpoint`"",
        '--control_gate_scale', [string]$Scale,
        '--eval_only', 'true',
        '--validation_use_ema', $UseEmaText,
        '--eval_num_samples', [string]$EvalNumSamples,
        '--validation_save_volume_pt', 'false',
        '--output_dir', "`"$CheckpointOutputDir`""
    )
    $PythonProcess = Start-Process `
        -FilePath (Get-Command python).Source `
        -ArgumentList $PythonArguments `
        -Wait `
        -PassThru `
        -WindowStyle Hidden `
        -RedirectStandardOutput $OutLog `
        -RedirectStandardError $ErrLog

    if ($PythonProcess.ExitCode -ne 0) {
        throw "Guidance evaluation failed for scale $Scale. See $ErrLog"
    }

    $Manifest.runs += [ordered]@{
        scale = $Scale
        output_dir = $CheckpointOutputDir
        out_log = $OutLog
        err_log = $ErrLog
        status = 'complete'
    }
    $Manifest | ConvertTo-Json -Depth 10 | Set-Content -LiteralPath (Join-Path $SweepRoot 'manifest.json') -Encoding utf8
}

Write-Host "Guidance sweep complete: $SweepRoot"
