$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"

Set-Location (Split-Path $PSScriptRoot -Parent)

$logPath = "runs/tier4_encoder_ft_pipeline.log"
$trainStdout = "runs/tier4_encoder_ft_train.stdout.log"
$trainStderr = "runs/tier4_encoder_ft_train.stderr.log"

function Write-LogLine {
    param([string]$Message)
    $line = "[{0}] {1}" -f (Get-Date -Format o), $Message
    $line | Tee-Object -FilePath $logPath -Append
}

function Invoke-Step {
    param(
        [string]$Name,
        [string[]]$Arguments
    )
    Write-LogLine "START $Name"
    & .\.venv\Scripts\python.exe @Arguments 2>&1 | Tee-Object -FilePath $logPath -Append
    Write-LogLine "END $Name"
}

New-Item -ItemType Directory -Force -Path "runs\retrieval_encoder_finetune" | Out-Null
if (Test-Path $logPath) { Remove-Item $logPath -Force }
if (Test-Path $trainStdout) { Remove-Item $trainStdout -Force }
if (Test-Path $trainStderr) { Remove-Item $trainStderr -Force }

Write-LogLine "Tier 4 realistic cross-view encoder fine-tune pipeline"

Invoke-Step -Name "train_retrieval_encoder" -Arguments @(
    "-m", "src.tools.train_retrieval_encoder",
    "--triplets", "runs/paris_realistic_crossview_train_triplets_v1.jsonl",
    "--query-images-dir", "data/paris_realistic_v1/street_combined",
    "--reference-images-dir", "data/paris_realistic_v1_combined",
    "--model-id", "openai/clip-vit-large-patch14",
    "--output-dir", "runs/retrieval_encoder_finetune/paris_realistic_crossview_v1_e1",
    "--report-output", "runs/retrieval_encoder_finetune/paris_realistic_crossview_v1_e1.report.json",
    "--train-scope", "vision_encoder",
    "--epochs", "1",
    "--batch-size", "8",
    "--learning-rate", "1e-5",
    "--weight-decay", "1e-4",
    "--margin", "0.08",
    "--temperature", "0.07",
    "--ce-weight", "0.2",
    "--sample-weight-mode", "triplet_weight",
    "--sample-weight-max", "3.0",
    "--seed", "42",
    "--device", "auto"
)

Invoke-Step -Name "build_realistic_aerial_index" -Arguments @(
    "-m", "src.tools.build_realistic_aerial_index",
    "--root", "data/paris_realistic_v1_combined",
    "--metadata", "aerial/metadata.csv",
    "--images-dir", "aerial/images",
    "--output", "indices/aerial_clip_index_retrieval_encoder_ft_v1_e1.npz",
    "--model-id", "runs/retrieval_encoder_finetune/paris_realistic_crossview_v1_e1"
)

Invoke-Step -Name "eval_realistic_crossview" -Arguments @(
    "-m", "src.tools.eval_realistic_crossview",
    "--test-pairs", "data/paris_realistic_v1_combined/splits_strict/test_pairs_probe240.csv",
    "--aerial-metadata", "data/paris_realistic_v1_combined/aerial/metadata.csv",
    "--street-images-dir", "data/paris_realistic_v1/street_combined",
    "--aerial-index", "data/paris_realistic_v1_combined/indices/aerial_clip_index_retrieval_encoder_ft_v1_e1.npz",
    "--embedding-model", "runs/retrieval_encoder_finetune/paris_realistic_crossview_v1_e1",
    "--output", "runs/eval_realistic_crossview_combined_strict_probe240_encoderft_v1_e1_full40k.json",
    "--top-k", "50"
)

Write-LogLine "Tier 4 pipeline complete"
