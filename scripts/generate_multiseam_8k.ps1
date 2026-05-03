# ----------------------------------------------------------------------------
# Generate ~8K multi-seam samples (2700 each of 2-seam, 3-seam, 4-seam).
#
# Pre-reqs:
#   $env:OPENROUTER_API_KEY = "sk-or-v1-..."
#
# Usage:
#   pwsh scripts\generate_multiseam_8k.ps1
#   pwsh scripts\generate_multiseam_8k.ps1 -SamplesPerType 1000   # quicker run
#   pwsh scripts\generate_multiseam_8k.ps1 -Workers 8             # lighter on the API
#   pwsh scripts\generate_multiseam_8k.ps1 -SkipMerge             # don't merge at the end
#   pwsh scripts\generate_multiseam_8k.ps1 -SkipChunk             # don't chunk after merge
#
# Each run is independently resumable. If a window crashes or you Ctrl+C
# halfway through, just re-run the same command -- build_training_dataset.py
# counts existing rows in the output CSV and only generates the remainder.
# ----------------------------------------------------------------------------

[CmdletBinding()]
param(
    [int]    $SamplesPerType = 2700,
    [int]    $Workers        = 16,
    [int]    $MaxPromptLen   = 3000,
    [string] $OutputDir      = "data",
    [string] $MergedName     = "train_or_multiseam_8k.csv",
    [switch] $SkipMerge,
    [switch] $SkipChunk
)

$ErrorActionPreference = "Stop"

if (-not $env:OPENROUTER_API_KEY) {
    Write-Error "OPENROUTER_API_KEY is not set. Run: `$env:OPENROUTER_API_KEY = 'sk-or-v1-...'"
    exit 1
}

# Resolve repo root from this script's location.
$RepoRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $RepoRoot

# Ensure output directory exists.
$OutDir = Join-Path $RepoRoot $OutputDir
if (-not (Test-Path $OutDir)) { New-Item -ItemType Directory -Force -Path $OutDir | Out-Null }

# The three jobs.
$jobs = @(
    @{
        Label    = "2-seam (ai_in_middle, h-ai-h)"
        Output   = Join-Path $OutputDir "train_or_2seam.csv"
        Args     = @(
            "--sample-types", "ai_in_middle"
        )
    },
    @{
        Label    = "3-seam (multi_seam n=4, h-ai-h-ai)"
        Output   = Join-Path $OutputDir "train_or_3seam.csv"
        Args     = @(
            "--sample-types", "multi_seam",
            "--multi-seam-segments", "4"
        )
    },
    @{
        Label    = "4-seam (multi_seam n=5, h-ai-h-ai-h)"
        Output   = Join-Path $OutputDir "train_or_4seam.csv"
        Args     = @(
            "--sample-types", "multi_seam",
            "--multi-seam-segments", "5"
        )
    }
)

$totalStart = Get-Date

foreach ($job in $jobs) {
    Write-Host ""
    Write-Host "============================================================" -ForegroundColor Cyan
    Write-Host " $($job.Label)" -ForegroundColor Cyan
    Write-Host " -> $($job.Output)  (target: $SamplesPerType samples)" -ForegroundColor Cyan
    Write-Host "============================================================" -ForegroundColor Cyan

    $jobStart = Get-Date

    $commonArgs = @(
        "scripts\build_training_dataset.py",
        "--provider", "openrouter",
        "--no-subsample",
        "--max-prompt-len", $MaxPromptLen,
        "--n-samples", $SamplesPerType,
        "--output", $job.Output,
        "--data-source", "mixed",
        "--cc-prob", "0.5",
        "--workers", $Workers
    ) + $job.Args

    & python @commonArgs
    if ($LASTEXITCODE -ne 0) {
        Write-Error "Job '$($job.Label)' failed with exit code $LASTEXITCODE. Re-run this script to resume."
        exit $LASTEXITCODE
    }

    $jobElapsed = (Get-Date) - $jobStart
    Write-Host (" -> finished in {0:N1} min" -f $jobElapsed.TotalMinutes) -ForegroundColor Green
}

if (-not $SkipMerge) {
    Write-Host ""
    Write-Host "============================================================" -ForegroundColor Cyan
    Write-Host " Merging into $MergedName" -ForegroundColor Cyan
    Write-Host "============================================================" -ForegroundColor Cyan

    $mergedPath = Join-Path $OutputDir $MergedName
    $mergeInputs = $jobs | ForEach-Object { $_.Output }
    & python "scripts\merge_csv_datasets.py" `
        --input @mergeInputs `
        --output $mergedPath
    if ($LASTEXITCODE -ne 0) { Write-Error "merge failed"; exit $LASTEXITCODE }

    if (-not $SkipChunk) {
        Write-Host ""
        Write-Host "============================================================" -ForegroundColor Cyan
        Write-Host " Chunking $MergedName to <=350 word rows" -ForegroundColor Cyan
        Write-Host "============================================================" -ForegroundColor Cyan
        & python "scripts\chunk_long_rows.py" `
            --input $mergedPath `
            --in-place
        if ($LASTEXITCODE -ne 0) { Write-Error "chunking failed"; exit $LASTEXITCODE }
    }
}

$totalElapsed = (Get-Date) - $totalStart
Write-Host ""
Write-Host "============================================================" -ForegroundColor Green
Write-Host (" All done. Total elapsed: {0:N1} min" -f $totalElapsed.TotalMinutes) -ForegroundColor Green
Write-Host "============================================================" -ForegroundColor Green
