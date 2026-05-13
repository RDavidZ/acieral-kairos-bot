# deploy/package.ps1 — Backup and bundle Acieral Kairos Bot for VPS deployment
#
# Run from project root:
#   cd C:\Users\zinda\projects\acieral-kairos-bot
#   .\deploy\package.ps1
#
# Output:
#   backups\{YYYY-MM-DD_HHMM}_acieral_kairos\   (full project copy)
#   Console prints included files + total size

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectRoot

# ---------------------------------------------------------------------------
# Step 1 — Backup current project
# ---------------------------------------------------------------------------
$Timestamp   = Get-Date -Format "yyyy-MM-dd_HHmm"
$BackupRoot  = Join-Path $ProjectRoot "backups"
$BackupDest  = Join-Path $BackupRoot "${Timestamp}_acieral_kairos"

New-Item -ItemType Directory -Force -Path $BackupRoot | Out-Null
New-Item -ItemType Directory -Force -Path $BackupDest | Out-Null

Write-Host "=== Acieral Kairos Bot Deployment Package ===" -ForegroundColor Cyan
Write-Host ""
Write-Host "Backing up to: $BackupDest" -ForegroundColor Yellow

# Robocopy: copy everything except venv and __pycache__
$RobocopyArgs = @(
    $ProjectRoot, $BackupDest,
    "/E",                          # all subdirectories
    "/XD", "venv", "__pycache__", ".git", "backups",
    "/XF", "*.pyc",
    "/NP",                         # no progress %
    "/NFL",                        # no file list (cleaner output)
    "/NDL"                         # no dir list
)
robocopy @RobocopyArgs | Out-Null

Write-Host "  Backup complete." -ForegroundColor Green
Write-Host ""

# ---------------------------------------------------------------------------
# Step 2 — Collect deployment files
# ---------------------------------------------------------------------------
Write-Host "Collecting deployment files..." -ForegroundColor Yellow

$ExcludeDirs = @(
    "venv",
    "__pycache__",
    ".git",
    "backups",
    "backtest\results",
    "backtest\results_train"
)

$ExcludeFilePatterns = @("*.pyc", "*.pyo")

# Parquet cache files — too large, fetched fresh on VPS
$ExcludeParquetDir = Join-Path $ProjectRoot "data\cache"

$IncludedFiles = [System.Collections.Generic.List[System.IO.FileInfo]]::new()
$TotalBytes    = 0

function Should-Exclude {
    param([System.IO.FileInfo]$File)

    $RelPath = $File.FullName.Substring($ProjectRoot.Length + 1)

    # Exclude venv, __pycache__, .git, backups, backtest/results*
    foreach ($dir in $ExcludeDirs) {
        if ($RelPath.StartsWith($dir, [System.StringComparison]::OrdinalIgnoreCase)) {
            return $true
        }
    }

    # Exclude *.pyc / *.pyo
    foreach ($pat in $ExcludeFilePatterns) {
        if ($File.Name -like $pat) { return $true }
    }

    # Exclude data/cache/*.parquet  (large — fetched fresh on VPS)
    if ($File.FullName.StartsWith($ExcludeParquetDir, [System.StringComparison]::OrdinalIgnoreCase) `
        -and $File.Extension -eq ".parquet") {
        return $true
    }

    return $false
}

Get-ChildItem -Path $ProjectRoot -File -Recurse | ForEach-Object {
    if (-not (Should-Exclude $_)) {
        $IncludedFiles.Add($_)
        $TotalBytes += $_.Length
    }
}

# ---------------------------------------------------------------------------
# Step 3 — Print summary
# ---------------------------------------------------------------------------
Write-Host ""
Write-Host "=== Included files ===" -ForegroundColor Cyan

$IncludedFiles | ForEach-Object {
    $Rel  = $_.FullName.Substring($ProjectRoot.Length + 1)
    $Size = if ($_.Length -ge 1MB) {
                "{0:F1} MB" -f ($_.Length / 1MB)
            } elseif ($_.Length -ge 1KB) {
                "{0:F0} KB" -f ($_.Length / 1KB)
            } else {
                "{0} B" -f $_.Length
            }
    Write-Host ("  {0,-60} {1,10}" -f $Rel, $Size)
}

Write-Host ""
Write-Host "=== Summary ===" -ForegroundColor Cyan
Write-Host ("  Files included : {0}" -f $IncludedFiles.Count)
Write-Host ("  Total size     : {0:F1} MB" -f ($TotalBytes / 1MB))
Write-Host ""
Write-Host "=== Next steps ===" -ForegroundColor Green
Write-Host "  1. SCP to VPS:"
Write-Host "     scp -r -i ~/.ssh/id_ed25519 . ubuntu@132.145.33.221:/home/ubuntu/acieral-kairos-bot"
Write-Host "  2. SSH and run setup:"
Write-Host "     ssh -i ~/.ssh/id_ed25519 ubuntu@132.145.33.221"
Write-Host "     cd /home/ubuntu/acieral-kairos-bot && ./deploy/setup_vps.sh"
Write-Host ""
