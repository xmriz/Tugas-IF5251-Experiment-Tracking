<#
.SYNOPSIS
  Jalankan seluruh pipeline dari awal (clean + repro + visualize).

.PARAMETER Clean
  Jika di-set, hapus folder output lama sebelum mereproduce pipeline.
#>

param(
  [switch]$Clean
)

# 1) Clean up jika diminta
if ($Clean) {
  Write-Host "==> Cleaning old outputs..." -ForegroundColor Cyan
  $pathsToRemove = @(
    "data\processed",
    "models",
    "logs",
    "mlruns",
    "dvc_plots"
  )
  foreach ($p in $pathsToRemove) {
    if (Test-Path $p) {
      Remove-Item -Recurse -Force $p
      Write-Host "  Removed $p"
    }
  }

  Write-Host "  Removing metrics files..."
  Get-ChildItem -Path . -Filter "metrics*.json" -File -ErrorAction SilentlyContinue | Remove-Item -Force
  if (Test-Path "metrics.csv") { Remove-Item "metrics.csv" -Force }
}

# 2) Reproduce pipeline with DVC
Write-Host "`n==> Running 'dvc repro --force'..." -ForegroundColor Cyan
dvc repro --force

# 3) Show metrics
Write-Host "`n==> Showing DVC metrics:" -ForegroundColor Cyan
dvc metrics show

# # 4) Show plots and open in browser
# Write-Host "`n==> Generating and opening DVC plots..." -ForegroundColor Cyan
# dvc plots show metrics.csv --open

# 5) Launch MLflow UI
Write-Host "`n==> Starting MLflow UI on port 5000..." -ForegroundColor Cyan
Start-Process mlflow -ArgumentList 'ui --backend-store-uri mlruns --port 5000'

# 6) Launch TensorBoard
Write-Host "`n==> Starting TensorBoard on port 6006..." -ForegroundColor Cyan
Start-Process tensorboard -ArgumentList '--logdir logs/nn --port 6006'

Write-Host "`nAll done!`n" -ForegroundColor Green
