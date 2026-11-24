$ErrorActionPreference = "Stop"
py -m pip install -r "$PSScriptRoot\requirements.txt"
py "$PSScriptRoot\vocab_atlas_15.py" --out "$PSScriptRoot\results"
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
Write-Host "`nOpen results\REPORT.md to see the census, and results\overlap_heatmap.png for the heatmap."
