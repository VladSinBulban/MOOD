$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$appPath = Join-Path $projectRoot 'app.py'
$enginePath = Join-Path $projectRoot 'src\image_engine.py'
$scenarioPath = Join-Path $projectRoot 'scenarios.json'

foreach ($path in @($appPath, $enginePath, $scenarioPath)) {
    if (-not (Test-Path $path)) { throw "Required file not found: $path" }
}

$app = Get-Content -Raw -Encoding UTF8 $appPath
$startMarker = '# LOCAL EDITING ENGINE'
$endMarker = '# GEMINI IMAGE EDITING'
$start = $app.IndexOf($startMarker, [StringComparison]::Ordinal)
$end = $app.IndexOf($endMarker, [StringComparison]::Ordinal)
if ($start -lt 0 -or $end -le $start) {
    throw 'Could not find the expected LOCAL EDITING ENGINE / GEMINI IMAGE EDITING markers in app.py. No changes were made.'
}
$replacement = "# LOCAL EDITING ENGINE`r`n`r`nfrom src.image_engine import local_edit`r`n`r`n"
$newApp = $app.Substring(0, $start) + $replacement + $app.Substring($end)

# Keep a backup outside the project so it cannot be accidentally committed.
$backup = Join-Path $env:TEMP ('MOOD_app.py.backup.' + (Get-Date -Format 'yyyyMMdd_HHmmss'))
Set-Content -Path $backup -Value $app -Encoding UTF8
Set-Content -Path $appPath -Value $newApp -Encoding UTF8
Write-Host 'MOOD scenario engine installed into app.py.' -ForegroundColor Green
Write-Host "Original app.py backup: $backup"
Write-Host 'Added scenarios.json and src/image_engine.py are now used by the local fallback.'
Write-Host 'Next: run the tests and start MOOD locally.'
