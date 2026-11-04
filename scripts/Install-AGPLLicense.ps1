# Run from the root of the release repository. Fetch the full official AGPL text.
# No GitHub credentials are used. Nothing is committed, pushed or made public.
[CmdletBinding()]
param()
$ErrorActionPreference = 'Stop'
$root = (Get-Location).Path
if (-not (Test-Path -LiteralPath (Join-Path $root 'scripts/preflight_release.py'))) {
    throw 'Execute este script desde la raiz del repositorio de entrega SolarSpine R2B.'
}
$mit = Join-Path $root 'LICENSE-MIT-ORIGINAL'
$license = Join-Path $root 'LICENSE'
if (-not (Test-Path -LiteralPath $mit)) {
    if ((Get-Content -LiteralPath $license -TotalCount 1) -ne 'MIT License') {
        throw 'Se esperaba una licencia MIT original; abortando para evitar sobrescritura.'
    }
    Copy-Item -LiteralPath $license -Destination $mit
}
# Avoid accidental replacement if a user edited LICENSE unexpectedly.
$first = Get-Content -LiteralPath $license -TotalCount 1
if ($first -ne 'MIT License' -and $first -ne 'GNU AFFERO GENERAL PUBLIC LICENSE') {
    throw 'LICENSE tiene contenido inesperado; revisarlo manualmente.'
}
[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
$uri = 'https://www.gnu.org/licenses/agpl-3.0.txt'
$temp = Join-Path $root 'LICENSE.agpl.download.tmp'
try {
    Invoke-WebRequest -Uri $uri -OutFile $temp -UseBasicParsing -TimeoutSec 60
    $raw = Get-Content -LiteralPath $temp -Raw -Encoding utf8
    if ($raw.Length -lt 25000 -or -not $raw.Contains('GNU AFFERO GENERAL PUBLIC LICENSE') -or -not $raw.Contains('Version 3, 19 November 2007')) {
        throw 'La descarga no parece ser el texto completo oficial AGPL-3.0.'
    }
    Move-Item -LiteralPath $temp -Destination $license -Force
    Write-Host 'AGPL_LICENSE_INSTALLED_OK' -ForegroundColor Green
    Write-Host ('FULL_LICENSE_BYTES: ' + (Get-Item -LiteralPath $license).Length)
    Write-Host 'MIT_ORIGINAL_PRESERVED'
} finally {
    if (Test-Path -LiteralPath $temp) { Remove-Item -LiteralPath $temp -Force }
}
Write-Host 'Revisa git diff; no se ha ejecutado commit, push ni cambio de visibilidad.'
