[CmdletBinding(SupportsShouldProcess)]
param()
$ErrorActionPreference = 'Stop'
& (Join-Path $PSScriptRoot 'stop.ps1') @PSBoundParameters
