param(
    [string]$OutputPath = ""
)

$ErrorActionPreference = "Stop"
$root = $PSScriptRoot
$manifest = Join-Path $root "blender_manifest.toml"
$versionLine = Select-String -LiteralPath $manifest -Pattern '^version\s*=\s*"([^"]+)"$'
$version = if ($versionLine) { $versionLine.Matches[0].Groups[1].Value } else { "dev" }
$dist = Join-Path $root "dist"
if (-not $OutputPath) {
    $OutputPath = Join-Path $dist "luna_engine_io-$version.zip"
}
$OutputPath = [System.IO.Path]::GetFullPath($OutputPath)

New-Item -ItemType Directory -Force -Path ([System.IO.Path]::GetDirectoryName($OutputPath)) | Out-Null
$stage = Join-Path ([System.IO.Path]::GetTempPath()) ("luna_engine_io_package_" + [guid]::NewGuid().ToString("N"))
New-Item -ItemType Directory -Path $stage | Out-Null
try {
    Get-ChildItem -LiteralPath $root -File | Where-Object {
        $_.Extension -eq ".py" -or $_.Name -in @("blender_manifest.toml", "LICENSE", "README.md", "blender_import_model_anim_symbols.json")
    } | Copy-Item -Destination $stage

    if (Test-Path -LiteralPath $OutputPath) {
        Remove-Item -LiteralPath $OutputPath -Force
    }
    Compress-Archive -Path (Join-Path $stage "*") -DestinationPath $OutputPath -CompressionLevel Optimal

    Add-Type -AssemblyName System.IO.Compression.FileSystem
    $archive = [System.IO.Compression.ZipFile]::OpenRead($OutputPath)
    try {
        $entries = @($archive.Entries | ForEach-Object { $_.FullName -replace '\\', '/' })
        foreach ($required in @(
            "blender_manifest.toml",
            "__init__.py",
            "registration.py",
            "model_export.py",
            "model_import.py"
        )) {
            if ($required -notin $entries) {
                throw "Package is missing required entry: $required"
            }
        }
    }
    finally {
        $archive.Dispose()
    }
}
finally {
    if (Test-Path -LiteralPath $stage) {
        Remove-Item -LiteralPath $stage -Recurse -Force
    }
}

$file = Get-Item -LiteralPath $OutputPath
Write-Host "Created Blender extension: $($file.FullName) ($($file.Length) bytes)"
