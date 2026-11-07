param(
    [Parameter(Mandatory = $true)]
    [string]$Archive
)

$ErrorActionPreference = 'Stop'
Add-Type -AssemblyName System.IO.Compression.FileSystem
$archivePath = (Resolve-Path -LiteralPath $Archive).Path
$zip = [System.IO.Compression.ZipFile]::OpenRead($archivePath)
try {
    $root = [System.IO.Path]::GetFileNameWithoutExtension($archivePath)
    $required = @(
        'UnderTheGun.exe',
        'cudart64_12.dll',
        'msvcp140.dll',
        'vcruntime140.dll',
        'vcruntime140_1.dll',
        'evaluator_tables/binaries_by_id_data.txt',
        'evaluator_tables/suitbit_by_id_data.txt',
        'evaluator_tables/flush_data.txt',
        'evaluator_tables/noflush7_data.txt',
        'evaluator_tables/suits_data.txt',
        'evaluator_tables/dp_data.txt',
        'licenses/PokerHandEvaluator.txt',
        'licenses/CUDA-Poker-Calculator.txt',
        'licenses/FLTK.txt'
    )
    foreach ($path in $required) {
        $entry = $zip.GetEntry("$root/$path")
        if ($null -eq $entry -or $entry.Length -eq 0) {
            throw "Release archive is missing a nonempty $path"
        }
    }
    foreach ($entry in $zip.Entries) {
        if ($entry.Name -match '^(kernel32|user32|gdi32|ntdll|nvcuda|api-ms-.*|ext-ms-.*)\.dll$') {
            throw "Release must not redistribute OS or driver DLL: $($entry.Name)"
        }
    }
} finally {
    $zip.Dispose()
}

$hash = (Get-FileHash -LiteralPath $archivePath -Algorithm SHA256).Hash.ToLowerInvariant()
"$hash  $([System.IO.Path]::GetFileName($archivePath))" |
    Set-Content -LiteralPath "$archivePath.sha256" -Encoding ascii
Write-Host "Verified release contents and wrote SHA-256 checksum: $archivePath"
