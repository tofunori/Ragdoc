$oldPath = [Environment]::GetEnvironmentVariable("PATH", "User")
$binPath = Join-Path $env:USERPROFILE "bin"

if ($oldPath -notmatch [regex]::Escape($binPath)) {
    $newPath = "$oldPath;$binPath"
    [Environment]::SetEnvironmentVariable("PATH", $newPath, "User")
    Write-Host "OK: PATH updated with $binPath"
    Write-Host "User PATH: $newPath"
}
else {
    Write-Host "OK: $binPath already in PATH"
}
