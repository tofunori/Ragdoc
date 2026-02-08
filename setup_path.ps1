$oldPath = [Environment]::GetEnvironmentVariable("PATH", "User")
$binPath = "C:\Users\thier\bin"

if ($oldPath -notmatch [regex]::Escape($binPath)) {
    $newPath = "$oldPath;$binPath"
    [Environment]::SetEnvironmentVariable("PATH", $newPath, "User")
    Write-Host "OK: PATH mise a jour avec $binPath"
    Write-Host "PATH utilisateur: $newPath"
}
else {
    Write-Host "OK: $binPath deja dans PATH"
}
