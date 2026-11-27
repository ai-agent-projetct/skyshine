<#
    Enables the Windows Subsystem for Linux feature.

    Run this elevated. It does NOT reboot and it does NOT change any other
    system setting - read it before you run it.

    Why DISM rather than `wsl --install`: on a machine where WSL has never
    been installed, wsl.exe is the inbox stub, and its help lists only
    --install, --update, --status and --help. The --no-distribution flag it
    was given does not exist there, so the command failed without doing
    anything. DISM enables the feature directly and always works.
#>
[CmdletBinding()]
param(
    # Also enable VirtualMachinePlatform, if it is somehow off.
    [switch]$IncludeVmPlatform
)

$ErrorActionPreference = 'Stop'

# Leave a trace on disk. Twice now this step has silently not run - once
# because the flag did not exist, once because the execution policy blocked
# the script - and both times the only way to tell was to infer it from the
# absence of side effects. A log file settles it.
$logPath = Join-Path $PSScriptRoot 'enable-wsl.log'
function Note($text) {
    $line = "{0}  {1}" -f (Get-Date -Format 'yyyy-MM-dd HH:mm:ss'), $text
    Add-Content -Path $logPath -Value $line -Encoding utf8
    Write-Host $text
}

Note "--- enable-wsl.ps1 started (pid $PID) ---"

$identity = [Security.Principal.WindowsIdentity]::GetCurrent()
$principal = New-Object Security.Principal.WindowsPrincipal($identity)
if (-not $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
    Note "NOT ELEVATED - nothing was changed."
    Write-Host "Right-click PowerShell > Run as administrator, then run this again." -ForegroundColor Red
    exit 1
}
Note "elevated as $($identity.Name)"

Write-Host "Enabling Microsoft-Windows-Subsystem-Linux ..." -ForegroundColor Cyan
& dism.exe /online /enable-feature /featurename:Microsoft-Windows-Subsystem-Linux /all /norestart
$wslExit = $LASTEXITCODE

if ($IncludeVmPlatform) {
    Write-Host "Enabling VirtualMachinePlatform ..." -ForegroundColor Cyan
    & dism.exe /online /enable-feature /featurename:VirtualMachinePlatform /all /norestart
}

Write-Host ""
Note "dism exit code: $wslExit"
if ($wslExit -eq 0 -or $wslExit -eq 3010) {
    Note "Feature staged (exit $wslExit)."
} else {
    Note "DISM exited with $wslExit - the feature was NOT enabled."
    exit $wslExit
}

# Fast Startup is on, which means "Shut down" followed by powering on is a
# kernel resume from hibernation, not a boot. Staged features never apply
# that way. Only "Restart" performs a true cold start of the kernel.
$hiberboot = (Get-ItemProperty 'HKLM:\SYSTEM\CurrentControlSet\Control\Session Manager\Power' -Name HiberbootEnabled -ErrorAction SilentlyContinue).HiberbootEnabled
Write-Host ""
Write-Host "NEXT: use Start > Power > Restart." -ForegroundColor Yellow
if ($hiberboot -eq 1) {
    Write-Host "Fast Startup is enabled on this machine, so 'Shut down' and then" -ForegroundColor Yellow
    Write-Host "powering on will NOT apply this. It must be Restart." -ForegroundColor Yellow
}
Write-Host ""
Write-Host "After the restart, run:  wsl --update" -ForegroundColor Cyan
