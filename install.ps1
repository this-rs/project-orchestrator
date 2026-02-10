#Requires -Version 5.1
<#
.SYNOPSIS
    Project Orchestrator installer for Windows.

.DESCRIPTION
    Downloads and installs the Project Orchestrator binary from GitHub Releases.

.PARAMETER Version
    Specific version to install (default: latest).

.PARAMETER NoFrontend
    Install the light variant without embedded frontend.

.PARAMETER InstallDir
    Installation directory (default: $env:LOCALAPPDATA\orchestrator\bin).

.EXAMPLE
    irm https://raw.githubusercontent.com/this-rs/project-orchestrator/main/install.ps1 | iex

.EXAMPLE
    .\install.ps1 -Version 1.0.0 -NoFrontend
#>

param(
    [string]$Version = "",
    [switch]$NoFrontend,
    [string]$InstallDir = "$env:LOCALAPPDATA\orchestrator\bin"
)

$ErrorActionPreference = "Stop"

$Repo = "this-rs/project-orchestrator"
$BinaryName = "orchestrator"

# --- Helpers ---

function Write-Info  { param([string]$Msg) Write-Host "info: " -ForegroundColor Green -NoNewline; Write-Host $Msg }
function Write-Warn  { param([string]$Msg) Write-Host "warn: " -ForegroundColor Yellow -NoNewline; Write-Host $Msg }
function Write-Err   { param([string]$Msg) Write-Host "error: " -ForegroundColor Red -NoNewline; Write-Host $Msg; exit 1 }

# --- Get latest version ---

function Get-LatestVersion {
    try {
        $release = Invoke-RestMethod -Uri "https://api.github.com/repos/$Repo/releases/latest" -UseBasicParsing
        return $release.tag_name -replace '^v', ''
    }
    catch {
        Write-Err "Failed to fetch latest version from GitHub API: $_"
    }
}

# --- Main ---

function Main {
    Write-Info "Detected platform: windows-x86_64"

    # Resolve version
    if ([string]::IsNullOrEmpty($Version)) {
        Write-Info "Fetching latest version..."
        $Version = Get-LatestVersion
    }
    Write-Info "Version: v$Version"

    # Build archive name
    if ($NoFrontend) {
        $variant = "light"
        $archiveName = "$BinaryName-$Version-windows-x86_64"
    }
    else {
        $variant = "full"
        $archiveName = "$BinaryName-full-$Version-windows-x86_64"
    }
    $archiveFile = "$archiveName.zip"

    Write-Info "Variant: $variant ($archiveFile)"

    # URLs
    $baseUrl = "https://github.com/$Repo/releases/download/v$Version"
    $archiveUrl = "$baseUrl/$archiveFile"
    $checksumsUrl = "$baseUrl/checksums-sha256.txt"

    # Temp directory
    $tmpDir = Join-Path $env:TEMP "orchestrator-install-$(Get-Random)"
    New-Item -ItemType Directory -Force -Path $tmpDir | Out-Null

    try {
        # Download archive
        Write-Info "Downloading $archiveUrl..."
        try {
            Invoke-WebRequest -Uri $archiveUrl -OutFile "$tmpDir\$archiveFile" -UseBasicParsing
        }
        catch {
            Write-Err "Failed to download archive. Check that version v$Version exists and has a windows-x86_64 build."
        }

        # Verify checksum
        try {
            Invoke-WebRequest -Uri $checksumsUrl -OutFile "$tmpDir\checksums-sha256.txt" -UseBasicParsing
            $checksumLine = Get-Content "$tmpDir\checksums-sha256.txt" | Where-Object { $_ -match $archiveFile }
            if ($checksumLine) {
                $expected = ($checksumLine -split '\s+')[0]
                $actual = (Get-FileHash "$tmpDir\$archiveFile" -Algorithm SHA256).Hash.ToLower()
                if ($expected -ne $actual) {
                    Write-Err "Checksum mismatch!`n  Expected: $expected`n  Actual:   $actual"
                }
                Write-Info "Checksum verified"
            }
            else {
                Write-Warn "Archive not found in checksums file - skipping verification"
            }
        }
        catch {
            Write-Warn "Could not download checksums - skipping verification"
        }

        # Extract
        Write-Info "Extracting..."
        Expand-Archive -Path "$tmpDir\$archiveFile" -DestinationPath $tmpDir -Force

        # Install
        New-Item -ItemType Directory -Force -Path $InstallDir | Out-Null
        Copy-Item "$tmpDir\$archiveName\$BinaryName.exe" "$InstallDir\$BinaryName.exe" -Force

        Write-Info "Installed $BinaryName to $InstallDir\$BinaryName.exe"

        # Check PATH
        $userPath = [Environment]::GetEnvironmentVariable("Path", "User")
        if ($userPath -notlike "*$InstallDir*") {
            Write-Warn "$InstallDir is not in your PATH"
            Write-Host ""
            Write-Host "  To add it permanently, run:"
            Write-Host "    [Environment]::SetEnvironmentVariable('Path', `"$InstallDir;`$env:Path`", 'User')"
            Write-Host ""

            # Offer to add automatically
            $answer = Read-Host "  Add to PATH now? (y/N)"
            if ($answer -eq 'y' -or $answer -eq 'Y') {
                [Environment]::SetEnvironmentVariable("Path", "$InstallDir;$userPath", "User")
                $env:Path = "$InstallDir;$env:Path"
                Write-Info "Added $InstallDir to user PATH"
            }
        }

        Write-Info "Installation complete! Run '$BinaryName --help' to get started."
    }
    finally {
        # Cleanup
        Remove-Item -Recurse -Force $tmpDir -ErrorAction SilentlyContinue
    }
}

Main
