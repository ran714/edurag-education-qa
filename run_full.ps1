param(
    [int]$Port = 8088,
    [switch]$NoBrowser,
    [switch]$SkipIndex
)

$ErrorActionPreference = "Stop"
$utf8 = New-Object System.Text.UTF8Encoding($false)
[Console]::InputEncoding = $utf8
[Console]::OutputEncoding = $utf8
$OutputEncoding = $utf8
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"

Set-Location -LiteralPath $PSScriptRoot
$env:DOCKER_CONFIG = Join-Path $PSScriptRoot ".docker-config"
$env:PYTHONPATH = $PSScriptRoot
New-Item -ItemType Directory -Path $env:DOCKER_CONFIG -Force | Out-Null

$python = Join-Path $PSScriptRoot "venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $python)) {
    throw "Python virtual environment not found: venv"
}

function Show-RunningService([string]$BaseUrl) {
    Write-Host "EduRAG full mode is already running." -ForegroundColor Green
    Write-Host "URL: $BaseUrl" -ForegroundColor Cyan
    Write-Host "Docs: $BaseUrl/docs" -ForegroundColor Cyan
    if (-not $NoBrowser) { Start-Process $BaseUrl }
}

function Show-LogTail([string]$Path, [int]$Lines = 40) {
    if (Test-Path -LiteralPath $Path) {
        Write-Host "--- $Path ---"
        Get-Content -LiteralPath $Path -Tail $Lines
    }
}

function Get-HealthyService([int]$ServicePort) {
    $listener = Get-NetTCPConnection -LocalPort $ServicePort -State Listen -ErrorAction SilentlyContinue | Select-Object -First 1
    if (-not $listener) { return $false }
    try {
        $health = Invoke-RestMethod -Uri "http://127.0.0.1:$ServicePort/health" -TimeoutSec 3
        return ($health.status -eq "healthy" -and $health.mode -eq "full" -and $health.version -eq "1.2.1")
    } catch {
        return $false
    }
}

# Reuse a healthy service immediately. This keeps repeated PyCharm runs fast and quiet.
if (Get-HealthyService $Port) {
    Show-RunningService "http://127.0.0.1:$Port"
    exit 0
}

if (Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue) {
    throw "Port $Port is already in use, possibly by an earlier EduRAG version. Choose a free port (for example -Port 8090). No running process was changed."
}

# Verify Docker and automatically start Docker Desktop when its engine is not ready.
# The old version failed immediately when the Docker named pipe did not exist.
$dockerCommand = Get-Command docker -ErrorAction SilentlyContinue
if (-not $dockerCommand) {
    throw "Docker CLI not found. Install Docker Desktop and add docker to PATH."
}

function Test-DockerReady {
    try {
        docker info 1>$null 2>$null
        return ($LASTEXITCODE -eq 0)
    } catch {
        return $false
    }
}

if (-not (Test-DockerReady)) {
    $programFilesX86 = [Environment]::GetEnvironmentVariable("ProgramFiles(x86)")
    $dockerDesktopCandidates = @(
        (Join-Path $env:ProgramFiles "Docker\Docker\Docker Desktop.exe"),
        (Join-Path $programFilesX86 "Docker\Docker\Docker Desktop.exe"),
        (Join-Path $env:LOCALAPPDATA "Docker\Docker Desktop.exe")
    ) | Where-Object { $_ -and (Test-Path -LiteralPath $_) }

    $dockerDesktop = $dockerDesktopCandidates | Select-Object -First 1
    if (-not $dockerDesktop) {
        throw "Docker Desktop was not found. Install it first."
    }

    $desktopProcess = Get-Process -Name "Docker Desktop" -ErrorAction SilentlyContinue
    if (-not $desktopProcess) {
        Write-Host "Docker Desktop is not running. Starting it automatically..." -ForegroundColor Yellow
        Start-Process -FilePath $dockerDesktop -WindowStyle Hidden
    } else {
        Write-Host "Docker Desktop is starting. Waiting for Docker Engine..." -ForegroundColor Yellow
    }

    $dockerDeadline = (Get-Date).AddSeconds(180)
    do {
        Start-Sleep -Seconds 3
        if (Test-DockerReady) { break }
        Write-Host "." -NoNewline -ForegroundColor DarkGray
    } while ((Get-Date) -lt $dockerDeadline)
    Write-Host

    if (-not (Test-DockerReady)) {
        throw "Docker Desktop started, but Docker Engine was not ready within 180 seconds. Open Docker Desktop, wait until it shows Running, then click the launcher again."
    }
}

Write-Host "Docker Engine is ready. Starting EduRAG..." -ForegroundColor Green
$serviceNames = @("mysql", "redis", "etcd", "minio", "milvus")
$existingServices = @()
$knownContainers = @(& docker ps -a --format '{{.Names}}')
if ($LASTEXITCODE -ne 0) { throw "Could not list Docker containers." }
foreach ($serviceName in $serviceNames) {
    if ($knownContainers -contains "edurag-$serviceName") {
        $containerJson = & docker inspect "edurag-$serviceName"
        if ($LASTEXITCODE -ne 0) { throw "Could not inspect edurag-$serviceName." }
        $container = ($containerJson | ConvertFrom-Json)[0]
        if ($container.Config.Labels.'com.docker.compose.service' -ne $serviceName -or
            -not $container.Config.Labels.'com.docker.compose.project') {
            throw "Container edurag-$serviceName exists but its Compose ownership could not be verified. No container was changed."
        }
        $existingServices += $container
    }
}

if ($existingServices.Count -gt 0) {
    $projects = @($existingServices | ForEach-Object {
        $_.Config.Labels.'com.docker.compose.project'
    } | Select-Object -Unique)
    if ($existingServices.Count -ne $serviceNames.Count -or $projects.Count -ne 1) {
        throw "An incomplete or mixed EduRAG container set exists. Inspect docker ps -a before continuing. No container was changed."
    }
    Write-Host "Reusing existing EduRAG containers (Compose project: $($projects[0]))." -ForegroundColor Cyan
    Write-Host "Their existing data mounts are preserved; this directory's .runtime is not substituted." -ForegroundColor DarkCyan
    foreach ($serviceName in @("etcd", "minio", "mysql", "redis", "milvus")) {
        $container = $existingServices | Where-Object { $_.Name -eq "/edurag-$serviceName" }
        if (-not $container.State.Running) {
            & docker start "edurag-$serviceName"
            if ($LASTEXITCODE -ne 0) { throw "Failed to start edurag-$serviceName." }
        }
    }
} else {
    $dockerLog = Join-Path $PSScriptRoot "docker.full.log"
    # Retain diagnostics without treating Docker's normal stderr progress as a PS error.
    $dockerProcess = Start-Process -FilePath $dockerCommand.Source `
        -ArgumentList @("compose", "--ansi", "never", "up", "-d") `
        -WorkingDirectory $PSScriptRoot `
        -RedirectStandardOutput $dockerLog `
        -RedirectStandardError (Join-Path $PSScriptRoot "docker.full.stderr.log") `
        -WindowStyle Hidden -Wait -PassThru
    if ($dockerProcess.ExitCode -ne 0) {
        Show-LogTail $dockerLog
        Show-LogTail (Join-Path $PSScriptRoot "docker.full.stderr.log")
        throw "Docker services failed to start. See docker.full.log and docker.full.stderr.log."
    }
}

function Wait-Tcp([string]$HostName, [int]$PortNumber, [int]$TimeoutSeconds = 180) {
    $deadline = (Get-Date).AddSeconds($TimeoutSeconds)
    do {
        if (Test-NetConnection -ComputerName $HostName -Port $PortNumber -InformationLevel Quiet -WarningAction SilentlyContinue) {
            return
        }
        Start-Sleep -Seconds 2
    } while ((Get-Date) -lt $deadline)
    throw "Timed out waiting for $HostName`:$PortNumber"
}

Wait-Tcp "127.0.0.1" 3306
Wait-Tcp "127.0.0.1" 6379
Wait-Tcp "127.0.0.1" 19530 300

$bootstrapLog = Join-Path $PSScriptRoot "bootstrap.full.log"
& $python (Join-Path $PSScriptRoot "bootstrap_database.py") *> $bootstrapLog
if ($LASTEXITCODE -ne 0) {
    Show-LogTail $bootstrapLog
    throw "MySQL initialization failed."
}

if (-not $SkipIndex) {
    $env:EDURAG_MODE = "full"
    $indexLog = Join-Path $PSScriptRoot "index.full.log"
    $collectionName = (& $python -c "from base.config import Config; print(Config().MILVUS_COLLECTION_NAME)").Trim()
    $indexErrorLog = Join-Path $PSScriptRoot "index.full.stderr.log"
    $indexArguments = @('"scripts\build_index.py"', '--collection', $collectionName)
    $indexProcess = Start-Process -FilePath $python -ArgumentList $indexArguments -WorkingDirectory $PSScriptRoot -RedirectStandardOutput $indexLog -RedirectStandardError $indexErrorLog -WindowStyle Hidden -Wait -PassThru
    if ($indexProcess.ExitCode -ne 0) {
        Show-LogTail $indexLog 80
        Show-LogTail $indexErrorLog 40
        throw "Milvus knowledge-base indexing failed."
    }
}

$env:EDURAG_MODE = "full"
$env:PORT = [string]$Port
$stdoutLog = Join-Path $PSScriptRoot "server.full.stdout.log"
$stderrLog = Join-Path $PSScriptRoot "server.full.stderr.log"
$pidFile = Join-Path $PSScriptRoot "server.full.pid"

if (Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue) {
    if (Get-HealthyService $Port) {
        Show-RunningService "http://127.0.0.1:$Port"
        exit 0
    }
    throw "Port $Port is occupied by another program or an unhealthy service."
}

$process = Start-Process -FilePath $python -ArgumentList "app.py" -WorkingDirectory $PSScriptRoot -RedirectStandardOutput $stdoutLog -RedirectStandardError $stderrLog -WindowStyle Hidden -PassThru
Set-Content -LiteralPath $pidFile -Value $process.Id -Encoding ASCII

$deadline = (Get-Date).AddMinutes(5)
do {
    Start-Sleep -Seconds 2
    try {
        $health = Invoke-RestMethod -Uri "http://127.0.0.1:$Port/health" -TimeoutSec 3
        if ($health.status -eq "healthy" -and $health.mode -eq "full") {
            $url = "http://127.0.0.1:$Port"
            Write-Host "EduRAG full mode started successfully. PID: $($process.Id)" -ForegroundColor Green
            Write-Host "URL: $url" -ForegroundColor Cyan
            Write-Host "Docs: $url/docs" -ForegroundColor Cyan
            if (-not $NoBrowser) { Start-Process $url }
            exit 0
        }
    } catch { }
    if ($process.HasExited) { break }
} while ((Get-Date) -lt $deadline)

Write-Host "EduRAG full mode failed to start." -ForegroundColor Red
Show-LogTail $stderrLog 80
Show-LogTail (Join-Path $PSScriptRoot "logs\app.log") 80
exit 1
