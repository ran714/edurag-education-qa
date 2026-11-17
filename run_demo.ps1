param(
    [int]$Port = 0,
    [switch]$NoBrowser
)

$ErrorActionPreference = "Stop"
Set-Location -LiteralPath $PSScriptRoot

if ($Port -le 0) {
    if ($env:PORT -and [int]::TryParse($env:PORT, [ref]$Port)) {
        # 使用环境变量中的端口。
    } else {
        $Port = 8088
    }
}

function Get-EduRagHealth([int]$CandidatePort) {
    try {
        return Invoke-RestMethod -Uri "http://127.0.0.1:$CandidatePort/health" -TimeoutSec 2
    } catch {
        return $null
    }
}

function Test-PortInUse([int]$CandidatePort) {
    $listener = Get-NetTCPConnection -LocalPort $CandidatePort -State Listen -ErrorAction SilentlyContinue |
        Select-Object -First 1
    return $null -ne $listener
}

function Open-EduRagPage([string]$Url) {
    if (-not $NoBrowser) {
        try {
            Start-Process $Url
        } catch {
            Write-Host "未能自动打开浏览器，请手动访问：$Url" -ForegroundColor Yellow
        }
    }
}

$health = Get-EduRagHealth $Port
if ($health -and $health.status -eq "healthy" -and $health.mode -eq "demo" -and $health.version -eq "1.2.1") {
    $url = "http://127.0.0.1:$Port"
    Write-Host "EduRAG 已经在运行，无需重复启动。" -ForegroundColor Green
    Write-Host "访问地址：$url" -ForegroundColor Cyan
    Open-EduRagPage $url
    exit 0
}

if (Test-PortInUse $Port) {
    $requestedPort = $Port
    $freePort = $null
    foreach ($candidate in (($requestedPort + 1)..($requestedPort + 10))) {
        if (-not (Test-PortInUse $candidate)) {
            $freePort = $candidate
            break
        }
    }

    if ($null -eq $freePort) {
        throw "端口 $requestedPort 到 $($requestedPort + 10) 均被占用，请使用 .\run_demo.ps1 -Port 其它端口。"
    }

    $Port = $freePort
    Write-Host "端口 $requestedPort 已被其他程序占用，自动改用端口 $Port。" -ForegroundColor Yellow
}

$env:EDURAG_MODE = "demo"
$env:PORT = [string]$Port
$url = "http://127.0.0.1:$Port"
$stdoutLog = Join-Path $PSScriptRoot "server.stdout.log"
$stderrLog = Join-Path $PSScriptRoot "server.stderr.log"
$pidFile = Join-Path $PSScriptRoot "server.pid"
$python = Join-Path $PSScriptRoot "venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $python)) {
    $python = (Get-Command python -ErrorAction Stop).Source
}

Write-Host "正在启动 EduRAG 演示模式……" -ForegroundColor Cyan
$process = Start-Process -FilePath $python `
    -ArgumentList "app.py" `
    -WorkingDirectory $PSScriptRoot `
    -RedirectStandardOutput $stdoutLog `
    -RedirectStandardError $stderrLog `
    -WindowStyle Hidden `
    -PassThru
Set-Content -LiteralPath $pidFile -Value $process.Id -Encoding ASCII

$deadline = (Get-Date).AddSeconds(25)
do {
    Start-Sleep -Milliseconds 500
    $health = Get-EduRagHealth $Port
    if ($health -and $health.status -eq "healthy") {
        Write-Host "EduRAG 启动成功（进程 PID：$($process.Id)，模式：$($health.mode)）。" -ForegroundColor Green
        Write-Host "访问地址：$url" -ForegroundColor Cyan
        Write-Host "接口文档：$url/docs" -ForegroundColor DarkCyan
        Open-EduRagPage $url
        exit 0
    }

    if ($process.HasExited) {
        break
    }
} while ((Get-Date) -lt $deadline)

Write-Host "EduRAG 未能正常启动。错误日志：" -ForegroundColor Red
if (Test-Path -LiteralPath $stderrLog) {
    Get-Content -LiteralPath $stderrLog -Tail 30
}
exit 1
