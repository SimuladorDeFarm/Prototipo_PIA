<#
.SYNOPSIS
    Levanta el backend (FastAPI/uvicorn) y el frontend (http.server) juntos
    en Windows, en una sola consola.

.DESCRIPTION
    Muestra el estado de cada proceso y el link del frontend. Con Ctrl+C
    detiene ambos de forma segura (mata el arbol de procesos con taskkill
    y libera los puertos como respaldo). Equivalente Windows de iniciar.sh.

.NOTES
    Si Windows bloquea la ejecucion de scripts, corre una vez en PowerShell
    (sin permisos de administrador):

        Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned

    Uso:
        .\iniciar.ps1
#>

$ProjectRoot   = Split-Path -Parent $MyInvocation.MyCommand.Path
$BackendDir    = Join-Path $ProjectRoot "backend"
$FrontendDir   = Join-Path $ProjectRoot "frontend"
$PythonBackend = Join-Path $BackendDir ".venv\Scripts\python.exe"

$BackendHost  = "0.0.0.0"
$BackendPort  = 8000
$FrontendPort = 5501

if (-not (Test-Path $PythonBackend)) {
    Write-Host "No se encontro el entorno virtual del backend en: $PythonBackend"
    Write-Host "Corre primero: .\install.ps1"
    exit 1
}

& $PythonBackend -c "import fastapi, uvicorn" *> $null
if ($LASTEXITCODE -ne 0) {
    Write-Host "El entorno virtual existe pero faltan dependencias (fastapi/uvicorn)."
    Write-Host "Corre primero: .\install.ps1"
    exit 1
}

function Liberar-Puerto {
    param([int]$Puerto)
    if (-not (Get-Command Get-NetTCPConnection -ErrorAction SilentlyContinue)) { return }
    try {
        Get-NetTCPConnection -LocalPort $Puerto -ErrorAction SilentlyContinue |
            ForEach-Object { Stop-Process -Id $_.OwningProcess -Force -ErrorAction SilentlyContinue }
    } catch {}
}

function Detener-Proceso {
    param($Proceso)
    if ($null -eq $Proceso) { return }
    if ($Proceso.HasExited) { return }
    try {
        & taskkill /PID $Proceso.Id /T /F *> $null
    } catch {}
}

$backendProc = $null
$frontendProc = $null

function Detener-Servicios {
    Write-Host ""
    Write-Host "Deteniendo backend y frontend..."
    Detener-Proceso $backendProc
    Detener-Proceso $frontendProc
    Start-Sleep -Seconds 1
    # Por si uvicorn --reload dejo algun proceso hijo vivo en el puerto.
    Liberar-Puerto -Puerto $BackendPort
    Liberar-Puerto -Puerto $FrontendPort
    Write-Host "Backend y frontend detenidos. Puertos $BackendPort y $FrontendPort liberados."
}

Write-Host "== PIA - Prediccion de Emociones =="
Write-Host ""

# Por si quedo algo colgado de una corrida anterior sin cerrar bien.
Liberar-Puerto -Puerto $BackendPort
Liberar-Puerto -Puerto $FrontendPort

try {
    Write-Host "Iniciando backend en http://localhost:$BackendPort ..."
    $backendProc = Start-Process -FilePath $PythonBackend `
        -ArgumentList @("-m", "uvicorn", "main:app", "--reload", "--host", $BackendHost, "--port", "$BackendPort") `
        -WorkingDirectory $BackendDir -NoNewWindow -PassThru

    Write-Host "Iniciando frontend en http://localhost:$FrontendPort ..."
    $frontendProc = Start-Process -FilePath "python" `
        -ArgumentList @("-m", "http.server", "$FrontendPort") `
        -WorkingDirectory $FrontendDir -NoNewWindow -PassThru

    Start-Sleep -Seconds 2

    if (-not $backendProc.HasExited) {
        Write-Host "[OK]    Backend activo   (PID $($backendProc.Id)) -> http://localhost:$BackendPort"
    } else {
        Write-Host "[ERROR] El backend no pudo iniciar (revisa el error arriba)."
    }

    if (-not $frontendProc.HasExited) {
        Write-Host "[OK]    Frontend activo  (PID $($frontendProc.Id)) -> http://localhost:$FrontendPort"
    } else {
        Write-Host "[ERROR] El frontend no pudo iniciar (revisa el error arriba)."
    }

    Write-Host ""
    Write-Host ">>> Abre http://localhost:$FrontendPort en tu navegador <<<"
    Write-Host "Presiona Ctrl+C para detener todo y liberar los puertos."
    Write-Host ""

    # Si uno de los dos procesos muere solo (crash), se detiene el otro tambien.
    while ((-not $backendProc.HasExited) -and (-not $frontendProc.HasExited)) {
        Start-Sleep -Seconds 1
    }

    if ($backendProc.HasExited)  { Write-Host "[aviso] El backend se detuvo solo." }
    if ($frontendProc.HasExited) { Write-Host "[aviso] El frontend se detuvo solo." }
}
finally {
    Detener-Servicios
}
