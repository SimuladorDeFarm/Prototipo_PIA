<#
.SYNOPSIS
    Instala todo lo necesario para correr PIA desde cero en Windows.

.DESCRIPTION
    Crea el entorno virtual del backend, instala las dependencias y
    descarga los checkpoints de los 3 modelos (voz, rostro, texto) desde
    Hugging Face Hub. Equivalente Windows de install.sh.

.NOTES
    Si Windows bloquea la ejecucion de scripts (.ps1 deshabilitados por
    politica), corre una vez en PowerShell (sin permisos de administrador):

        Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned

    Uso:
        .\install.ps1

    Despues, para levantar el proyecto: .\iniciar.ps1
#>

$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$BackendDir  = Join-Path $ProjectRoot "backend"
$VenvDir     = Join-Path $BackendDir ".venv"
$PythonVenv  = Join-Path $VenvDir "Scripts\python.exe"

Write-Host "== PIA - Instalacion =="
Write-Host ""

$pythonCmd = Get-Command python -ErrorAction SilentlyContinue
if (-not $pythonCmd) {
    Write-Host "[ERROR] No se encontro 'python' en el PATH. Instala Python 3.11+ (marcando 'Add to PATH' en el instalador) y vuelve a correr este script."
    exit 1
}
Write-Host "Python detectado: $(python --version)"

if (Test-Path $PythonVenv) {
    Write-Host "[OK] El entorno virtual ya existe en $VenvDir (no se vuelve a crear)."
} else {
    Write-Host "Creando entorno virtual en $VenvDir ..."
    python -m venv $VenvDir
    if (-not (Test-Path $PythonVenv)) {
        Write-Host "[ERROR] No se pudo crear el entorno virtual."
        exit 1
    }
}

Write-Host ""
Write-Host "Instalando dependencias (backend/requirements.txt) ..."
& $PythonVenv -m pip install --upgrade pip
& $PythonVenv -m pip install -r (Join-Path $BackendDir "requirements.txt")
if ($LASTEXITCODE -ne 0) {
    Write-Host "[ERROR] Fallo la instalacion de dependencias. Revisa el error de arriba y vuelve a correr .\install.ps1."
    exit 1
}
Write-Host "[OK] Dependencias instaladas."

Write-Host ""
Write-Host "Descargando modelos (voz, rostro, texto) desde Hugging Face Hub ..."
Push-Location $BackendDir
try {
    & $PythonVenv -m models.descargar_modelos
} finally {
    Pop-Location
}

Write-Host ""
Write-Host "== Instalacion terminada =="
Write-Host "Si algun modelo no se pudo descargar, revisa el aviso de arriba: te da el link"
Write-Host "para bajarlo a mano (ver tambien README -> Modelos)."
Write-Host ""
Write-Host "Para levantar el backend y el frontend juntos: .\iniciar.ps1"
