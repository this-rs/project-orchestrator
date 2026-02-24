; Ensure MSVC runtime DLLs (msvcp140.dll, vcruntime140.dll, vcruntime140_1.dll)
; are available at runtime. Required because ort-sys (ONNX Runtime) is compiled
; with the dynamic CRT (/MD).
;
; Strategy (belt-and-suspenders):
; 1. Try installing vc_redist.x64.exe system-wide (cleanest long-term solution)
; 2. If that fails or DLLs still missing: copy bundled DLLs next to the exe
;    (guaranteed fallback — no admin needed, no external installer)
;
; We use POSTINSTALL because resource files are only extracted to $INSTDIR
; AFTER the install phase. At PREINSTALL time, the files do not exist yet.

!macro NSIS_HOOK_POSTINSTALL
  ; --- Phase 1: Try system-wide VC++ Redistributable install ---
  ReadRegDWORD $0 HKLM "SOFTWARE\Microsoft\VisualStudio\14.0\VC\Runtimes\X64" "Installed"
  ${If} $0 != 1
    ${If} ${FileExists} "$INSTDIR\resources\vc_redist.x64.exe"
      DetailPrint "Installing Visual C++ Redistributable..."
      ExecWait '"$INSTDIR\resources\vc_redist.x64.exe" /install /quiet /norestart' $1
      DetailPrint "VC++ Redistributable exited with code $1"
      ${If} $1 == 0
        DetailPrint "VC++ Redistributable installed successfully"
      ${ElseIf} $1 == 1638
        DetailPrint "VC++ Redistributable: newer version already installed"
      ${ElseIf} $1 == 3010
        DetailPrint "VC++ Redistributable installed (reboot recommended)"
      ${Else}
        DetailPrint "VC++ Redistributable install returned code $1"
      ${EndIf}
    ${Else}
      DetailPrint "vc_redist.x64.exe not found — skipping system-wide install"
    ${EndIf}
  ${Else}
    DetailPrint "VC++ Redistributable already installed (registry check passed)"
  ${EndIf}

  ; --- Phase 2: Ensure DLLs are next to the exe (guaranteed fallback) ---
  ; Even if vc_redist installed successfully, copying local DLLs is harmless
  ; and provides a safety net in case the system install didn't take effect yet.
  ; Windows DLL search order checks the exe directory first, so local DLLs
  ; take precedence and guarantee the app can start.
  ${If} ${FileExists} "$INSTDIR\resources\msvcp140.dll"
    CopyFiles /SILENT "$INSTDIR\resources\msvcp140.dll" "$INSTDIR"
    CopyFiles /SILENT "$INSTDIR\resources\vcruntime140.dll" "$INSTDIR"
    CopyFiles /SILENT "$INSTDIR\resources\vcruntime140_1.dll" "$INSTDIR"
    DetailPrint "Copied MSVC runtime DLLs to application directory"
    ; Clean up from resources to avoid duplicates
    Delete "$INSTDIR\resources\msvcp140.dll"
    Delete "$INSTDIR\resources\vcruntime140.dll"
    Delete "$INSTDIR\resources\vcruntime140_1.dll"
  ${EndIf}

  ; Clean up vc_redist installer to save ~24 MB disk space
  Delete "$INSTDIR\resources\vc_redist.x64.exe"
!macroend
