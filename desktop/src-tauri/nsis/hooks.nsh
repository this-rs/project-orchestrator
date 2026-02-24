; Install Visual C++ Redistributable 2015-2022 (x64) if not present.
; Required because ort-sys (ONNX Runtime) is compiled with the dynamic CRT
; and expects MSVCP140.dll / VCRUNTIME140.dll at runtime.
;
; The vc_redist.x64.exe is bundled as a Tauri resource by the CI workflow,
; so no internet access is needed at install time.
;
; We use POSTINSTALL because resource files are only extracted to $INSTDIR
; AFTER the install phase. At PREINSTALL time, the file does not exist yet.

!macro NSIS_HOOK_POSTINSTALL
  ; Check if VC++ 2015-2022 Redistributable (x64) is already installed
  ; Registry key exists when any version of the 14.x runtime is present
  ReadRegDWORD $0 HKLM "SOFTWARE\Microsoft\VisualStudio\14.0\VC\Runtimes\X64" "Installed"
  ${If} $0 != 1
    ; Verify the bundled redistributable exists before attempting install
    ${If} ${FileExists} "$INSTDIR\resources\vc_redist.x64.exe"
      DetailPrint "Installing Visual C++ Redistributable from $INSTDIR\resources\vc_redist.x64.exe..."
      ; /install is REQUIRED for silent mode — without it vc_redist may do nothing
      ; /quiet suppresses all UI, /norestart prevents reboot prompt
      ExecWait '"$INSTDIR\resources\vc_redist.x64.exe" /install /quiet /norestart' $1
      DetailPrint "VC++ Redistributable exited with code $1"
      ${If} $1 == 0
        DetailPrint "VC++ Redistributable installed successfully"
      ${ElseIf} $1 == 1638
        DetailPrint "VC++ Redistributable: newer version already installed"
      ${ElseIf} $1 == 3010
        DetailPrint "VC++ Redistributable installed (reboot recommended)"
      ${Else}
        DetailPrint "VC++ Redistributable exited with code $1 — may need manual install"
      ${EndIf}
      ; Clean up the installer from resources to save ~24 MB disk space
      Delete "$INSTDIR\resources\vc_redist.x64.exe"
    ${Else}
      DetailPrint "WARNING: vc_redist.x64.exe not found in $INSTDIR\resources\ — cannot install VC++ Runtime"
    ${EndIf}
  ${Else}
    DetailPrint "VC++ Redistributable already installed (registry check passed), skipping"
  ${EndIf}
!macroend
