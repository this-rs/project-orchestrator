; Install Visual C++ Redistributable 2015-2022 (x64) if not present.
; Required because ort-sys (ONNX Runtime) is compiled with the dynamic CRT
; and expects MSVCP140.dll / VCRUNTIME140.dll at runtime.
;
; The vc_redist.x64.exe is bundled as a Tauri resource by the CI workflow,
; so no internet access is needed at install time.
;
; IMPORTANT: We use POSTINSTALL (not PREINSTALL) because the resource files
; are only extracted to $INSTDIR AFTER the install phase. At PREINSTALL time,
; $INSTDIR/resources/vc_redist.x64.exe does not exist yet.

!macro NSIS_HOOK_POSTINSTALL
  ; Check if VC++ 2015-2022 Redistributable (x64) is already installed
  ; Registry key exists when any version of the 14.x runtime is present
  ReadRegDWORD $0 HKLM "SOFTWARE\Microsoft\VisualStudio\14.0\VC\Runtimes\X64" "Installed"
  ${If} $0 != 1
    DetailPrint "Installing Visual C++ Redistributable..."
    ; vc_redist.x64.exe is placed in the resources dir by CI
    ExecWait '"$INSTDIR\resources\vc_redist.x64.exe" /quiet /norestart' $1
    ${If} $1 != 0
      DetailPrint "VC++ Redistributable install exited with code $1 (may already be installed)"
    ${Else}
      DetailPrint "VC++ Redistributable installed successfully"
    ${EndIf}
  ${Else}
    DetailPrint "VC++ Redistributable already installed, skipping"
  ${EndIf}
!macroend
