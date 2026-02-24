; Ensure ALL MSVC runtime DLLs are available at runtime.
; ort-sys (ONNX Runtime) is compiled with the dynamic CRT (/MD) and needs
; msvcp140.dll, msvcp140_1.dll, vcruntime140.dll, vcruntime140_1.dll, etc.
;
; Strategy (belt-and-suspenders):
; 1. ALWAYS run vc_redist.x64.exe (idempotent — handles already-installed)
; 2. Copy ALL bundled *.dll from resources/ next to the exe (guaranteed fallback)
;
; We use POSTINSTALL because resource files are only extracted to $INSTDIR
; AFTER the install phase. At PREINSTALL time, the files do not exist yet.

!macro NSIS_HOOK_POSTINSTALL
  ; --- Phase 1: Always run VC++ Redistributable (idempotent) ---
  ; Don't skip based on registry — partial/corrupt installs can leave the
  ; registry key set while DLLs are missing. vc_redist handles all cases:
  ;   exit 0    = installed successfully
  ;   exit 1638 = newer version already installed (no-op)
  ;   exit 3010 = installed, reboot recommended
  ${If} ${FileExists} "$INSTDIR\resources\vc_redist.x64.exe"
    DetailPrint "Installing Visual C++ Redistributable..."
    ExecWait '"$INSTDIR\resources\vc_redist.x64.exe" /install /quiet /norestart' $1
    DetailPrint "VC++ Redistributable exited with code $1"
  ${Else}
    DetailPrint "vc_redist.x64.exe not found — skipping system-wide install"
  ${EndIf}

  ; --- Phase 2: Copy ALL bundled CRT DLLs next to the exe ---
  ; Even if vc_redist installed successfully, local DLLs are a safety net.
  ; Windows DLL search order checks the exe directory first, so local DLLs
  ; take precedence and guarantee the app can start.
  ;
  ; The CI bundles ALL DLLs from the VS 2022 CRT directory (~12 files):
  ;   concrt140.dll, msvcp140.dll, msvcp140_1.dll, msvcp140_2.dll,
  ;   msvcp140_atomic_wait.dll, msvcp140_codecvt_ids.dll,
  ;   vcamp140.dll, vccorlib140.dll, vcomp140.dll,
  ;   vcruntime140.dll, vcruntime140_1.dll, vcruntime140_threads.dll

  FindFirst $0 $1 "$INSTDIR\resources\*.dll"
  _copy_dll_loop:
    StrCmp $1 "" _copy_dll_done
    CopyFiles /SILENT "$INSTDIR\resources\$1" "$INSTDIR"
    Delete "$INSTDIR\resources\$1"
    FindNext $0 $1
    Goto _copy_dll_loop
  _copy_dll_done:
  FindClose $0
  DetailPrint "Copied MSVC runtime DLLs to application directory"

  ; Clean up vc_redist installer to save ~24 MB disk space
  Delete "$INSTDIR\resources\vc_redist.x64.exe"
!macroend
