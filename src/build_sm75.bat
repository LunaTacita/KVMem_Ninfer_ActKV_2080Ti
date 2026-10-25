@echo off
REM ===========================================================================
REM  NInfer sm_75 build entry  --  baseline: kvmem/kvmem 0.18 (ninfer backend)
REM  ASCII only on purpose: a zh-CN locale misreads non-BOM UTF-8 .bat files.
REM  Pinned toolchain, why each pin exists:
REM   - MSVC 14.38.33130 : this host also has 14.51.36231, and CUDA 12.6's
REM     cudafe++ crashes (0xC0000005) while probing the 14.51 toolset.
REM   - CUDA 12.6        : the only toolkit installed and the one proven here.
REM   - VCPKG_ROOT/TRIPLET: vcvars64 of VS 2026 OVERWRITES VCPKG_ROOT with the
REM     VS-bundled vcpkg, whose x64-windows triplet has no ffmpeg. Both must be
REM     set AFTER the vcvars call. Without this, configure dies inside CMake
REM     4.3's FindPackageHandleStandardArgs with a bogus
REM     "Invalid character escape '\M'" syntax error that hides the real cause.
REM   - CMAKE_MAKE_PROGRAM : otherwise CMake may pick D:\Python\...\ninja.exe.
REM   - NINFER_KVMEM_SOURCE_DIR : the CMake default resolves to E:\AI\kvmem,
REM     which does not exist, and CMake then raises FATAL_ERROR.
REM   - CMAKE_CUDA_ARCHITECTURES=75-real : defines NINFER_SM75 for the whole
REM     tree; "-real" emits sm_75 SASS only (no PTX), as the previous stack did.
REM  N4 deps-prefix: cl.exe prints /showIncludes lines as UTF-8 but CMake decodes
REM  them as ANSI, so rules.ninja never matches and every dep file ends up empty;
REM  incremental builds then link objects from different header generations and
REM  the binary crashes at startup with all-green signals. VSLANG=1033,
REM  PreferredUILang and chcp 65001 were all measured to have NO effect, so the
REM  one value is rewritten at byte level. CMake's Ninja generator writes that
REM  file as <build>/CMakeFiles/rules.ninja, not <build>/rules.ninja.
REM  NEVER kill a running build: it corrupts .ninja_deps (trap #32 / N32 family).
REM  Usage:  build_sm75.bat            -> configure + deps-prefix fix
REM          build_sm75.bat apps       -> build the four product executables
REM          build_sm75.bat full       -> build everything, -k 0 (keep going)
REM ===========================================================================
setlocal
set "VS=D:\Microsoft Visual Studio\18\Community"
set "VCVARS=%VS%\VC\Auxiliary\Build\vcvars64.bat"
set "CMAKE=%VS%\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe"
set "NINJA=%VS%\Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja\ninja.exe"
set "MSVCBIN=%VS%\VC\Tools\MSVC\14.38.33130\bin\Hostx64\x64"
set "CUDA=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6"
set "SRC=E:\AI\ninfer-fusion\kvmem_src"
set "BUILD=E:\AI\ninfer-fusion\kvmem_build"
set "RULES=%BUILD%\CMakeFiles\rules.ninja"
set "FIXDEPS=%SRC%\fix_msvc_deps_prefix.ps1"
set "KVMEMCORE=E:/AI/kvmem-0.18/kvmem"

call "%VCVARS%" -vcvars_ver=14.38.33130 >nul 2>&1
set "VSLANG=1033"
set "VCPKG_ROOT=E:\AI\ninfer-fusion\deps\vcpkg"
set "VCPKG_TARGET_TRIPLET=x64-windows"
set "CUDACXX=%CUDA%\bin\nvcc.exe"
set "PATH=%CUDA%\bin;%PATH%"
if not exist "%BUILD%" mkdir "%BUILD%"

REM Idempotent deps-prefix guard: it must hold before ANY compile, so it also
REM runs ahead of apps/full, not only after a configure.
if exist "%RULES%" powershell -NoProfile -ExecutionPolicy Bypass -File "%FIXDEPS%" -RulesPath "%RULES%" > "%BUILD%\fix_deps_prefix.log" 2>&1

REM -k 0 keeps ninja going, so one expensive pass reports EVERY broken target
REM instead of stopping at the first one (upstream does the same in build_merge.bat).
if /I "%~1"=="apps" "%CMAKE%" --build "%BUILD%" --target ninfer ninfer-serve ninfer-perplexity ninfer-calibrate --parallel -- -k 0 > "%BUILD%\build_apps.log" 2>&1
if /I "%~1"=="apps" echo BUILD_APPS_EXIT=%ERRORLEVEL%
if /I "%~1"=="apps" exit /b %ERRORLEVEL%

if /I "%~1"=="full" "%CMAKE%" --build "%BUILD%" --parallel -- -k 0 > "%BUILD%\build_full.log" 2>&1
if /I "%~1"=="full" echo BUILD_FULL_EXIT=%ERRORLEVEL%
if /I "%~1"=="full" exit /b %ERRORLEVEL%

"%CMAKE%" -S "%SRC%" -B "%BUILD%" -G Ninja -DCMAKE_MAKE_PROGRAM="%NINJA%" -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_COMPILER="%MSVCBIN%\cl.exe" -DCMAKE_CXX_COMPILER="%MSVCBIN%\cl.exe" -DCMAKE_CUDA_COMPILER="%CUDA%\bin\nvcc.exe" -DCMAKE_CUDA_ARCHITECTURES=75-real -DNINFER_KVMEM_SOURCE_DIR=%KVMEMCORE% -DNINFER_BUILD_APPS=ON -DNINFER_BUILD_TOOLS=OFF -DNINFER_BUILD_BENCHMARKS=OFF -DBUILD_TESTING=ON > "%BUILD%\configure.log" 2>&1
set "CONF=%ERRORLEVEL%"
echo CONFIGURE_EXIT=%CONF%
if not "%CONF%"=="0" exit /b %CONF%

if not exist "%RULES%" echo RULES_NINJA_MISSING & exit /b 1
powershell -NoProfile -ExecutionPolicy Bypass -File "%FIXDEPS%" -RulesPath "%RULES%" > "%BUILD%\fix_deps_prefix.log" 2>&1
echo DEPS_PREFIX_FIX_EXIT=%ERRORLEVEL%
type "%BUILD%\fix_deps_prefix.log"
exit /b 0
