# Fix the MSVC dependency-capture prefix in a generated rules.ninja.
#
# Why this exists (measured 2026-10-12, VS 18 / MSVC 19.38.33145 + CMake 4.3.1):
#   cl.exe emits its /showIncludes lines as UTF-8 bytes, but CMake decodes them using the local
#   ANSI code page, then writes the resulting (already corrupted) string back as UTF-8. The bytes
#   stored in rules.ninja therefore never match the bytes cl actually prints, ninja matches no
#   dependency line, every object ends up with an empty dep file, and incremental builds link
#   objects built from different header generations -- the symptom is a startup crash
#   (0xC0000409 / heap corruption) with otherwise all-green signals.
#
#   VSLANG=1033, PreferredUILang=en-US and chcp 65001 were all measured to have NO effect on this
#   toolchain, so forcing ASCII output is not an option.
#
# This script rewrites only the bytes of that one value, through ReadAllBytes/WriteAllBytes. It must
# not be done as a text read/write: rewriting line endings would corrupt rules.ninja/build.ninja.
#
# No non-ASCII literal appears below on purpose: a BOM-less .ps1 is read as ANSI by a PS 5.1 host.
# The 22 bytes below are the UTF-8 encoding of the prefix cl.exe prints on a zh-CN host, namely
# U+6CE8 U+610F U+003A U+0020 U+5305 U+542B U+6587 U+4EF6 U+003A U+0020 ("Note: including file: ").

param(
    [Parameter(Mandatory = $true)][string]$RulesPath
)

$ErrorActionPreference = 'Stop'

if (-not (Test-Path $RulesPath)) { throw "rules.ninja not found: $RulesPath" }

function Find-Bytes([byte[]]$hay, [byte[]]$needle) {
    for ($i = 0; $i -le $hay.Length - $needle.Length; $i++) {
        $ok = $true
        for ($j = 0; $j -lt $needle.Length; $j++) {
            if ($hay[$i + $j] -ne $needle[$j]) { $ok = $false; break }
        }
        if ($ok) { return $i }
    }
    return -1
}

# UTF-8 bytes of the prefix cl.exe actually prints. One trailing space covers every include depth,
# because ninja trims the remainder of the line after matching the prefix.
$newValue = [byte[]](
    0xE6, 0xB3, 0xA8, 0xE6, 0x84, 0x8F, 0x3A, 0x20,
    0xE5, 0x8C, 0x85, 0xE5, 0x90, 0xAB, 0xE6, 0x96, 0x87, 0xE4, 0xBB, 0xB6, 0x3A, 0x20
)

$key = [System.Text.Encoding]::ASCII.GetBytes('msvc_deps_prefix = ')
$raw = [System.IO.File]::ReadAllBytes($RulesPath)
$pos = Find-Bytes $raw $key
if ($pos -lt 0) { throw "msvc_deps_prefix not found in $RulesPath" }

$valueStart = $pos + $key.Length
$eol = -1
for ($i = $valueStart; $i -lt $raw.Length - 1; $i++) {
    if ($raw[$i] -eq 13 -and $raw[$i + 1] -eq 10) { $eol = $i; break }
}
if ($eol -lt 0) { throw "line ending not found after msvc_deps_prefix" }

$old = $raw[$valueStart..($eol - 1)]
$already = ($old.Length -eq $newValue.Length)
if ($already) {
    for ($i = 0; $i -lt $newValue.Length; $i++) {
        if ($old[$i] -ne $newValue[$i]) { $already = $false; break }
    }
}
if ($already) {
    Write-Output "N4_PATCH=already-correct"
    exit 0
}

$out = [System.Collections.Generic.List[byte]]::new()
$out.AddRange([byte[]]$raw[0..($valueStart - 1)])
$out.AddRange([byte[]]$newValue)
$out.AddRange([byte[]]$raw[$eol..($raw.Length - 1)])
[System.IO.File]::WriteAllBytes($RulesPath, $out.ToArray())

Write-Output ("N4_PATCH=rewritten old=" + (($old | ForEach-Object { $_.ToString('X2') }) -join '') +
              " new=" + (($newValue | ForEach-Object { $_.ToString('X2') }) -join '') +
              " size=" + $raw.Length + "->" + ([System.IO.File]::ReadAllBytes($RulesPath)).Length)
