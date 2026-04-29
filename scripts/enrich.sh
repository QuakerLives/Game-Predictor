#!/usr/bin/env bash
# Enrichment runner — Windows equivalent of the Mac caffeinate version.
# Usage: ./scripts/enrich.sh --max-duration 8h

set -e

MAX_DURATION="4h"

while [[ $# -gt 0 ]]; do
    case $1 in
        --max-duration) MAX_DURATION="$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

echo "Enrichment run: max-duration=${MAX_DURATION}"

# Prevent Windows from sleeping during the run (requires no elevation).
powershell.exe -NoProfile -Command "
\$code = @'
using System;
using System.Runtime.InteropServices;
public class Sleep {
    [DllImport(\"kernel32.dll\")] public static extern uint SetThreadExecutionState(uint esFlags);
}
'@
Add-Type -TypeDefinition \$code
[Sleep]::SetThreadExecutionState(0x80000003) | Out-Null
Write-Host 'Sleep prevention active'
" 2>/dev/null || echo "(sleep prevention unavailable — set your display/sleep timeout manually)"

uv run game-predictor-run --enrich-only --max-duration "$MAX_DURATION"

# Re-allow sleep.
powershell.exe -NoProfile -Command "
\$code = @'
using System;
using System.Runtime.InteropServices;
public class Sleep {
    [DllImport(\"kernel32.dll\")] public static extern uint SetThreadExecutionState(uint esFlags);
}
'@
Add-Type -TypeDefinition \$code
[Sleep]::SetThreadExecutionState(0x80000000) | Out-Null
Write-Host 'Sleep prevention released'
" 2>/dev/null || true

echo "Enrichment complete."
