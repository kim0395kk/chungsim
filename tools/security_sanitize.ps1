param(
  [switch]$CleanLogs
)

$ErrorActionPreference = "SilentlyContinue"

Write-Host "=== Security Sanitize: Secret Pattern Scan ==="

$patterns = @(
  'AIza[0-9A-Za-z_-]{20,}',
  'gsk_[A-Za-z0-9_]{20,}',
  'sb_publishable_[A-Za-z0-9._-]{20,}',
  'sb_secret_[A-Za-z0-9._-]{20,}',
  'GEMINI_API_KEY\s*=\s*"[^"]+"',
  'GROQ_API_KEY\s*=\s*"[^"]+"',
  'SUPABASE_KEY\s*=\s*"[^"]+"'
)

$files = Get-ChildItem -Recurse -File -Force | Where-Object { $_.FullName -notmatch '\\.git\\' }
$hits = @()
foreach ($f in $files) {
  $txt = Get-Content -Raw -Path $f.FullName
  foreach ($p in $patterns) {
    if ([regex]::IsMatch($txt, $p)) {
      $hits += [PSCustomObject]@{
        File = $f.FullName
        Pattern = $p
      }
      break
    }
  }
}

if ($hits.Count -eq 0) {
  Write-Host "WORKTREE_SECRET_HITS=0"
} else {
  Write-Host ("WORKTREE_SECRET_HITS={0}" -f $hits.Count)
  $hits | Select-Object -ExpandProperty File -Unique | ForEach-Object { Write-Host $_ }
}

Write-Host ""
Write-Host "=== Git History Secret Scan (quick) ==="
$commits = git rev-list --all
$found = New-Object System.Collections.Generic.HashSet[string]
foreach ($c in $commits) {
  foreach ($p in $patterns[0..3]) {
    $out = git grep -I -l -E -- "$p" $c -- 2>$null
    if ($LASTEXITCODE -eq 0 -and $out) {
      foreach ($line in $out) {
        [void]$found.Add($line)
      }
    }
  }
}
Write-Host ("GIT_HISTORY_SECRET_HITS={0}" -f $found.Count)
if ($found.Count -gt 0) {
  $found | Select-Object -First 50 | ForEach-Object { Write-Host $_ }
}

if ($CleanLogs) {
  Write-Host ""
  Write-Host "=== Cleaning local log files ==="
  $logFiles = Get-ChildItem -Recurse -File -Include *.log,*run*.txt,*run*.out,*run*.err
  foreach ($lf in $logFiles) {
    Remove-Item -Force $lf.FullName
    Write-Host ("Removed: {0}" -f $lf.FullName)
  }
}

Write-Host ""
Write-Host "Done."

