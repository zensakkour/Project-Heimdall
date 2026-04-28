param(
  [Parameter(Mandatory = $true)]
  [ValidateSet("feat", "fix", "chore", "docs", "refactor", "test", "perf", "ci")]
  [string]$Type,

  [Parameter(Mandatory = $true)]
  [string]$Name
)

$slug = $Name.ToLowerInvariant()
$slug = $slug -replace "[^a-z0-9\-]+", "-"
$slug = $slug -replace "-+", "-"
$slug = $slug.Trim("-")

if (-not $slug) {
  throw "Branch name is empty after slugify."
}

$date = Get-Date -Format "yyyyMMdd"
$branch = "$Type/$date-$slug"

git switch -c $branch
$plan = @"
# Branch Plan: `$branch`

## Status
- Branch purpose:
- Completion policy:

## Problem

## Direction
### Stage 1: Compare
- Candidate A:
- Candidate B:
- Candidate C:

### Stage 2: Choose
- Decision gate:

### Stage 3: Next Move
- If the compare phase wins, deepen here:

## Benchmarks and Evidence Rules
- Use leakage-safe evaluation for improvement claims.
- Record major results in `PROGRESS.md`.

## Immediate Next Actions
1.
2.
3.
"@
$plan | Set-Content -Path "plan.md" -Encoding utf8
Write-Host "Switched to $branch"
Write-Host "Created plan.md"
