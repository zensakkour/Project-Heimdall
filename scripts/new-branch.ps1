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
Write-Host "Switched to $branch"
