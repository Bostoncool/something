# update_gitignore.ps1
# Find files larger than 100MB and add to .gitignore

$MAX_SIZE = 100MB
$GITIGNORE = ".gitignore"

Write-Host "Searching for files larger than 100MB..." -ForegroundColor Yellow

# Find files larger than 100MB
$largeFiles = Get-ChildItem -Path . -Recurse -File -ErrorAction SilentlyContinue | 
    Where-Object { $_.Length -gt $MAX_SIZE }

if ($largeFiles) {
    # Read existing .gitignore content if exists
    $existingIgnores = @()
    if (Test-Path $GITIGNORE) {
        $existingIgnores = Get-Content $GITIGNORE
    }

    foreach ($file in $largeFiles) {
        # Get relative path with forward slashes (Git style)
        $relativePath = $file.FullName.Replace($PWD.Path + "\", "").Replace("\", "/")
        
        # Check if already in .gitignore
        if ($existingIgnores -notcontains $relativePath) {
            Add-Content -Path $GITIGNORE -Value $relativePath
            Write-Host "Added $relativePath to .gitignore" -ForegroundColor Green
        } else {
            Write-Host "$relativePath already in .gitignore" -ForegroundColor Gray
        }
    }

    # Remove ignored large files from Git cache
    Write-Host ""
    Write-Host "Removing large files from Git cache..." -ForegroundColor Yellow
    
    $ignoredFiles = git ls-files -i --exclude-from=.gitignore 2>$null
    if ($ignoredFiles) {
        git rm --cached $ignoredFiles 2>$null
        Write-Host "Removed files from Git cache" -ForegroundColor Green
    } else {
        Write-Host "No files to remove from Git cache" -ForegroundColor Gray
    }
} else {
    Write-Host "No files larger than 100MB found" -ForegroundColor Green
}

Write-Host ""
Write-Host "Done!" -ForegroundColor Cyan
