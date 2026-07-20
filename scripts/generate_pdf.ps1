# PowerShell script to convert README_FULL.md to PDF using pandoc
# Requires: pandoc installed and accessible in PATH

$input = "README_FULL.md"
$output = "README_FULL.pdf"

if (!(Get-Command pandoc -ErrorAction SilentlyContinue)) {
    Write-Error "pandoc not found. Install pandoc: https://pandoc.org/installing.html"
    exit 1
}

pandoc $input -o $output
if ($LASTEXITCODE -eq 0) {
    Write-Host "PDF generated: $output"
} else {
    Write-Error "pandoc failed (exit $LASTEXITCODE)"
}
