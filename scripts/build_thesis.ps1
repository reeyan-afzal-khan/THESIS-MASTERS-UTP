<#
.SYNOPSIS
    Compile the thesis: Thesis\main.tex -> Thesis\main.pdf.

.DESCRIPTION
    pdfLaTeX -> biber -> pdfLaTeX -> pdfLaTeX, run inside Thesis\ - the same
    steps as the LaTeX Workshop recipe in .vscode\settings.json.

    The build is in place on purpose: biblatex always reads a main.bbl that
    sits next to main.tex, so building into another folder would keep using
    the shipped main.bbl and silently ignore edits to ref.bib.  Every build
    artefact is git-ignored, and while ref.bib is unchanged biber regenerates
    main.bbl byte-for-byte, so a build leaves the tracked sources untouched.

.EXAMPLE
    .\scripts\build_thesis.ps1
#>
$miktex = "$env:LOCALAPPDATA\Programs\MiKTeX\miktex\bin\x64"
if (-not (Get-Command pdflatex -ErrorAction SilentlyContinue) -and (Test-Path $miktex)) {
    $env:Path = "$miktex;$env:Path"            # MiKTeX installed after this shell started
}
if (-not (Get-Command pdflatex -ErrorAction SilentlyContinue)) {
    throw "pdflatex not found - install MiKTeX first:  winget install MiKTeX.MiKTeX"
}

Push-Location (Join-Path (Split-Path -Parent $PSScriptRoot) "Thesis")
try {
    $latexArgs = "-interaction=nonstopmode", "-halt-on-error", "-file-line-error", "-synctex=1", "main.tex"
    foreach ($step in "pdflatex", "biber", "pdflatex", "pdflatex") {
        Write-Host "==> $step"
        if ($step -eq "biber") { biber main | Out-Null } else { pdflatex @latexArgs | Out-Null }
        if ($LASTEXITCODE) {
            $log = if ($step -eq "biber") { "main.blg" } else { "main.log" }
            Select-String -Path $log -Pattern "^!|^\S+:\d+: |ERROR" | Select-Object -First 5 | ForEach-Object { $_.Line }
            throw "$step failed - see Thesis\$log"
        }
    }
    $undefined = Select-String -Path main.log -Pattern "There were undefined (references|citations)"
    if ($undefined) { Write-Warning "main.log reports undefined references or citations" }
    $pages = (Select-String -Path main.log -Pattern "Output written on main\.pdf \((\d+) pages").Matches.Groups[1].Value
    Write-Host "`nBuilt Thesis\main.pdf ($pages pages)"
}
finally {
    Pop-Location
}
