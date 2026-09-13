param(
    [Parameter(Mandatory = $true)]
    [string]$DocxPath,
    [Parameter(Mandatory = $true)]
    [string]$PdfPath
)

$ErrorActionPreference = 'Stop'
$resolvedDocx = (Resolve-Path -LiteralPath $DocxPath).Path
$resolvedPdf = [System.IO.Path]::GetFullPath($PdfPath)
$pdfDir = [System.IO.Path]::GetDirectoryName($resolvedPdf)
if (-not [System.IO.Directory]::Exists($pdfDir)) {
    [System.IO.Directory]::CreateDirectory($pdfDir) | Out-Null
}

$word = $null
$doc = $null
$stage = 'start'
try {
    $stage = 'create-word'
    $word = New-Object -ComObject Word.Application
    $word.Visible = $false
    $word.DisplayAlerts = 0
    $stage = 'open-document'
    $doc = $word.Documents.Open($resolvedDocx, $false, $false)

    for ($i = $doc.Paragraphs.Count; $i -ge 1; $i--) {
        $para = $doc.Paragraphs.Item($i)
        $raw = $para.Range.Text.Trim([char]13, [char]7)
        if (-not $raw.StartsWith('EQ::')) {
            continue
        }

        $body = $raw.Substring(4)
        $parts = $body -split "`t", 2
        $formula = $parts[0]
        $numberPart = $null
        if ($parts.Count -gt 1) {
            $numberPart = $parts[1]
        }

        $formula = $formula.Replace('Delta', [string][char]0x0394)
        $formula = $formula.Replace('eta', [string][char]0x03B7)
        $formula = $formula.Replace('epsilon', [string][char]0x03B5)
        $formula = $formula.Replace('omega', [string][char]0x03C9)
        $formula = $formula.Replace('tau', [string][char]0x03C4)
        $formula = $formula.Replace('rho', [string][char]0x03C1)
        $formula = $formula.Replace('mu', [string][char]0x03BC)
        $formula = $formula.Replace('alpha', [string][char]0x03B1)
        $formula = $formula.Replace('nu', [string][char]0x03BD)
        $formula = $formula.Replace('sum_', ([string][char]0x2211) + '_')
        $formula = $formula.Replace('<=', [string][char]0x2264)
        $formula = $formula.Replace('>=', [string][char]0x2265)
        $formula = $formula.Replace('...', [string][char]0x2026)

        $replacement = $formula
        if ($null -ne $numberPart) {
            $replacement += "`t$numberPart"
        }
        $replacement += "`r"
        $para.Range.Text = $replacement

        $stage = "equation-$i"
        try {
            $eqRange = $doc.Range($para.Range.Start, $para.Range.Start + $formula.Length)
            $eqRange.Font.Name = 'Cambria Math'
            $eqRange.Font.NameFarEast = 'Cambria Math'
            $math = $doc.OMaths.Add($eqRange)
            $math.BuildUp()
            $para.Alignment = 1
        }
        catch {
            Write-Warning "Equation conversion failed at paragraph $i : $($_.Exception.Message)"
        }
    }

    $stage = 'update-fields'
    foreach ($field in $doc.Fields) {
        [void]$field.Update()
    }
    $stage = 'update-toc'
    foreach ($toc in $doc.TablesOfContents) {
        [void]$toc.Update()
    }

    $stage = 'save-document'
    $doc.Save()
    $stage = 'export-pdf'
    $doc.ExportAsFixedFormat($resolvedPdf, 17)
    $doc.Close($false)
    $doc = $null
    $word.Quit()
    $word = $null
    Write-Output $resolvedDocx
    Write-Output $resolvedPdf
}
catch {
    Write-Error "Failure at stage $stage : $($_.Exception.Message)"
}
finally {
    if ($null -ne $doc) {
        $doc.Close($false)
    }
    if ($null -ne $word) {
        $word.Quit()
    }
    [System.GC]::Collect()
    [System.GC]::WaitForPendingFinalizers()
}
