param([Parameter(Mandatory=$true)][string]$OutputFile)
$ErrorActionPreference = 'Stop'
Add-Type -AssemblyName System.Drawing
$taskBitmap = New-Object System.Drawing.Bitmap(640,220,[System.Drawing.Imaging.PixelFormat]::Format24bppRgb)
$taskGraphics = [System.Drawing.Graphics]::FromImage($taskBitmap)
$taskTitle = New-Object System.Drawing.Font('Microsoft YaHei',24,[System.Drawing.FontStyle]::Bold)
$taskBody = New-Object System.Drawing.Font('Microsoft YaHei',15)
$taskNote = New-Object System.Drawing.Font('Microsoft YaHei',11)
$taskInk = New-Object System.Drawing.SolidBrush([System.Drawing.ColorTranslator]::FromHtml('#28594c'))
$taskMuted = New-Object System.Drawing.SolidBrush([System.Drawing.ColorTranslator]::FromHtml('#748575'))
try {
    $taskGraphics.Clear([System.Drawing.ColorTranslator]::FromHtml('#f4f5ee'))
    $taskGraphics.TextRenderingHint = [System.Drawing.Text.TextRenderingHint]::AntiAliasGridFit
    $taskGraphics.DrawString('棋局复刻',$taskTitle,$taskInk,36,28)
    $taskGraphics.DrawString('正在准备本地运行环境',$taskBody,$taskInk,38,100)
    $taskGraphics.DrawString('正在展开内置模型，请稍候…',$taskNote,$taskMuted,38,150)
    $taskBitmap.Save([IO.Path]::GetFullPath($OutputFile),[System.Drawing.Imaging.ImageFormat]::Bmp)
} finally {
    $taskGraphics.Dispose(); $taskBitmap.Dispose()
    $taskTitle.Dispose(); $taskBody.Dispose(); $taskNote.Dispose()
    $taskInk.Dispose(); $taskMuted.Dispose()
}
