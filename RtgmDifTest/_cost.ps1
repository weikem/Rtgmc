# Times each flow on the 200-frame 422p clip (400 output frames) so the quality decision has a
# cost next to it. Runs in-process through RtgmDif.dll with a caller-supplied stream, which is
# the same path the real pipeline uses. The scratch output is deleted at the end.
#
# usage: powershell -NoProfile -File e:\github\NVEnc\RtgmDifTest\_cost.ps1

$exe = 'e:\github\NVEnc\RtgmDifTest\build\RtgmDifTest.exe'
$common = @(
    '-i', 'E:\uyvy\src_422p.yuv',
    '--width', '1920', '--height', '1080', '--in-csp', 'yuv422p',
    '--fps', '25', '--frames', '200'
)
$out = 'E:\uyvy\_cost_tmp.yuv'

# Warm-up. The kernels are PTX and get JIT-compiled on first use, so whichever process runs first
# pays for the compilation: without this, faster_nn1 measured 14.3 s against slower's 9.0 s, and
# the position in the list decided the result.
& $exe @common --flow fast_deint -o $out *> $null

$rows = @()
foreach ($flow in @('faster_nn1', 'slower', 'fast_deint', 'fast_both', 'fast_opt')) {
    $t = Measure-Command { & $exe @common --flow $flow -o $out *> $null }
    $rows += [pscustomobject]@{ flow = $flow; seconds = [math]::Round($t.TotalSeconds, 1) }
}
$rows | Format-Table -AutoSize

Remove-Item $out -ErrorAction SilentlyContinue
