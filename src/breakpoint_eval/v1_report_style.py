"""Stylesheet for the v0.1 report: plain CSS, system fonts, light/dark/print/mobile."""

CSS = """
:root{
  --bg:#F8FAFC;--fg:#111827;--muted:#4B5563;--rule:#CBD5E1;--panel:#FFFFFF;
  --baseline:#2563EB;--stress:#B45309;
  --baseline-tint:rgba(37,99,235,.14);--stress-tint:rgba(180,83,9,.16);
  --sans:system-ui,-apple-system,"Segoe UI",Roboto,"Helvetica Neue",Arial,sans-serif;
  --mono:ui-monospace,"SF Mono","Cascadia Mono",Menlo,Consolas,"Liberation Mono",monospace;
}
@media (prefers-color-scheme:dark){
  :root{--bg:#111827;--fg:#F8FAFC;--muted:#9CA3AF;--rule:#374151;--panel:#18202F;
    --baseline-text:#93C5FD;--stress-text:#FBBF24}
}
*{box-sizing:border-box}
html{-webkit-text-size-adjust:100%}
body{margin:0;background:var(--bg);color:var(--fg);font-family:var(--sans);line-height:1.6;
  font-size:1rem}
main{max-width:64rem;margin:0 auto;padding:3rem 1.25rem 4rem}
header{margin-bottom:2.5rem}
h1{font-size:2rem;line-height:1.2;margin:.25rem 0 1rem;font-weight:650}
h2{font-size:1.35rem;margin:2.5rem 0 .75rem;font-weight:650}
h3{font-size:1.05rem;margin:2rem 0 .5rem;font-weight:650}
p,ul{margin:.75rem 0}
.eyebrow{font-family:var(--mono);font-size:.8rem;letter-spacing:.04em;text-transform:uppercase;
  color:var(--muted);margin:0}
.lead{font-size:1.1rem;max-width:46rem}
.facts{font-family:var(--mono);font-size:.85rem;color:var(--muted)}
code,.num{font-family:var(--mono);font-size:.88em}
code{display:inline}
.unit{color:var(--muted)}
ul.plain{padding-left:1.25rem}
ul.plain li{margin:.4rem 0}
ul.caveats li{padding-left:.25rem}
a{color:var(--baseline)}
@media (prefers-color-scheme:dark){a{color:var(--baseline-text)}}
.callout{border-left:4px solid var(--stress);background:var(--stress-tint);padding:.75rem 1rem;
  border-radius:0 .25rem .25rem 0}
.callout.derived{border-left-color:var(--baseline);background:var(--baseline-tint)}
.scroll{overflow-x:auto;margin:1rem 0}
table{border-collapse:collapse;width:100%;font-size:.92rem}
caption{text-align:left;color:var(--muted);padding:.25rem 0 .5rem;font-size:.85rem}
th,td{padding:.5rem .6rem;border-bottom:1px solid var(--rule);text-align:left;vertical-align:top}
thead th{font-weight:600;border-bottom:2px solid var(--fg)}
td.num{text-align:right;white-space:nowrap}
th code{display:block;color:var(--muted);font-weight:400}
tr.baseline th,tr.baseline td{background:var(--baseline-tint)}
.mark.baseline{color:var(--baseline)}
.mark.stress{color:var(--stress)}
@media (prefers-color-scheme:dark){
  .mark.baseline{color:var(--baseline-text)}.mark.stress{color:var(--stress-text)}}
img{display:block;max-width:100%;height:auto;margin:1.25rem 0}
table.matrix{width:auto;font-family:var(--mono);font-size:.8rem}
table.matrix th,table.matrix td{padding:.2rem .3rem;text-align:center;border-bottom:1px solid var(--rule)}
table.matrix th[scope=row]{text-align:left;white-space:nowrap;padding-right:.8rem}
table.matrix td.lost{background:var(--stress-tint);color:var(--stress);font-weight:700}
table.matrix td.gained{background:var(--baseline-tint);color:var(--baseline);font-weight:700}
@media (prefers-color-scheme:dark){
  table.matrix td.lost{color:var(--stress-text)}table.matrix td.gained{color:var(--baseline-text)}}
.legend{font-size:.85rem;color:var(--muted)}
.legend .lost{color:var(--stress)}.legend .gained{color:var(--baseline)}
@media (prefers-color-scheme:dark){
  .legend .lost{color:var(--stress-text)}.legend .gained{color:var(--baseline-text)}}
.tabs{margin-top:2.5rem}
.tabs>input{position:absolute;opacity:0;pointer-events:none}
.tablist{display:flex;gap:.25rem;overflow-x:auto;border-bottom:1px solid var(--rule)}
.tablist label{padding:.6rem .9rem;cursor:pointer;white-space:nowrap;border-bottom:3px solid transparent;
  color:var(--muted)}
.panel{display:none;padding-top:.5rem}
details{border-top:1px solid var(--rule);padding:.75rem 0}
details:last-of-type{border-bottom:1px solid var(--rule)}
summary{cursor:pointer;font-weight:600}
dl{display:grid;grid-template-columns:minmax(10rem,16rem) 1fr;gap:.4rem 1.25rem;margin:1rem 0}
dt{color:var(--muted)}dd{margin:0;overflow-wrap:anywhere}
ul.evidence{padding-left:1.25rem}
footer{margin-top:3rem;color:var(--muted);font-size:.85rem}
@media (max-width:40rem){
  main{padding:1.5rem 1rem 3rem}h1{font-size:1.6rem}
  dl{grid-template-columns:1fr}dt{margin-top:.5rem}
}
@media print{
  :root{--bg:#fff;--fg:#111827;--muted:#374151;--rule:#9CA3AF}
  body{font-size:10.5pt}main{max-width:none;padding:0}
  .tablist,.tabs>input{display:none}
  .panel{display:block!important;break-before:page}
  tr,img,.callout{break-inside:avoid}
  a{color:inherit;text-decoration:underline}
}
"""
