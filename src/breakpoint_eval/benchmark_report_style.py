"""Report palette and stylesheet. One source for HTML, plots, media, and render_report."""

from __future__ import annotations

# Logo ink and accent; neutrals chosen to sit with the ink.
INK = "#17211f"
ACCENT = "#d95229"
LIGHT_BG = "#ffffff"
LIGHT_MUTED = "#5d6663"
LIGHT_RULE = "#d6dad8"
LIGHT_NEUTRAL_BG = "#eef0ef"
DARK_BG = "#17211f"
DARK_FG = "#f6f5ef"
DARK_MUTED = "#a8b0ad"
DARK_RULE = "#3a4441"
ACCENT_TINT = "rgba(217, 82, 41, 0.14)"
INK_TINT = "rgba(23, 33, 31, 0.08)"


def hex_to_rgb(value: str) -> tuple[int, int, int]:
    h = value.lstrip("#")
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


INK_RGB = hex_to_rgb(INK)
ACCENT_RGB = hex_to_rgb(ACCENT)
LIGHT_BG_RGB = hex_to_rgb(LIGHT_BG)
DARK_FG_RGB = hex_to_rgb(DARK_FG)
MUTED_RGB = hex_to_rgb(DARK_MUTED)

CSS = f"""
:root{{
  --bg:{LIGHT_BG};--fg:{INK};--muted:{LIGHT_MUTED};--rule:{LIGHT_RULE};
  --baseline:{INK};--stress:{ACCENT};
  --baseline-tint:{INK_TINT};--stress-tint:{ACCENT_TINT};
  --gained-bg:{LIGHT_NEUTRAL_BG};
  --sans:system-ui,-apple-system,"Segoe UI",Roboto,"Helvetica Neue",Arial,sans-serif;
  --mono:ui-monospace,"SF Mono","Cascadia Mono",Menlo,Consolas,"Liberation Mono",monospace;
}}
@media (prefers-color-scheme:dark){{
  :root{{--bg:{DARK_BG};--fg:{DARK_FG};--muted:{DARK_MUTED};--rule:{DARK_RULE};
    --baseline:{DARK_FG};--baseline-tint:rgba(246,245,239,.08);
    --gained-bg:rgba(246,245,239,.12)}}
}}
*{{box-sizing:border-box}}
html{{-webkit-text-size-adjust:100%}}
body{{margin:0;background:var(--bg);color:var(--fg);font-family:var(--sans);line-height:1.6;
  font-size:1rem}}
main{{max-width:64rem;margin:0 auto;padding:3rem 1.25rem 4rem}}
header{{margin-bottom:2.5rem}}
h1{{font-size:2rem;line-height:1.2;margin:0 0 .75rem;font-weight:650}}
h2{{font-size:1.35rem;margin:2.5rem 0 .75rem;font-weight:650}}
h3{{font-size:1.05rem;margin:2rem 0 .5rem;font-weight:650}}
p,ul{{margin:.75rem 0}}
code,.num{{font-family:var(--mono);font-size:.88em}}
code{{display:inline}}
.unit{{color:var(--muted)}}
ul.plain{{padding-left:1.25rem}}
ul.plain li{{margin:.4rem 0}}
ul.caveats li{{padding-left:.25rem}}
a{{color:var(--fg);text-decoration:underline}}
.scroll{{overflow-x:auto;margin:1rem 0}}
table{{border-collapse:collapse;width:100%;font-size:.92rem}}
caption{{text-align:left;color:var(--muted);padding:.25rem 0 .5rem;font-size:.85rem}}
th,td{{padding:.5rem .6rem;border-bottom:1px solid var(--rule);text-align:left;vertical-align:top}}
thead th{{font-weight:600;border-bottom:2px solid var(--fg)}}
td.num{{text-align:right;white-space:nowrap}}
th code{{display:block;color:var(--muted);font-weight:400}}
tr.baseline th,tr.baseline td{{background:var(--baseline-tint)}}
img{{display:block;max-width:100%;height:auto;margin:1.25rem 0}}
table.matrix{{width:auto;font-family:var(--mono);font-size:.8rem}}
table.matrix th,table.matrix td{{padding:.2rem .3rem;text-align:center;border-bottom:1px solid var(--rule)}}
table.matrix th[scope=row]{{text-align:left;white-space:nowrap;padding-right:.8rem}}
table.matrix td.lost{{background:var(--stress-tint);color:var(--stress);font-weight:700}}
table.matrix td.gained{{background:var(--gained-bg);color:var(--baseline);font-weight:700}}
.legend{{font-size:.85rem;color:var(--muted)}}
.legend .lost{{color:var(--stress)}}.legend .gained{{color:var(--baseline)}}
.tabs{{margin-top:2.5rem}}
.tabs-hint{{margin:0 0 .5rem;color:var(--muted);font-size:.9rem}}
.tabs>input{{position:absolute;opacity:0;pointer-events:none}}
.tablist{{display:flex;gap:.25rem;overflow-x:auto;border-bottom:1px solid var(--rule)}}
.tablist label{{padding:.6rem .9rem;cursor:pointer;white-space:nowrap;border-bottom:3px solid transparent;
  color:var(--muted)}}
.panel{{display:none;padding-top:.5rem}}
details{{border-top:1px solid var(--rule);padding:.75rem 0}}
details:last-of-type{{border-bottom:1px solid var(--rule)}}
summary{{cursor:pointer;font-weight:600}}
summary:focus-visible{{outline:2px solid var(--fg);outline-offset:2px}}
dl{{display:grid;grid-template-columns:minmax(10rem,16rem) 1fr;gap:.4rem 1.25rem;margin:1rem 0}}
dt{{color:var(--muted)}}dd{{margin:0;overflow-wrap:anywhere}}
ul.evidence{{padding-left:1.25rem}}
footer{{margin-top:3rem;color:var(--muted);font-size:.85rem}}
@media (max-width:40rem){{
  main{{padding:1.5rem 1rem 3rem}}h1{{font-size:1.6rem}}
  dl{{grid-template-columns:1fr}}dt{{margin-top:.5rem}}
}}
@media print{{
  :root{{--bg:{LIGHT_BG};--fg:{INK};--muted:{LIGHT_MUTED};--rule:{LIGHT_RULE};
    --baseline:{INK};--stress:{ACCENT};--baseline-tint:{INK_TINT};--stress-tint:{ACCENT_TINT};
    --gained-bg:{LIGHT_NEUTRAL_BG}}}
  body{{font-size:10.5pt}}main{{max-width:none;padding:0}}
  .tablist,.tabs>input{{display:none}}
  .panel{{display:block!important;break-before:page}}
  tr,img{{break-inside:avoid}}
  .tabs-hint{{display:none}}
  details::details-content{{content-visibility:visible;display:block}}
  a{{color:inherit;text-decoration:underline}}
}}
"""
