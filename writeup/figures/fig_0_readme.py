"""README animation: one call loses its closing tag, BFCL drops it, the patch keeps it, and the score bar shows the cost.

A hand-written SVG with CSS keyframes (GitHub renders those in an <img>), on its own light card so it reads in
light and dark themes. The call is an illustration in the shape of the saved outputs. The counts and scores come
from writeup/build.numbers, the same source as the text. With prefers-reduced-motion the finished frame is shown.

    .venv/bin/python -I writeup/figures/fig_0_readme.py  ->  writeup/figures/generated/readme.svg
"""
import html
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import style as S  # noqa: E402

T = 16.0  # seconds per loop
W, H = 860, 432
CH = 7.8  # monospace character width at 13px, enforced with textLength so the timing marks line up
MONO = "ui-monospace, SFMono-Regular, Menlo, Consolas, 'DejaVu Sans Mono', monospace"
SANS = "-apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, 'DejaVu Sans', sans-serif"
CARD = "#fbfaf7"
PATCH, PATCH_BAND = "#0072b2", "#eaf3f9"  # as in figure 1
RED_TINT, GREEN_TINT = "#fbeceb", "#e9f6f1"

css, body = [], []


def pc(t):
    return f"{100 * t / T:.2f}%"


def fade(name, t_in, dur=0.35):
    """Hidden until t_in, then shown until the loop fades out."""
    css.append(f"@keyframes {name}{{0%,{pc(t_in)}{{opacity:0}}{pc(t_in + dur)},94%{{opacity:1}}98%,100%{{opacity:0}}}}"
               f".{name}{{animation:{name} {T}s linear infinite both}}")
    return name


def flash(name, t_in, t_out):
    """Shown only between t_in and t_out (hidden in the static frame)."""
    css.append(f"@keyframes {name}{{0%,{pc(t_in)}{{opacity:0}}{pc(t_in + 0.2)},{pc(t_out)}{{opacity:1}}"
               f"{pc(t_out + 0.3)},100%{{opacity:0}}}}.{name}{{opacity:0;animation:{name} {T}s linear infinite both}}")
    return name


def grow(name, t0, t1):
    """A bar that grows from its left edge between t0 and t1."""
    # opacity is pinned on every frame, or it would interpolate over the whole loop
    css.append(f"@keyframes {name}{{0%,{pc(t0)}{{transform:scaleX(0);opacity:1}}{pc(t1)},94%{{transform:scaleX(1);opacity:1}}"
               f"98%,100%{{transform:scaleX(1);opacity:0}}}}"
               f".{name}{{transform-box:fill-box;transform-origin:left center;"
               f"animation:{name} {T}s cubic-bezier(.3,.7,.4,1) infinite both}}")
    return name


def text(x, y, s, size=13, fill=S.INK, cls="", mono=False, weight="normal", anchor="start", spacing=None):
    fam = MONO if mono else SANS
    extra = f' textLength="{len(s) * CH:.1f}" lengthAdjust="spacingAndGlyphs"' if mono else ""
    if spacing:
        extra += f' letter-spacing="{spacing}"'
    c = f' class="{cls}"' if cls else ""
    body.append(f'<text x="{x}" y="{y}" font-family="{fam}" font-size="{size}" fill="{fill}" font-weight="{weight}"'
                f' text-anchor="{anchor}"{extra}{c}>{html.escape(s)}</text>')


def rect(x, y, w, h, fill, stroke="none", rx=6, cls="", dash=None, sw=1):
    c = f' class="{cls}"' if cls else ""
    d = f' stroke-dasharray="{dash}"' if dash else ""
    body.append(f'<rect x="{x}" y="{y}" width="{w:.1f}" height="{h}" rx="{rx}" fill="{fill}" stroke="{stroke}"'
                f' stroke-width="{sw}"{d}{c}/>')


def typed(name, x, y, s, t0, t1, fill=S.INK):
    """A code line that is typed out: a cover with a caret slides off it, clipped to the code box."""
    text(x, y, s, mono=True, fill=fill)
    w = len(s) * CH + 4
    # the caret rests at the line end until the next line starts, then leaves the box
    css.append(f"@keyframes {name}{{0%,{pc(t0)}{{transform:translateX(0)}}{pc(t1)},{pc(t1 + 0.12)}{{transform:translateX({w:.0f}px)}}"
               f"{pc(t1 + 0.13)},96%{{transform:translateX(440px)}}100%{{transform:translateX(0)}}}}"
               f".{name}{{transform:translateX(440px);animation:{name} {T}s linear infinite both}}")
    # clip on the outer group, move the inner one, so the clip stays put while the cover slides out of it
    body.append(f'<g clip-path="url(#code)"><g class="{name}"><rect x="{x - 1}" y="{y - 14}" width="420" height="20" fill="white"/>'
                f'<rect x="{x - 1}" y="{y - 13}" width="2" height="17" fill="{S.INK}"/></g></g>')


def main():
    A, _, verify = S.data()
    n = S.build.numbers(os.path.join(S.ROOT, "data/audit"), verify, os.path.join(S.ROOT, "data/raw"))
    off, tol, q8 = (float(n[k]) for k in ("iq3_sp_off", "iq3_sp_tol", "q8_0_sp_off"))
    unclosed, short, items = n["iq3_sp_unclosed"], n["iq3_sp_one_short"], n["sp_items"]
    assert off < tol <= 100 and off < q8, (off, tol, q8)  # the patch only adds back, and the 3-bit build is below 8-bit
    assert 2 * short > unclosed, "the footnote says most are one token short"

    # header
    text(28, 42, "One missing token, one lost call", size=19, weight="600")
    text(28, 64, "Qwen3-4B-Instruct-2507 at IQ3_XXS (3-bit), scored by BFCL on simple_python", size=12.5, fill=S.MUTED)

    # 1. what the model writes
    text(28, 102, "THE MODEL WRITES", size=10.5, fill=S.MUTED, weight="600", spacing="0.08em")
    rect(28, 112, 404, 150, "white", S.RULE, rx=8)
    typed("ty1", 46, 142, "<tool_call>", 0.5, 1.1)
    typed("ty2", 46, 166, '{"name": "area", "arguments": {"r": 2}}', 1.2, 2.8)
    ghost = fade("ghost", 3.1)
    rect(42, 178, len("</tool_call>") * CH + 8, 24, "#fdf6e6", S.TAG, rx=4, dash="4 3", sw=1.4, cls=ghost)
    body.append(f'<g class="{ghost}">')
    text(46, 195, "</tool_call>", mono=True, fill="#b07800")
    body.append("</g>")
    end = fade("imend", 3.6)
    body.append(f'<g class="{end}">')
    text(46 + len("</tool_call>") * CH + 18, 195, "<|im_end|>", mono=True, fill=S.MUTED)
    body.append("</g>")
    note = fade("note", 4.0)
    body.append(f'<g class="{note}">')
    rect(46, 214, 9, 9, S.TAG, rx=2)
    text(62, 223, "closing tag never written: the model ends its turn", size=12, fill=S.INK)
    text(62, 243, "one token short of a well-formed call", size=12, fill=S.MUTED)
    body.append("</g>")

    # 2. BFCL's parser
    text(460, 102, "BFCL'S PARSER", size=10.5, fill=S.MUTED, weight="600", spacing="0.08em")
    rx_s = r"<tool_call>\n(.*?)\n</tool_call>"
    rect(460, 112, 372, 44, "#f6f5f0", S.RULE, rx=8)
    text(476, 139, rx_s, mono=True)
    scan = flash("scan", 4.7, 5.9)
    css.append(f"@keyframes sweep{{0%,{pc(4.7)}{{transform:translateX(0)}}{pc(5.9)},100%{{transform:translateX({18 * CH:.0f}px)}}}}"
               f".sweep{{animation:sweep {T}s linear infinite both}}")
    body.append(f'<g class="{scan}"><rect class="sweep" x="474" y="122" width="{3 * CH:.0f}" height="24" rx="3" '
                f'fill="{S.GOOD}" opacity="0.18"/></g>')
    miss = fade("miss", 6.0)
    i0 = rx_s.index(r"\n</")
    rect(476 + i0 * CH - 2, 122, (len(rx_s) - i0) * CH + 4, 24, "none", S.BAD, rx=3, sw=1.6, cls=miss)
    drop = fade("drop", 6.4)
    body.append(f'<g class="{drop}">')
    rect(460, 166, 372, 30, RED_TINT, S.BAD, rx=6)
    text(476, 186, "✕  no match, so the call is dropped without a trace", size=12.5, fill=S.INK)
    body.append("</g>")

    # 3. what gets recorded
    rec = fade("rec", 7.2)
    body.append(f'<g class="{rec}">')
    text(460, 220, "BFCL records", size=11.5, fill=S.MUTED)
    text(560, 220, "no call: scored like a plain-text answer", size=12.5, fill=S.INK, weight="600")
    body.append("</g>")
    fix = fade("fix", 8.2)
    body.append(f'<g class="{fix}">')
    rect(452, 232, 388, 32, PATCH_BAND, "none", rx=6)
    text(460, 253, "With the patch", size=11.5, fill=PATCH, weight="600")
    text(560, 253, "✓  call kept (tolerant mode, opt-in)", size=12.5, fill=S.INK)
    body.append("</g>")

    # 4. the cost on the score
    x0, sc, by, bh = 28, 7.0, 330, 20  # 0-100 % over 700 px
    text(28, 304, "simple_python accuracy (%)", size=12, fill=S.MUTED)
    rect(x0, by, 100 * sc, bh, "#efede6", rx=4)
    tag = grow("gtag", 11.0, 12.2)
    rect(x0, by, tol * sc, bh, S.TAG, rx=4, cls=tag)
    base = grow("gbase", 9.2, 10.6)
    rect(x0, by, off * sc, bh, S.SCORE, rx=4, cls=base)
    ref = fade("ref", 9.2)
    body.append(f'<g class="{ref}"><line x1="{x0 + q8 * sc:.1f}" x2="{x0 + q8 * sc:.1f}" y1="{by - 8}" y2="{by + bh + 8}" '
                f'stroke="{S.INK}" stroke-width="1.2" stroke-dasharray="1.5 2.5"/>')
    text(x0 + q8 * sc + 6, by - 4, f"Q8_0 (8-bit) {S.one_decimal(q8)}", size=11.5, fill=S.MUTED)
    body.append("</g>")
    l1 = fade("l1", 10.6)
    body.append(f'<g class="{l1}">')
    rect(x0, by + bh + 14, 9, 9, S.SCORE, rx=2)
    text(x0 + 15, by + bh + 23, f"BFCL score {S.one_decimal(off)}", size=12, fill=S.INK)
    body.append("</g>")
    l2 = fade("l2", 12.2)
    body.append(f'<g class="{l2}">')
    rect(x0 + 170, by + bh + 14, 9, 9, S.TAG, rx=2)
    text(x0 + 185, by + bh + 23, f"{S.one_decimal(tol)} once the missing tag is tolerated", size=12, fill=S.INK)
    body.append("</g>")
    text(28, 412, f"{unclosed} of {items} answers lose the closing tag, {short} of them only that one token. "
                  "The call above is an illustration.", size=11, fill=S.MUTED)

    svg = (f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" role="img" '
           f'aria-labelledby="t d">\n<title id="t">One missing token, one lost call</title>\n'
           f'<desc id="d">A quantized model writes a tool call but ends its turn before the closing tag. BFCL\'s regex '
           f'does not match, so the call is dropped and scored as no call. With the patch\'s opt-in tolerant mode the '
           f'call is kept. The IQ3_XXS simple_python score is {S.one_decimal(off)}, and {S.one_decimal(tol)} once the '
           f'missing tag is tolerated, against {S.one_decimal(q8)} for Q8_0.</desc>\n'
           f'<defs><clipPath id="code"><rect x="29" y="113" width="402" height="148" rx="8"/></clipPath></defs>\n'
           f'<style>{"".join(css)}@media (prefers-reduced-motion: reduce){{*{{animation:none!important}}}}</style>\n'
           f'<rect x="0.5" y="0.5" width="{W - 1}" height="{H - 1}" rx="12" fill="{CARD}" stroke="{S.RULE}"/>\n'
           + "\n".join(body) + "\n</svg>\n")
    os.makedirs(S.OUT, exist_ok=True)
    with open(os.path.join(S.OUT, "readme.svg"), "w") as f:
        f.write(svg)
    print("wrote writeup/figures/generated/readme.svg")


if __name__ == "__main__":
    main()
