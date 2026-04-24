"""Generate data/sampling_report.typ from data/sampling_report_data.json.

Produces a typst document with one section per prompt template,
showing all language translations and 3 sample completions each.
"""
import json
from pathlib import Path

DATA = Path(__file__).resolve().parent.parent / "data"


def escape_typst(s: str) -> str:
    """Escape characters that are special in typst markup."""
    # Replace actual newlines with visible \n
    s = s.replace("\n", "\\n")
    # Escape typst special chars
    for ch in ["#", "$", "@", "*", "_", "~", "`", "<", ">"]:
        s = s.replace(ch, "\\" + ch)
    # Truncate display
    if len(s) > 350:
        s = s[:347] + "..."
    return s


def dim_label(ic, ts, ss):
    return f"IC={ic} TS={ts} SS={ss}"


def main():
    with open(DATA / "sampling_report_data.json") as f:
        data = json.load(f)

    lines = []
    w = lines.append

    w('#set document(title: "Cultural Completion Sampling Report")')
    w('#set page(margin: 2cm)')
    w('#set text(font: "New Computer Modern", size: 10pt)')
    w('#set heading(numbering: "1.")')
    w('#set par(justify: true)')
    w("")
    w("#align(center)[")
    w('  #text(size: 16pt, weight: "bold")[Cultural Completion Sampling Report]')
    w("  #v(0.3em)")
    w('  #text(size: 11pt)[All Templates, Languages, and Sample Completions]')
    w("  #v(0.3em)")
    w('  #text(size: 9pt, fill: gray)[Generated from culture.db — v2 classifier, trimmed prompts, lang-matched]')
    w("]")
    w("")
    w("#v(1em)")
    w("")
    w("This document shows every prompt template used in the cultural completion experiment, ")
    w("its translation into each of 27 languages, and 3 randomly sampled completions per language ")
    w("with their v2 classifier labels. Completions are drawn from diverse models where possible. ")
    w("Newlines in completions are shown as `\\\\n`. Completions are truncated at 350 characters.")
    w("")

    for tmpl in data["templates"]:
        tid = tmpl["template_id"]
        target = tmpl.get("cultural_target", "")
        w(f"#pagebreak()")
        w(f'= {tid}')
        w(f'#text(size: 9pt, fill: gray)[Cultural target: {escape_typst(target)}]')
        w("")

        for lang_data in tmpl["languages"]:
            lang = lang_data["lang"]
            lang_name = lang_data.get("lang_name", lang)
            prompt = lang_data["prompt_text"]
            vidx = lang_data.get("variant_idx", "?")
            completions = lang_data.get("completions", [])

            w(f'== {lang_name} ({lang}) #h(1fr) #text(size: 8pt, fill: gray)[v{vidx}]')
            w(f'*Prompt:* "{escape_typst(prompt)}"')
            w("")

            if not completions:
                w('#text(size: 8pt, fill: gray)[(no completions)]')
                w("")
                continue

            # Use a compact table-like format
            for i, comp in enumerate(completions, 1):
                model = comp.get("model_id", "?")
                text = escape_typst(comp.get("completion_text", ""))
                cat = comp.get("content_category", "?")
                ic = comp.get("dim_ic", "?")
                ts = comp.get("dim_ts", "?")
                ss = comp.get("dim_ss", "?")

                # Model tag + category in small colored text
                w(f'#text(size: 8pt)[')
                w(f'  #text(fill: rgb("#666"))[{model}] '
                  f'#h(0.5em) #text(weight: "bold")[{cat}] '
                  f'#h(0.3em) #text(fill: rgb("#888"))[IC={ic} TS={ts} SS={ss}] \\')
                w(f'  #text(fill: rgb("#333"))[{text}]')
                w(f']')
                w("")

        w("")

    out_path = DATA / "sampling_report.typ"
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {out_path} ({len(lines)} lines)")


if __name__ == "__main__":
    main()
