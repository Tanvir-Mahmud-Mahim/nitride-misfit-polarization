"""Length of the Letter by the APS rules for Letters
(https://journals.aps.org/authors/length-guide): words of the body text and of
the figure and table captions, plus 16 words per row of single-column
displayed math, [300 / (0.5 * aspect)] + 40 words per double-column figure,
[150 / aspect] + 20 per single-column figure, and 13 + 6.5 words per row of a
single-column table (26 + 13 per row for a double-column one). Title,
authors, abstract, acknowledgments and references are excluded. The PRB limit
for a Letter is 4500 words.

usage: python3 scripts/wordcount.py paper/letter.tex
"""

import os
import re
import subprocess
import sys


def pdf_aspect(path):
    out = subprocess.run(["pdfinfo", path], capture_output=True, text=True).stdout
    m = re.search(r"Page size:\s+([\d.]+) x ([\d.]+)", out)
    w, h = float(m.group(1)), float(m.group(2))
    return w / h


def main(tex):
    src = open(tex).read()
    src = re.sub(r"(?<!\\)%.*", "", src)
    body = src[src.index(r"\maketitle"):src.index(r"\begin{acknowledgments}")]
    tail = src[src.index(r"\end{acknowledgments}"):src.index(r"\bibliography")]
    body += tail   # data-availability statement counted as text
    base = os.path.dirname(os.path.abspath(tex))
    extra = 0.0
    items = []
    for env in re.finditer(r"\\begin\{(figure\*?|table\*?)\}(.*?)\\end\{\1\}", body, re.S):
        kind, content = env.group(1), env.group(2)
        if kind.startswith("figure"):
            g = re.search(r"\\includegraphics\[width=([\d.]*)\\(textwidth|columnwidth)\]\{([^}]*)\}", content)
            asp = pdf_aspect(os.path.join(base, g.group(3)))
            words = 300 / (0.5 * asp) + 40 if kind == "figure*" else 150 / asp + 20
            items.append((g.group(3), round(words)))
        else:
            rows = len(re.findall(r"\\\\", content.split(r"\begin{tabular}")[1].split(r"\end{tabular}")[0]))
            words = 26 + 13 * rows if kind == "table*" else 13 + 6.5 * rows
            items.append(("table", round(words)))
        extra += words
    eq_rows = 0
    for eq in re.finditer(r"\\begin\{equation\}(.*?)\\end\{equation\}", body, re.S):
        eq_rows += 1 + eq.group(1).count(r"\\")
    extra += 16 * eq_rows
    # text: remove equations and tabular bodies, keep captions
    text = re.sub(r"\\begin\{equation\}.*?\\end\{equation\}", " ", body, flags=re.S)
    text = re.sub(r"\\begin\{tabular\}.*?\\end\{tabular\}", " ", text, flags=re.S)
    text = re.sub(r"\\includegraphics(\[[^]]*\])?\{[^}]*\}", " ", text)
    text = re.sub(r"\\(label|ref|cite|onlinecite|eqref)\{[^}]*\}", " X ", text)
    text = re.sub(r"\$[^$]*\$", " X ", text)
    plain = subprocess.run(["detex", "-n"], input=text, capture_output=True, text=True).stdout
    nwords = len([w for w in plain.split() if re.search(r"[A-Za-z0-9]", w)])
    total = nwords + extra
    print("text words (body + captions): %d" % nwords)
    for name, w in items:
        print("  %s: %d" % (name, w))
    print("display-math rows: %d (%d words)" % (eq_rows, 16 * eq_rows))
    print("TOTAL: %d words (PRB Letter limit 4500)" % round(total))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "paper/letter.tex")
