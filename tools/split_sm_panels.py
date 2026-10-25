# Split the two multi-panel real-device figures into single-panel PDFs for
# vertical (a)(b)(c) stacking in the manuscript. Originals untouched.
#
# The crop runs in PDF space so the panels stay vector: the bar charts, error
# bars and annotation text are all drawn, not sampled, which is what lets the
# manuscript keep them legible at the 0.29-column width these panels use.
import os

import pymupdf

FIGDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'fig')

JOBS = [
    ("sm_real_momentum.pdf", ["sm_real_momentum_a.pdf", "sm_real_momentum_b.pdf"]),
    ("sm_real_position.pdf", ["sm_real_position_a.pdf", "sm_real_position_b.pdf",
                              "sm_real_position_c.pdf"]),
]

for src, outs in JOBS:
    doc = pymupdf.open(os.path.join(FIGDIR, src))
    box = doc[0].rect
    n = len(outs)
    print(f"{src}: {box.width:.0f}x{box.height:.0f} pt -> {n} panels")
    for i, out in enumerate(outs):
        clip = pymupdf.Rect(box.x0 + box.width * i / n, box.y0,
                            box.x0 + box.width * (i + 1) / n, box.y1)
        dest = pymupdf.open()
        dest.new_page(width=clip.width, height=clip.height).show_pdf_page(
            pymupdf.Rect(0, 0, clip.width, clip.height), doc, 0, clip=clip)
        path = os.path.join(FIGDIR, out)
        dest.save(path, garbage=4, deflate=True)
        dest.close()
        print(f"  {out}: {clip.width:.0f}x{clip.height:.0f} pt  "
              f"({os.path.getsize(path) / 1024:.0f} KB)")
    doc.close()

print("done")
