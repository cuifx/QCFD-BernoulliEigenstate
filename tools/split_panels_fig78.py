# Split the (alpha, R) remedy map into single panels for vertical stacking.
# The source keeps its embedded "(a)/(b)" titles: they sit directly on top of
# the wide legend boxes, so the titles cannot be removed without damaging the
# legend. No extra LaTeX labels are added.
#
# The crop is done in PDF space, not on pixels: the panels stay vector, so the
# lettering and the isolines remain sharp at any size the manuscript uses.
# bernoulli_stationarity_a/b are NOT split here: those panels are written
# directly by spectral_stationarity.py.
import os

import pymupdf

FIGDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'fig')


def split(src, outs):
    """Cut one page into equal horizontal strips, keeping vector content."""
    doc = pymupdf.open(os.path.join(FIGDIR, src))
    box = doc[0].rect
    n = len(outs)
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


split("circulation_gauge_remedy_map.pdf",
      ["circulation_remedy_a.pdf", "circulation_remedy_b.pdf"])
print("done")
