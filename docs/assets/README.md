# Paper figures

These images are rendered from the figure PDFs referenced by `IJCNN2026.tex` in
the authors' WCCI/IJCNN 2026 LaTeX source for
[Unified Long Video Inpainting and Outpainting via Overlapping High-Order Co-Denoising](https://arxiv.org/abs/2511.03272).

| Image | Original figure file | Paper figure |
| --- | --- | --- |
| `teaser.png` | `figures/Teaser_4.pdf` | Figure 1: showcase |
| `framework.png` | `figures/Framework.pdf` | Figure 2: training and inference framework |
| `qualitative-results.png` | `figures/main_results.pdf` | Figure 3: qualitative comparisons |

The figures retain their original content and labels. They are rendered using
the PDFs' crop boxes to exclude the unused page canvas. No experimental results
were regenerated for this documentation update.

Render each source figure with Poppler:

```bash
pdftoppm -f 1 -singlefile -cropbox -scale-to-x 2400 -scale-to-y -1 -png \
  figures/Teaser_4.pdf teaser
```
