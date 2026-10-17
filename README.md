# eeffoc

Exact-match search for manga: crop a panel, find the page it came from. The
same idea as [curtain](https://github.com/koushik255/curtain), for manga pages
instead of movie frames.

## mangashot

`mangashot/` fakes the ways people screenshot manga: crops, panels and
strips; speech bubbles, watermarks, mirroring, rotation, stretching, borders;
low resolution, heavy compression, upscaling; and scenes around the page
(phone scroll and paged readers, social posts, chat apps, a desktop browser,
photos of a screen or print, two-page views). `mangashot.make(kind, page, rng)`
makes one named kind (the benchmark), `mangashot.random_shot(page, rng)` a
random mix (training). Layouts, fonts, captions and backgrounds come in
variant pools; every third variant is only used with `profile="test"`, so
the benchmark shows the model screenshot styles it never trained on.

## Training

Fine-tunes curtain's ResNet-18 on Berserk, One Piece and Vagabond (37k
pages; never Steel Ball Run). Each page gives a pair: a `random_shot` of it,
and the clean tile or whole page the index stores for that region.

```sh
.venv/bin/python -m training.prepare        # grey 1024 px pages -> .local/train-pages.tar
modal volume put eeffoc-training-data .local/train-pages.tar /datasets/train-pages.tar
modal run --detach cloud/train.py --run-name manga-ft1 --epochs 12
modal volume get eeffoc-training-data /runs/manga-ft1/best.pt .local/runs/manga-ft1/best.pt
```

`manga-ft1`: 12 epochs on one L4 with 32 CPUs, ~3 min an epoch. Exact page
on held-out Vagabond pages went from 55.5% to 95.5% (best at epoch 10).

## Benchmark

```sh
uv sync
.venv/bin/python -m bench.catalog --set sbr                     # all of Steel Ball Run -> .local/pages
.venv/bin/python -m bench.suite --set sbr --name sbr-v2         # 600 queries, 24 kinds (mangashot, test profile)
.venv/bin/python -m bench.run --suite sbr-v2 --systems curtain manga-ft1
```

### sbr-v2 results (2026-10-17)

All 3,726 Steel Ball Run pages indexed, 600 queries, exact page first.
"+ align 10" re-orders the model's top 10 with SIFT + RANSAC, as curtain's
site does.

| type | curtain | manga-ft1 | curtain + align 10 | manga-ft1 + align 10 |
|---|---|---|---|---|
| full_page | 25/25 | 25/25 | 25/25 | 25/25 |
| wide_crop | 19/25 | 25/25 | 21/25 | 25/25 |
| medium_crop | 23/25 | 25/25 | 24/25 | 25/25 |
| tight_crop | 14/25 | 23/25 | 19/25 | 24/25 |
| panel | 6/25 | 18/25 | 8/25 | 24/25 |
| strip | 1/25 | 14/25 | 3/25 | 21/25 |
| low_res | 16/25 | 25/25 | 22/25 | 25/25 |
| heavy_compression | 23/25 | 25/25 | 24/25 | 25/25 |
| upscaled | 16/25 | 24/25 | 21/25 | 24/25 |
| greyscale | 24/25 | 25/25 | 25/25 | 25/25 |
| colour_shift | 18/25 | 24/25 | 21/25 | 25/25 |
| speech_bubbles | 15/25 | 25/25 | 21/25 | 25/25 |
| watermark | 15/25 | 25/25 | 22/25 | 25/25 |
| rotated | 16/25 | 25/25 | 19/25 | 25/25 |
| mirrored | 3/25 | 23/25 | 6/25 | 24/25 |
| stretched | 20/25 | 24/25 | 23/25 | 24/25 |
| border | 8/25 | 25/25 | 14/25 | 25/25 |
| scroll_reader | 3/25 | 23/25 | 4/25 | 24/25 |
| paged_reader | 1/25 | 25/25 | 2/25 | 25/25 |
| social_post | 0/25 | 19/25 | 1/25 | 21/25 |
| chat | 0/25 | 16/25 | 0/25 | 18/25 |
| browser | 0/25 | 21/25 | 1/25 | 25/25 |
| photo | 0/25 | 19/25 | 1/25 | 22/25 |
| two_page_view | 3/25 | 19/25 | 6/25 | 24/25 |
| **all** | **269/600** | **542/600** | **333/600** | **575/600** |
| median time | 77 ms | 75 ms | 470 ms | 476 ms |

The queries come from mangashot, which also made the training shots (the
test profile only swaps layouts, fonts and captions), so this measures the
kinds of screenshot we thought of. Real screenshots people take are the
next test.

### v1 results (2026-10-17)

The first benchmark, before mangashot: `.local/suite/v1`, made from
`bench.catalog --set v1`.

754 pages (Steel Ball Run colour v01, v12; Berserk v10, v26), 360 queries,
everything in grey, 16-core CPU. Exact page first:

| type | curtain ResNet-18 | curtain + align 10 | SIFT + RANSAC |
|---|---|---|---|
| full_page | 20/20 | 20/20 | 20/20 |
| wide_crop | 12/20 | 18/20 | 20/20 |
| medium_crop | 17/20 | 19/20 | 20/20 |
| tight_crop | 15/20 | 18/20 | 20/20 |
| panel | 9/20 | 14/20 | 20/20 |
| strip | 0/20 | 6/20 | 20/20 |
| low_res | 17/20 | 18/20 | 20/20 |
| heavy_jpeg | 19/20 | 19/20 | 20/20 |
| greyscale | 20/20 | 20/20 | 20/20 |
| colour_shift | 17/20 | 19/20 | 20/20 |
| text_edit | 15/20 | 18/20 | 20/20 |
| rotated | 7/20 | 16/20 | 20/20 |
| mirrored | 9/20 | 9/20 | 13/20 |
| aspect_squash | 16/20 | 19/20 | 19/20 |
| upscaled | 16/20 | 18/20 | 18/20 |
| phone_screenshot | 2/20 | 4/20 | 20/20 |
| photo | 2/20 | 4/20 | 19/20 |
| two_page_view | 1/20 | 6/20 | 20/20 |
| **all** | **214/360** | **265/360** | **349/360** |
| median time | 55 ms | 490 ms | 3,243 ms |

- **curtain** is the movie model (`combined-rocm-50k-v1`) used as is: 49
  shared-trunk tiles + whole page, query + 4 corner pieces.
- **align 10** re-orders curtain's top 10 with SIFT + RANSAC. It turns almost
  every top-10 hit into the exact page (265 of 267), so what limits curtain
  is getting the page into its top 10.
- **SIFT + RANSAC** searches 2.2M RootSIFT features directly. It's the most
  accurate but far too slow to scale; it's the accuracy target, not the search.
- **SSCD** (ResNet-50) was stopped at 400/754 pages (~3 s a page on CPU) and
  has no results yet.

The queries are made from the indexed digital pages themselves, so the suite
doesn't yet test a different scan or a translated edition.
