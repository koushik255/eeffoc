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

## Find a screenshot

```sh
.venv/bin/python -m bench.find shot.jpg other.png      # -> .local/found/<name>.jpg
```

Searches Steel Ball Run with manga-ft1 + fine tiles + views + SIFT check,
then shows the matched page with the page before and after it (screenshots
often catch two pages). Every shown page SIFT confirms gets an outline of
the part of the screenshot that matched it: red on the match, blue on a
neighbour.

## Benchmark

```sh
uv sync
.venv/bin/python -m bench.catalog --set sbr                     # all of Steel Ball Run -> .local/pages
.venv/bin/python -m bench.suite --set sbr --name sbr-v3         # 600 queries, 24 kinds (mangashot, test profile)
.venv/bin/python -m bench.run --suite sbr-v3 --systems curtain manga-ft1 manga-ft1+views manga-ft1+views+align
```

### sbr-v3 results (2026-10-18)

3,630 Steel Ball Run pages (the 96 repeated credit, blank and colophon pages
in `bench/sbr-skip.json` left out), 600 queries, exact page first.
"+ views" also searches with the manga cropped out of a screenshot, windows
along strips, and the halves of landscape images (`bench/views.py`);
"+ align 10" re-orders the top 10 with SIFT + RANSAC, as curtain's site does.

| type | curtain | manga-ft1 | manga-ft1 + views | manga-ft1 + align 10 | manga-ft1 + views + align 10 |
|---|---|---|---|---|---|
| full_page | 25/25 | 25/25 | 25/25 | 25/25 | 25/25 |
| wide_crop | 18/25 | 25/25 | 25/25 | 25/25 | 25/25 |
| medium_crop | 22/25 | 25/25 | 25/25 | 25/25 | 25/25 |
| tight_crop | 15/25 | 24/25 | 21/25 | 25/25 | 24/25 |
| panel | 5/25 | 16/25 | 20/25 | 22/25 | 22/25 |
| strip | 1/25 | 14/25 | 23/25 | 21/25 | 25/25 |
| low_res | 20/25 | 25/25 | 24/25 | 25/25 | 25/25 |
| heavy_compression | 23/25 | 25/25 | 25/25 | 25/25 | 25/25 |
| upscaled | 17/25 | 25/25 | 25/25 | 25/25 | 25/25 |
| greyscale | 24/25 | 25/25 | 25/25 | 25/25 | 25/25 |
| colour_shift | 19/25 | 24/25 | 23/25 | 25/25 | 25/25 |
| speech_bubbles | 16/25 | 25/25 | 24/25 | 25/25 | 25/25 |
| watermark | 13/25 | 25/25 | 25/25 | 25/25 | 25/25 |
| rotated | 15/25 | 25/25 | 25/25 | 25/25 | 25/25 |
| mirrored | 3/25 | 23/25 | 24/25 | 24/25 | 24/25 |
| stretched | 21/25 | 25/25 | 25/25 | 25/25 | 25/25 |
| border | 7/25 | 25/25 | 25/25 | 25/25 | 25/25 |
| scroll_reader | 4/25 | 22/25 | 21/25 | 23/25 | 23/25 |
| paged_reader | 1/25 | 25/25 | 25/25 | 25/25 | 25/25 |
| social_post | 0/25 | 20/25 | 24/25 | 22/25 | 24/25 |
| chat | 0/25 | 16/25 | 25/25 | 19/25 | 25/25 |
| browser | 0/25 | 21/25 | 25/25 | 25/25 | 25/25 |
| photo | 0/25 | 20/25 | 25/25 | 23/25 | 25/25 |
| two_page_view | 1/25 | 19/25 | 22/25 | 25/25 | 25/25 |
| **all** | 270/600 | 544/600 | 576/600 | 579/600 | 592/600 |

Median search time (16-core CPU): curtain 121 ms, manga-ft1 157 ms, + views
212 ms, + align 10 653 ms, + views + align 10 936 ms.

**Fine tiles** ("+fine", 2026-10-18): a 6 x 6 grid indexed as well (11 x 11
half-stride tiles, ~2.8% of the page each), 171 vectors a page instead of 50,
one more trunk pass at 1152 px (0.68 s a page on CPU).

| | manga-ft1 | + views | + views + align 10 |
|---|---|---|---|
| 4 x 4 tiles | 544 | 576 | 592 |
| + 6 x 6 tiles | 539 | 578 | 595 |

They find the tiny regions (tight crop 21 -> 25, panel 20 -> 21 with views;
24/25 panels with align) but on their own add near misses elsewhere (browser,
photo, mirrored), since the model only ever trained against 4 x 4 tiles.
Left with fine + views + align (5): one thin panel, one mirrored crop, two
scroll-reader shots where the next page fills most of the screen and the
model returns that page, and one small social-post image. Counting the page
before or after as a hit ("page or its neighbour", now in bench.run's
report), those two scroll-reader shots count: 597/600.

**Speedups** (same answers): the SIFT check stops after the first page when it
alone gets 50+ agreeing points (57 of 60 sampled queries), pages are scored
with one torch matrix multiply instead of numpy's einsum, and all of a
query's views go through the model in one batch. fine + views: 249 -> 151 ms
median; fine + views + align 10: 619 -> 238 ms (16-core CPU, 3,630 pages).

What's left with views + align (8 misses): tiny regions (a panel or tight
crop of 5-6% of the page, smaller than an index tile), one mirrored crop,
two scroll-reader shots and one social post. The extra views cost a few
plain crops (tight crop 24 -> 21 without align) where a white area makes
the content detector crop too.

The queries come from mangashot, which also made the training shots (the
test profile only swaps layouts, fonts and captions), so this measures the
kinds of screenshot we thought of. Real screenshots people take are the
next test.

On the earlier sbr-v2 suite (all 3,726 pages): curtain 269, manga-ft1 542,
manga-ft1 + align 10 575 of 600.

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
