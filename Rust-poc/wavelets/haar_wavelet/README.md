# haar-wavelet

A Rust CLI tool that:
1. Loads a PNG and converts it to grayscale
2. Applies a multi-level 2-D Haar wavelet transform
3. Accepts a pixel-space bounding box and marks every wavelet tree node
   whose spatial footprint overlaps that region
4. Saves a colour-coded visualization of the coefficient pyramid
5. Optionally reconstructs an image using only the selected coefficients

---

## Build

```bash
cargo build --release
# binary at target/release/haar-wavelet
```

Requires Rust ≥ 1.70.  Image dimensions are automatically padded to the next
power of two; no manual pre-processing needed.

---

## Usage

```
haar-wavelet [OPTIONS] <INPUT>

Arguments:
  <INPUT>   Input PNG image

Options:
  -l, --levels <N>           Decomposition levels [default: 3]
      --bbox X0 Y0 X1 Y1     Bounding box (inclusive pixel coords)
      --expand-ancestors      Also mark all coarser parent nodes
      --expand-descendants    Also mark all finer child nodes
  -o, --output <FILE>        Wavelet visualization [default: wavelet_vis.png]
      --reconstruct <FILE>   Reconstruction from marked coefficients only
      --debug-grids          Print ASCII coefficient grids (small images)
  -h, --help
```

### Examples

```bash
# Visualise the wavelet pyramid (3 levels)

haar-wavelet photo.png

cargo run "wavelets/img/photo_01_512x512.png"

# Visualise with 4 levels

haar-wavelet photo.png --levels 4

cargo run "wavelets/img/photo_01_512x512.png" --levels 4

# Select a region and show which nodes it hits

haar-wavelet photo.png --bbox 100 50 300 200

cargo run "wavelets/img/photo_01_512x512.png" --bbox 100 50 300 200

# Select a region, propagate to all finer descendants, reconstruct

haar-wavelet photo.png --levels 3 \
    --bbox 100 50 300 200 \
    --expand-descendants \
    --output wv.png \
    --reconstruct region_only.png

# Debug small test image (prints ASCII grids)

haar-wavelet test.png --levels 2 --debug-grids

```

---

## Architecture

```
haar_wavelet/
├── Cargo.toml
└── src/
    ├── main.rs           CLI + orchestration + visualization
    ├── haar.rs           1-D / 2-D Haar forward & inverse DWT
    └── wavelet_tree.rs   WaveletTree, NodeId, Subband, bbox selection
```

### Coefficient layout

After *L* decomposition levels on a *W × H* image the coefficients are stored
in a single `W × H` array with this tiled pyramid layout:

```
┌──────┬──────┬───────────────────┐
│ LL_L │ LH_L │                   │
├──────┼──────┤      LH_1         │
│ HL_L │ HH_L │                   │
├─────────────┼───────────────────┤
│             │                   │
│    HL_1     │      HH_1         │
│             │                   │
└─────────────┴───────────────────┘
```

Subband origins at level *l*:

| Subband | Row offset | Col offset | Size (rows × cols) |
|---------|-----------|-----------|-------------------|
| LL (coarsest only) | 0 | 0 | H/2^L × W/2^L |
| LH_l | 0 | W/2^l | H/2^l × W/2^l |
| HL_l | H/2^l | 0 | H/2^l × W/2^l |
| HH_l | H/2^l | W/2^l | H/2^l × W/2^l |

### Bounding-box → node mapping

A pixel at `(px, py)` maps to wavelet coefficient `(py >> l, px >> l)` at
level *l*.  Therefore bounding box `[x0, y0, x1, y1]` selects coefficients
at rows `[y0 >> l, y1 >> l]` and cols `[x0 >> l, x1 >> l]` for every level.

### Wavelet tree parent–child links

```
parent of (level, subband, row, col)  →  (level+1, subband, row>>1, col>>1)
children of (level, subband, row, col) →  (level-1, subband, 2r, 2c)
                                           (level-1, subband, 2r+1, 2c)
                                           (level-1, subband, 2r, 2c+1)
                                           (level-1, subband, 2r+1, 2c+1)
```

Each detail subband (LH, HL, HH) forms an independent quadtree.

---

## Running the built-in tests

```bash
cargo test
```

The test suite checks:

| Test | What it verifies |
|------|-----------------|
| `roundtrip_1d` | `haar_1d_inv(haar_1d_fwd(x)) == x` |
| `roundtrip_2d` | `haar_2d_inv(haar_2d_fwd(x)) == x` for 8×8, 3 levels |
| `energy_preservation_2d` | Parseval relation: energy scales by `(1/4)^levels` |
