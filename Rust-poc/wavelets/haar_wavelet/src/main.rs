// haar_wavelet/src/main.rs
// ────────────────────────────────────────────────────────────────────────────
// main.rs  –  CLI for the Haar wavelet system
//
// Usage examples
// ──────────────
//   # Just decompose and visualize the wavelet pyramid:
//   haar-wavelet photo.png --levels 4 --output wavelet.png
//
//   # Select a bounding box and see which nodes it hits:
//   haar-wavelet photo.png --levels 3 --bbox 100 50 300 200
//
//   # Also reconstruct the image from only those wavelet coefficients:
//   haar-wavelet photo.png --levels 3 --bbox 100 50 300 200 \
//       --output wv.png --reconstruct bbox_region.png
//
//   # Expand the selection to include all descendant (finer) nodes:
//   haar-wavelet photo.png --levels 3 --bbox 100 50 300 200 \
//       --expand-descendants --reconstruct region.png
//
//   # Print per-subband coefficient grids (small images only):
//   haar-wavelet test.png --levels 2 --debug-grids
// ────────────────────────────────────────────────────────────────────────────

mod haar;
mod wavelet_tree;

use clap::Parser;
use image::{GrayImage, Luma, Rgb, RgbImage};
use std::path::PathBuf;
use wavelet_tree::{NodeId, Subband, WaveletTree};

// ── CLI ─────────────────────────────────────────────────────────────────────

#[derive(Parser, Debug)]
#[command(
    name    = "haar-wavelet",
    version = "0.1",
    about   = "2-D Haar wavelet decomposition with spatial bounding-box node selection",
    long_about = None,
)]
struct Args {
    /// Input PNG image (converted to grayscale internally).
    input: PathBuf,

    /// Number of decomposition levels.
    #[arg(short, long, default_value = "3")]
    levels: usize,

    /// Bounding box for node selection: X0 Y0 X1 Y1 (pixel coords, inclusive).
    #[arg(long, num_args = 4, value_names = ["X0", "Y0", "X1", "Y1"])]
    bbox: Option<Vec<u32>>,

    /// After bbox selection, also mark all ancestor (coarser) nodes.
    #[arg(long, default_value_t = false)]
    expand_ancestors: bool,

    /// After bbox selection, also mark all descendant (finer) nodes.
    #[arg(long, default_value_t = false)]
    expand_descendants: bool,

    /// Output path for the wavelet coefficient visualization.
    #[arg(short, long, default_value = "wavelet_vis.png")]
    output: PathBuf,

    /// Save a reconstruction built only from the marked coefficients.
    #[arg(long)]
    reconstruct: Option<PathBuf>,

    /// Print ASCII coefficient grids for each subband (best for small images).
    #[arg(long, default_value_t = false)]
    debug_grids: bool,
}

// ── Image helpers ────────────────────────────────────────────────────────────

fn load_gray_f32(path: &PathBuf) -> Result<(Vec<f32>, usize, usize), Box<dyn std::error::Error>> {
    let img = image::open(path)?.to_luma8();
    let (w, h) = (img.width() as usize, img.height() as usize);
    let px: Vec<f32> = img.pixels().map(|p| p[0] as f32).collect();
    Ok((px, w, h))
}

/// Pad to the next power of two in both dimensions (zero-padding on right/bottom).
fn pad_power_of_two(px: &[f32], w: usize, h: usize) -> (Vec<f32>, usize, usize) {
    let pw = if w.is_power_of_two() {
        w
    } else {
        w.next_power_of_two()
    };
    let ph = if h.is_power_of_two() {
        h
    } else {
        h.next_power_of_two()
    };
    if pw == w && ph == h {
        return (px.to_vec(), w, h);
    }
    let mut out = vec![0.0_f32; pw * ph];
    for r in 0..h {
        out[r * pw..r * pw + w].copy_from_slice(&px[r * w..r * w + w]);
    }
    (out, pw, ph)
}

// ── Visualization ────────────────────────────────────────────────────────────

/// Build an RGB visualization of the wavelet coefficient pyramid.
///
/// • Each subband is normalized independently (absolute value for detail subbands,
///   direct range for LL) and rendered as grayscale.
/// • Marked nodes are highlighted with a red overlay.
/// • Subband boundaries are drawn as thin blue lines.
/// • The image is cropped to the original dimensions before saving.
fn visualize(tree: &WaveletTree, orig_w: usize, orig_h: usize) -> RgbImage {
    let (w, h) = (tree.width, tree.height);
    let mut out = RgbImage::new(w as u32, h as u32);

    // Helper: normalize a flat slice to bytes
    let to_bytes = |vals: &[f32]| -> Vec<u8> {
        let min = vals.iter().cloned().fold(f32::INFINITY, f32::min);
        let max = vals.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let rng = (max - min).max(1e-6);
        vals.iter()
            .map(|&v| ((v - min) / rng * 255.0).clamp(0.0, 255.0) as u8)
            .collect()
    };

    // ── 1.  Paint LL subband (coarsest level) ─────────────────────────────
    {
        let (sh, sw) = tree.subband_size(tree.levels);
        let vals: Vec<f32> = (0..sh)
            .flat_map(|r| (0..sw).map(move |c| tree.coeffs[r * w + c]))
            .collect();
        let bytes = to_bytes(&vals);
        for (i, &v) in bytes.iter().enumerate() {
            let (r, c) = (i / sw, i % sw);
            out.put_pixel(c as u32, r as u32, Rgb([v, v, v]));
        }
    }

    // ── 2.  Paint detail subbands (all levels, absolute value) ────────────
    for level in 1..=tree.levels {
        let (sh, sw) = tree.subband_size(level);

        for &sb in Subband::details().iter() {
            let (row_off, col_off) = tree.subband_origin(level, sb);

            let vals: Vec<f32> = (0..sh)
                .flat_map(|r| {
                    (0..sw).map(move |c| tree.coeffs[(row_off + r) * w + (col_off + c)].abs())
                })
                .collect();
            let bytes = to_bytes(&vals);

            for (i, &v) in bytes.iter().enumerate() {
                let (r, c) = (i / sw + row_off, i % sw + col_off);
                out.put_pixel(c as u32, r as u32, Rgb([v, v, v]));
            }
        }
    }

    // ── 3.  Draw subband boundary lines ───────────────────────────────────
    let border_color = Rgb([60u8, 100u8, 210u8]);
    for level in 1..=tree.levels {
        let vline = (w >> level) as u32; // vertical boundary col
        let hline = (h >> level) as u32; // horizontal boundary row
        let bw = (w >> (level.saturating_sub(1))) as u32;
        let bh = (h >> (level.saturating_sub(1))) as u32;

        for r in 0..bh {
            if vline < out.width() {
                out.put_pixel(vline, r, border_color);
            }
        }
        for c in 0..bw {
            if hline < out.height() {
                out.put_pixel(c, hline, border_color);
            }
        }
    }

    // ── 4.  Red overlay on marked nodes ───────────────────────────────────
    for node in &tree.marked {
        let (row_off, col_off) = tree.subband_origin(node.level, node.subband);
        let abs_r = (row_off + node.row) as u32;
        let abs_c = (col_off + node.col) as u32;
        if abs_r < out.height() && abs_c < out.width() {
            let existing = *out.get_pixel(abs_c, abs_r);
            // blend: keep green/blue channels dimmed, pump red
            out.put_pixel(
                abs_c,
                abs_r,
                Rgb([
                    255,
                    (existing[1] as u32 * 2 / 5) as u8,
                    (existing[2] as u32 * 2 / 5) as u8,
                ]),
            );
        }
    }

    // ── 5.  Crop to original size if we padded ────────────────────────────
    if orig_w < w || orig_h < h {
        image::imageops::crop_imm(&out, 0, 0, orig_w as u32, orig_h as u32).to_image()
    } else {
        out
    }
}

/// Reconstruct a grayscale image from only the marked wavelet coefficients.
fn reconstruct(tree: &WaveletTree, orig_w: usize, orig_h: usize) -> GrayImage {
    let masked = tree.masked_coeffs();
    let restored = haar::haar_2d_inv(&masked, tree.width, tree.height, tree.levels);

    let mut img = GrayImage::new(orig_w as u32, orig_h as u32);
    for r in 0..orig_h {
        for c in 0..orig_w {
            let v = restored[r * tree.width + c].clamp(0.0, 255.0) as u8;
            img.put_pixel(c as u32, r as u32, Luma([v]));
        }
    }
    img
}

// ── Entry point ──────────────────────────────────────────────────────────────

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();

    // ── Load image ──────────────────────────────────────────────────────────
    let (pixels, orig_w, orig_h) = load_gray_f32(&args.input)?;
    println!(
        "▸ Loaded   {}  ({}×{} px, grayscale)",
        args.input.display(),
        orig_w,
        orig_h
    );

    // ── Pad to power of two ─────────────────────────────────────────────────
    let (padded, pad_w, pad_h) = pad_power_of_two(&pixels, orig_w, orig_h);
    if pad_w != orig_w || pad_h != orig_h {
        println!(
            "▸ Padded   {}×{} → {}×{} (next power of two)",
            orig_w, orig_h, pad_w, pad_h
        );
    }

    // ── Clamp levels to what the image can support ──────────────────────────
    let max_levels = pad_w.min(pad_h).trailing_zeros() as usize;
    if args.levels > max_levels {
        eprintln!(
            "⚠  Requested {} levels but image supports at most {}; clamping.",
            args.levels, max_levels
        );
    }
    let levels = args.levels.min(max_levels);

    // ── Forward DWT ─────────────────────────────────────────────────────────
    let coeffs = haar::haar_2d_fwd(&padded, pad_w, pad_h, levels);
    println!(
        "▸ Haar DWT  {} levels  (coarsest LL: {}×{})",
        levels,
        pad_w >> levels,
        pad_h >> levels
    );

    // ── Build tree ──────────────────────────────────────────────────────────
    let mut tree = WaveletTree::new(coeffs, pad_w, pad_h, levels);

    // ── BBox selection ──────────────────────────────────────────────────────
    if let Some(ref bbox) = args.bbox {
        let (x0, y0) = (bbox[0], bbox[1]);
        let (x1, y1) = (
            bbox[2].min(orig_w as u32 - 1),
            bbox[3].min(orig_h as u32 - 1),
        );

        println!("▸ BBox     ({x0},{y0}) → ({x1},{y1})");
        let new_nodes = tree.mark_bbox(x0, y0, x1, y1);
        println!(
            "▸ Marked   {} nodes (direct bbox coverage)",
            new_nodes.len()
        );

        if args.expand_ancestors {
            let before = tree.marked.len();
            tree.mark_ancestors();
            println!("▸ Expanded +{} ancestor nodes", tree.marked.len() - before);
        }
        if args.expand_descendants {
            let before = tree.marked.len();
            tree.mark_descendants();
            println!(
                "▸ Expanded +{} descendant nodes",
                tree.marked.len() - before
            );
        }

        println!();
        tree.print_marked_report();
    }

    // ── Optional: debug grids ───────────────────────────────────────────────
    if args.debug_grids {
        let max_print_cols = 16;
        for level in (1..=levels).rev() {
            for &sb in Subband::all().iter() {
                if sb == Subband::LL && level < levels {
                    continue;
                }
                tree.print_subband_grid(level, sb, max_print_cols);
            }
        }
    }

    // ── Visualization ───────────────────────────────────────────────────────
    let vis = visualize(&tree, orig_w, orig_h);
    vis.save(&args.output)?;
    println!("▸ Saved    visualization → {}", args.output.display());

    // ── Optional: reconstruction ─────────────────────────────────────────────
    if let Some(ref recon_path) = args.reconstruct {
        if tree.marked.is_empty() {
            eprintln!("⚠  No nodes marked; skipping reconstruction.");
        } else {
            let recon_img = reconstruct(&tree, orig_w, orig_h);
            recon_img.save(recon_path)?;
            println!("▸ Saved    reconstruction → {}", recon_path.display());
        }
    }

    Ok(())
}
