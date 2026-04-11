// haar_wavelet/src/wavelet_tree.rs
// ────────────────────────────────────────────────────────────────────────────
// wavelet_tree.rs  –  wavelet tree data structure and spatial node selection
//
// Key ideas
// ─────────
// • A WaveletTree wraps the flat coefficient array produced by `haar_2d_fwd`.
// • Each coefficient is uniquely addressed by a NodeId {level, subband, row, col}.
// • level 1  = finest (largest) subbands; level L = coarsest.
// • The parent-child tree relationship is the standard quadtree link:
//     parent  → (level+1, row>>1, col>>1)   same subband
//     children→ (level-1, 2r, 2c) … (level-1, 2r+1, 2c+1)   same subband
// • mark_bbox() maps pixel coordinates to coefficients at *every* level by
//     coeff_col = pixel_x >> level
//     coeff_row = pixel_y >> level
//   so the selected nodes form a cone in the tree (wide at fine, narrow at coarse).
// ────────────────────────────────────────────────────────────────────────────

use std::collections::HashSet;

// ── Subband ─────────────────────────────────────────────────────────────────

/// The four wavelet subbands produced by a single 2-D Haar pass.
///
/// Naming: first letter = row direction filter, second = column direction.
///   LL – low-low   (approximation, only meaningful at the coarsest level)
///   LH – low-high  (horizontal detail, occupies top-right quadrant at level l)
///   HL – high-low  (vertical detail,   occupies bottom-left quadrant)
///   HH – high-high (diagonal detail,   occupies bottom-right quadrant)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Subband {
    LL,
    LH,
    HL,
    HH,
}

impl Subband {
    /// Human-readable short name.
    pub fn name(self) -> &'static str {
        match self {
            Self::LL => "LL",
            Self::LH => "LH",
            Self::HL => "HL",
            Self::HH => "HH",
        }
    }

    /// Brief description of what the subband encodes.
    pub fn description(self) -> &'static str {
        match self {
            Self::LL => "approximation (dc + low-frequency)",
            Self::LH => "horizontal detail (left-right edges)",
            Self::HL => "vertical detail   (top-bottom edges)",
            Self::HH => "diagonal detail   (checkerboard edges)",
        }
    }

    /// All three detail subbands in a fixed order.
    pub fn details() -> [Subband; 3] {
        [Self::LH, Self::HL, Self::HH]
    }

    /// All four subbands.
    pub fn all() -> [Subband; 4] {
        [Self::LL, Self::LH, Self::HL, Self::HH]
    }
}

impl std::fmt::Display for Subband {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.name())
    }
}

// ── NodeId ──────────────────────────────────────────────────────────────────

/// Uniquely identifies one coefficient in the wavelet pyramid.
///
/// `level` is 1-indexed: 1 = finest (largest subbands), levels = coarsest.
/// `row` and `col` are offsets *within the subband* (not within the full image).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct NodeId {
    pub level: usize,
    pub subband: Subband,
    pub row: usize,
    pub col: usize,
}

impl NodeId {
    // ── Spatial helpers ─────────────────────────────────────────────────────

    /// Pixel block in the original image that this coefficient covers.
    ///
    /// Returns `(x0, y0, x1, y1)` inclusive pixel coordinates.
    pub fn pixel_block(&self) -> (usize, usize, usize, usize) {
        let scale = 1 << self.level;
        let x0 = self.col * scale;
        let y0 = self.row * scale;
        (x0, y0, x0 + scale - 1, y0 + scale - 1)
    }

    /// Absolute `(image_row, image_col)` position of this coefficient
    /// in the flat pyramid array (stride = `img_width`).
    pub fn image_pos(&self, img_width: usize, img_height: usize) -> (usize, usize) {
        match self.subband {
            Subband::LL => (self.row, self.col),
            Subband::LH => (self.row, (img_width >> self.level) + self.col),
            Subband::HL => ((img_height >> self.level) + self.row, self.col),
            Subband::HH => (
                (img_height >> self.level) + self.row,
                (img_width >> self.level) + self.col,
            ),
        }
    }

    // ── Tree navigation ─────────────────────────────────────────────────────

    /// Parent node (one level coarser).  Returns `None` at the coarsest level.
    pub fn parent(&self, max_levels: usize) -> Option<NodeId> {
        if self.level >= max_levels {
            return None;
        }
        Some(NodeId {
            level: self.level + 1,
            subband: self.subband,
            row: self.row >> 1,
            col: self.col >> 1,
        })
    }

    /// Four children (one level finer).  Returns `None` at level 1.
    pub fn children(&self) -> Option<[NodeId; 4]> {
        if self.level <= 1 {
            return None;
        }
        let (r, c) = (self.row << 1, self.col << 1);
        Some([
            NodeId {
                level: self.level - 1,
                subband: self.subband,
                row: r,
                col: c,
            },
            NodeId {
                level: self.level - 1,
                subband: self.subband,
                row: r + 1,
                col: c,
            },
            NodeId {
                level: self.level - 1,
                subband: self.subband,
                row: r,
                col: c + 1,
            },
            NodeId {
                level: self.level - 1,
                subband: self.subband,
                row: r + 1,
                col: c + 1,
            },
        ])
    }
}

impl std::fmt::Display for NodeId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "L{}·{}[{},{}]",
            self.level, self.subband, self.row, self.col
        )
    }
}

// ── WaveletTree ─────────────────────────────────────────────────────────────

/// The central data structure: holds the coefficient pyramid and tracks which
/// nodes are currently marked (selected by a bounding-box query).
pub struct WaveletTree {
    pub width: usize,            // padded image width  (power of two)
    pub height: usize,           // padded image height (power of two)
    pub levels: usize,           // number of decomposition levels
    pub coeffs: Vec<f32>,        // wavelet coefficients in standard pyramid layout
    pub marked: HashSet<NodeId>, // currently selected nodes
}

impl WaveletTree {
    pub fn new(coeffs: Vec<f32>, width: usize, height: usize, levels: usize) -> Self {
        Self {
            width,
            height,
            levels,
            coeffs,
            marked: HashSet::new(),
        }
    }

    // ── Accessors ────────────────────────────────────────────────────────────

    /// Coefficient value for a node.
    pub fn value(&self, node: &NodeId) -> f32 {
        let (r, c) = node.image_pos(self.width, self.height);
        self.coeffs[r * self.width + c]
    }

    /// Set (mutate) coefficient value for a node.
    pub fn set_value(&mut self, node: &NodeId, v: f32) {
        let (r, c) = node.image_pos(self.width, self.height);
        self.coeffs[r * self.width + c] = v;
    }

    /// Size `(rows, cols)` of any subband at the given level.
    pub fn subband_size(&self, level: usize) -> (usize, usize) {
        (self.height >> level, self.width >> level)
    }

    /// Absolute `(row_offset, col_offset)` of a subband in the coefficient image.
    pub fn subband_origin(&self, level: usize, subband: Subband) -> (usize, usize) {
        match subband {
            Subband::LL => (0, 0),
            Subband::LH => (0, self.width >> level),
            Subband::HL => (self.height >> level, 0),
            Subband::HH => (self.height >> level, self.width >> level),
        }
    }

    // ── BBox selection ───────────────────────────────────────────────────────

    /// Mark every wavelet node whose pixel-block overlaps the given bounding box.
    ///
    /// `(x0, y0, x1, y1)` are inclusive pixel coordinates in the *original* image.
    ///
    /// At each level l, pixel p maps to coefficient index `p >> l`, so the
    /// selected coefficient range is `[x0>>l, x1>>l]` × `[y0>>l, y1>>l]`.
    ///
    /// All detail subbands (LH, HL, HH) are marked at every level.
    /// The LL subband is marked only at the coarsest level.
    ///
    /// Returns the list of newly marked nodes.
    pub fn mark_bbox(&mut self, x0: u32, y0: u32, x1: u32, y1: u32) -> Vec<NodeId> {
        let mut new_nodes = Vec::new();

        for level in 1..=self.levels {
            let (sh, sw) = self.subband_size(level);

            // Coefficient index range that overlaps [x0,x1]×[y0,y1]
            let c0 = (x0 as usize) >> level;
            let c1 = ((x1 as usize) >> level).min(sw.saturating_sub(1));
            let r0 = (y0 as usize) >> level;
            let r1 = ((y1 as usize) >> level).min(sh.saturating_sub(1));

            // Which subbands to include at this level
            let include_ll = level == self.levels;

            for &subband in Subband::all().iter() {
                if subband == Subband::LL && !include_ll {
                    continue;
                }
                for r in r0..=r1 {
                    for c in c0..=c1 {
                        let node = NodeId {
                            level,
                            subband,
                            row: r,
                            col: c,
                        };
                        if self.marked.insert(node) {
                            new_nodes.push(node);
                        }
                    }
                }
            }
        }

        new_nodes
    }

    /// Remove all marks.
    pub fn clear_marks(&mut self) {
        self.marked.clear();
    }

    // ── Tree expansion ───────────────────────────────────────────────────────

    /// Expand the current mark set to include all ancestors (coarser nodes)
    /// of every already-marked node up to the root.
    pub fn mark_ancestors(&mut self) {
        let mut to_add: Vec<NodeId> = self.marked.iter().cloned().collect();
        while let Some(node) = to_add.pop() {
            if let Some(p) = node.parent(self.levels) {
                if self.marked.insert(p) {
                    to_add.push(p);
                }
            }
        }
    }

    /// Expand the current mark set to include all descendants (finer nodes)
    /// of every already-marked node down to level 1.
    pub fn mark_descendants(&mut self) {
        let mut to_add: Vec<NodeId> = self.marked.iter().cloned().collect();
        while let Some(node) = to_add.pop() {
            if let Some(children) = node.children() {
                let (sh, sw) = self.subband_size(node.level - 1);
                for child in children {
                    if child.row < sh && child.col < sw {
                        if self.marked.insert(child) {
                            to_add.push(child);
                        }
                    }
                }
            }
        }
    }

    // ── Reconstruction helper ────────────────────────────────────────────────

    /// Return a coefficient array where every *unmarked* coefficient is zeroed.
    /// Pass this to `haar::haar_2d_inv` to reconstruct only the spatial
    /// contribution of the selected nodes.
    pub fn masked_coeffs(&self) -> Vec<f32> {
        let mut masked = vec![0.0_f32; self.coeffs.len()];
        for node in &self.marked {
            let (r, c) = node.image_pos(self.width, self.height);
            let idx = r * self.width + c;
            masked[idx] = self.coeffs[idx];
        }
        masked
    }

    // ── Reports ──────────────────────────────────────────────────────────────

    /// Print a hierarchical report of all marked nodes to stdout.
    pub fn print_marked_report(&self) {
        println!("╔══════════════════════════════════════════════════════════╗");
        println!("║           Wavelet Tree  –  Marked Node Report            ║");
        println!("╚══════════════════════════════════════════════════════════╝");
        println!("  Image   : {}×{} (padded)", self.width, self.height);
        println!("  Levels  : {}", self.levels);
        println!("  Marked  : {} nodes total", self.marked.len());
        println!();

        // Iterate coarsest → finest
        for level in (1..=self.levels).rev() {
            let scale = 1 << level;
            let subbands: &[Subband] = if level == self.levels {
                &[Subband::LL, Subband::LH, Subband::HL, Subband::HH]
            } else {
                &[Subband::LH, Subband::HL, Subband::HH]
            };

            // Check if any nodes exist at this level
            let level_count = self.marked.iter().filter(|n| n.level == level).count();
            if level_count == 0 {
                continue;
            }

            println!(
                "  ├── Level {level}  (block {}×{}, {} nodes)",
                scale, scale, level_count
            );

            for &sb in subbands {
                let mut nodes: Vec<&NodeId> = self
                    .marked
                    .iter()
                    .filter(|n| n.level == level && n.subband == sb)
                    .collect();
                if nodes.is_empty() {
                    continue;
                }

                nodes.sort();
                println!("  │   ├── {} – {}", sb, sb.description());

                for node in &nodes {
                    let val = self.value(node);
                    let (x0, y0, x1, y1) = node.pixel_block();
                    let parent_str = match node.parent(self.levels) {
                        Some(p) => format!("↑{p}"),
                        None => "root".to_string(),
                    };
                    let child_str = match node.children() {
                        Some(_) => "↓4 children".to_string(),
                        None => "leaf".to_string(),
                    };
                    println!(
                        "  │   │   [{:>3},{:>3}]  val={:>9.4}  pixels=({x0:>4},{y0:>4})–({x1:>4},{y1:>4})  {parent_str}  {child_str}",
                        node.row, node.col, val
                    );
                }
            }
        }
        println!();
    }

    /// Print a compact ASCII grid of coefficient magnitudes for one subband.
    /// Useful for debugging small images.
    pub fn print_subband_grid(&self, level: usize, subband: Subband, width: usize) {
        let (sh, sw) = self.subband_size(level);
        let sw = sw.min(width);
        println!("  {} (level {level})  {:>4}×{:<4}", subband, sh, sw);
        for r in 0..sh {
            print!("    |");
            for c in 0..sw {
                let node = NodeId {
                    level,
                    subband,
                    row: r,
                    col: c,
                };
                let v = self.value(&node);
                let mark = if self.marked.contains(&node) {
                    "*"
                } else {
                    " "
                };
                print!(" {mark}{:>6.2}", v);
            }
            println!(" |");
        }
        println!();
    }
}
