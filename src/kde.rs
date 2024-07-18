//! The top-level module for the simple 2D KDE crate.  This crate (currently) makes many
//! simplifying assumptions.
//!
//!  - All input locations are integers
//!  - The only supported kernel is a symmetric Gaussian
//!  - Only 2D estimation is supported
//!

use core::f64::consts::PI;
use ndarray::Array2;

/// Value of `np.finfo(float).eps` taken from
/// [KDEPy](https://github.com/tommyod/KDEpy/blob/d7bea3fa65c8a7373188983cf6767854be42c798/KDEpy/BaseKDE.py#L215C41-L215C60).
const KDE_EPSILON: f64 = 2.220446049250313e-16;

/// Records the dimensions of the grid on which
/// estimation will be carried out.  The grid is
/// always assumed to start at (0, 0), and so the
/// width and height correspond to the maximal
/// x and y values for which density will be
/// estimated.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GridDimensions {
    pub width: usize,
    pub height: usize,
}

/// Holds relevant information about a
/// grid of (weighted) samples and the
/// eventual associated KDE matrix.
///
/// Assumes that the bin width is the
/// same in both the x and y dimensions.
pub struct KDEGrid {
    width: usize,
    height: usize,
    bin_width: usize,
    bandwidth: f64,
    data: Array2<f64>,
    kde_matrix: Array2<f64>,
}

/// A KDE model that returns the estimate for each queried
/// point.
#[allow(unused)]
pub struct KDEModel {
    width: usize,
    height: usize,
    bin_width: usize,
    bandwidth: f64,
    pub data: Array2<f64>,
}

/// Functionality provided by the [KDEModel]
impl KDEModel {
    // The total weight of the model
    #[allow(unused)]
    pub fn sum(&self) -> f64 {
        self.data.sum()
    }

    #[inline(always)]
    pub fn try_query_bin(&self, query: (i64, i64)) -> Option<f64> {
        let (bins_x, bins_y) = self.data.dim();
        let (x, y) = query;
        if x >= 0 && y >= 0 && (x as usize) < bins_x && (y as usize) < bins_y {
            Some(self.data[[x as usize, y as usize]])
        } else {
            None
        }
    }
}

impl std::ops::Index<(usize, usize)> for KDEModel {
    type Output = f64;
    /// Returns the density estimtae associated with the provided
    /// point `index`.  This model performs nearest-neighbor lookup
    /// and so the reutrned estimate is just the KDE value recorded
    /// at the nearest grid point.
    #[inline(always)]
    fn index(&self, index: (usize, usize)) -> &Self::Output {
        let blx = index.0 / self.bin_width;
        let bly = index.1 / self.bin_width;
        let (bins_x, bins_y) = self.data.dim();
        if blx >= bins_x || bly >= bins_y {
            &KDE_EPSILON
        } else {
            &self.data[[blx, bly]]
        }
    }
}

impl KDEModel {
    /// Returns the density estimtae associated with the provided
    /// point `index`.  This model performs interpolated lookup
    /// and so the reutrned estimate is the weighted average
    /// of the neighboring grid points.
    #[allow(dead_code)]
    #[inline(always)]
    pub fn query_interp(&self, index: (usize, usize)) -> f64 {
        // the actual bin in which our query point falls must be valid
        let blx = index.0 / self.bin_width;
        let bly = index.1 / self.bin_width;
        let (bins_x, bins_y) = self.data.dim();
        if blx >= bins_x || bly >= bins_y {
            KDE_EPSILON
        } else {
            let bandwidth = self.bandwidth;
            let bw_sq = bandwidth * bandwidth;
            let bwf = self.bin_width as f64;
            let half_bin_width = (bwf) / 2.;
            let kernel_norm = 1.0 / (2.0 * PI * bw_sq);
            let sq_distance = |x: f64, y: f64, nx: i64, ny: i64| {
                let px = (nx as f64 * bwf) + half_bin_width;
                let py = (ny as f64 * bwf) + half_bin_width;
                let dx = x - px;
                let dy = y - py;
                dx * dx + dy * dy
            };
            // here the query bin is valid, so get the neighboring bins.
            let blx = blx as i64;
            let bly = bly as i64;
            let left = blx - 1;
            let right = blx + 1;
            let down = bly - 1;
            let up = bly + 1;
            let x = index.0 as f64;
            let y = index.1 as f64;
            let mut weighted_density = 0_f64;
            let mut denom = 0_f64;
            // neighbors to query
            for (nx, ny) in [
                (blx, bly),
                (left, bly),
                (right, bly),
                (blx, down),
                (blx, up),
                (left, down),
                (right, down),
                (left, up),
                (right, up),
            ] {
                if let Some(sample_d) = self.try_query_bin((nx, ny)) {
                    let weight = ((sq_distance(x, y, nx, ny) / bw_sq) * 0.5).exp() * kernel_norm;
                    denom += weight;
                    weighted_density += sample_d * weight;
                }
            }
            if denom > 0. {
                weighted_density / denom
            } else {
                KDE_EPSILON
            }
        }
    }
}

impl KDEGrid {
    /// Construct a new [KDEGrid] having the dimensions specified by `grid_dim`
    /// with bins of width `bin_width`. Optionally, a bandwidth may be provided
    /// via the `bandwidth` parameter; if [None] is provided, the bandwidth is
    /// set to 1.
    pub fn new(grid_dim: GridDimensions, bin_width: usize, bandwidth: Option<f64>) -> Self {
        let calc_num_bins = |extent: usize, bw: usize| -> usize {
            if extent % bw == 0 {
                extent / bw
            } else {
                (extent / bw) + 1
            }
        };

        let num_x_bins = calc_num_bins(grid_dim.width, bin_width) + 1;
        let num_y_bins = calc_num_bins(grid_dim.height, bin_width) + 1;

        KDEGrid {
            width: grid_dim.width,
            height: grid_dim.height,
            bin_width,
            bandwidth: bandwidth.unwrap_or(1.0),
            data: Array2::zeros((num_x_bins, num_y_bins)),
            kde_matrix: Array2::zeros((num_x_bins, num_y_bins)),
        }
    }

    #[allow(unused)]
    /// Construct a new [KDEGrid] and populate it with the provided data and weights.
    /// If the optional `grid_dim` argument is provided, it is used to set the grid
    /// dimensions. Otherwise, the maxima within the `data` parameter are used.
    /// Likewise, the optional bandwidth parameter can be passed, if None, the bandwidth
    /// is set to 1.
    pub fn from_data_with_binwidth(
        data: &[u64],
        weights: &[f64],
        grid_dim: Option<GridDimensions>,
        bin_width: usize,
        bandwidth: Option<f64>,
    ) -> Self {
        let n = data.len() / 2;
        let data = Array2::from_shape_vec(((data.len() / 2), 2), data.to_vec()).unwrap();

        let (mut max_x, mut max_y) = (0_u64, 0_u64);
        for i in 0..n {
            max_x = data[[i, 0]].max(max_x);
            max_y = data[[i, 1]].max(max_y);
        }

        let gd = match grid_dim {
            Some(GridDimensions { width, height }) => grid_dim.unwrap(),
            None => GridDimensions {
                width: (max_x + 1) as usize,
                height: (max_y + 1) as usize,
            },
        };

        let mut grid = Self::new(gd, bin_width, bandwidth);
        for i in 0..n {
            grid.add_observation(data[[i, 0]] as usize, data[[i, 1]] as usize, weights[i]);
        }
        grid
    }

    /// Adds the weighted observation at position (`x`, `y`) to the current
    /// [KDEGrid] with weight `w`.  **Note**: If (`x`, `y`) is outside of the
    /// bounds of this [KDEGrid], then no contribution is added (i.e. this point
    /// is skipped).
    #[inline(always)]
    pub fn add_observation(&mut self, x: usize, y: usize, w: f64) {
        let bx = x as i64 / self.bin_width as i64;
        let by = y as i64 / self.bin_width as i64;

        let (x_cells, y_cells) = self.data.dim();
        let x_cells = x_cells as i64;
        let y_cells = y_cells as i64;

        // exit if this point is out of bounds
        if bx >= x_cells || by >= y_cells {
            return;
        }

        let bandwidth = self.bandwidth;
        let bwf = self.bin_width as f64;
        let half_bin_width = bwf / 2.0;

        let dist_thresh = 10. * bandwidth; // number of standard deviations
        let bw_sq = bandwidth * bandwidth;

        // NOTE: missing sqrt below --- currently to match Python impl
        let kernel_norm = 1.0 / (2.0 * PI * bandwidth.powi(2));
        let half_width = (dist_thresh / (self.bin_width as f64) + 1.) as i64;
        for i1 in (0.max(bx - half_width))..(x_cells.min(bx + half_width)) {
            let k1 = bwf * i1 as f64 + half_bin_width;
            for j1 in (0.max(by - half_width))..(y_cells.min(by + half_width)) {
                let k2 = bwf * j1 as f64 + half_bin_width;
                //println!("({}, {}) to ({}, {})", x, y, k1, k2);

                let dx = k1 - x as f64;
                let dy = k2 - y as f64;
                let distance_sq = dx * dx + dy * dy;
                let distance = distance_sq.sqrt();

                let dweight = w;
                let py_dist_sq = distance_sq / bw_sq;
                if distance <= dist_thresh {
                    let contrib = dweight * (-py_dist_sq / 2.0).exp() * kernel_norm;
                    self.kde_matrix[[i1 as usize, j1 as usize]] += contrib;
                }
            }
        }
    }

    /// This function is called once all points have been added to the
    /// [KDEGrid], the resulting density estimate is normalized, and
    /// a [KDEModel] is returned.  The result is `OK(`[KDEModel]`)`
    /// if a valid [KDEModel] can be returned, and [anyhow::Error] otherwise.
    pub fn get_kde(&mut self) -> anyhow::Result<KDEModel> {
        let mut new_kde_matrix = self.kde_matrix.clone() + KDE_EPSILON;
        new_kde_matrix /= new_kde_matrix.sum();

        Ok(KDEModel {
            width: self.width,
            height: self.height,
            bin_width: self.bin_width,
            bandwidth: self.bandwidth,
            data: new_kde_matrix,
        })
    }
}
