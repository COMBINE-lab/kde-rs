//! The top-level module for the simple 2D KDE crate.  This crate (currently) makes many
//! simplifying assumptions.
//!
//!  - All input locations are integers
//!  - The only supported kernel is a symmetric Gaussian
//!  - XXX
//!

use core::f64::consts::PI;
use ndarray::Array2;

const KDE_EPSILON: f64 = 2.220446049250313e-16;

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

/// Yes, it has the same members as
/// the [KDEGrid] but this is the
/// evaluated density, not the observed
/// weights
#[allow(unused)]
pub struct KDEModel {
    width: usize,
    height: usize,
    bin_width: usize,
    pub data: Array2<f64>,
}

/// Functionality provided by the [KDEModel]
impl KDEModel {
    // The total weight of the model
    #[allow(unused)]
    pub fn sum(&self) -> f64 {
        self.data.sum()
    }
}

impl std::ops::Index<(usize, usize)> for KDEModel {
    type Output = f64;
    fn index(&self, index: (usize, usize)) -> &Self::Output {
        let blx = index.0 / self.bin_width;
        let bly = index.1 / self.bin_width;
        /*
        println!(
            "d = {}",
            ((index.0 as f64 - ((blx * self.bin_width) as f64 + 5.)).powi(2)
                + (index.1 as f64 - ((bly * self.bin_width) as f64 + 5.)).powi(2))
            .sqrt()
        );*/
        &self.data[[blx, bly]]
    }
}

impl KDEGrid {
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
    pub fn from_data_with_binwidth(data: &[u64], weights: &[f64], bin_width: usize) -> Self {
        let n = data.len() / 2;
        let data = Array2::from_shape_vec(((data.len() / 2), 2), data.to_vec()).unwrap();

        let (mut max_x, mut max_y) = (0_u64, 0_u64);
        for i in 0..n {
            max_x = data[[i, 0]].max(max_x);
            max_y = data[[i, 1]].max(max_y);
        }

        let gd = GridDimensions {
            width: (max_x + 1) as usize,
            height: (max_y + 1) as usize,
        };

        let mut grid = Self::new(gd, bin_width, None);
        for i in 0..n {
            grid.add_observation(data[[i, 0]] as usize, data[[i, 1]] as usize, weights[i]);
        }
        grid
    }

    #[inline(always)]
    pub fn add_observation(&mut self, x: usize, y: usize, w: f64) {
        let bx = x as i64 / self.bin_width as i64;
        let by = y as i64 / self.bin_width as i64;

        let (x_cells, y_cells) = self.data.dim();
        let x_cells = x_cells as i64;
        let y_cells = y_cells as i64;

        let bandwidth = self.bandwidth; //1.0_f64;
        let bwf = self.bin_width as f64;
        let half_bin_width = bwf / 2.0;

        let dist_thresh = 10. * bandwidth; // number of standard deviations

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

                let dweight = w;
                if distance_sq.sqrt() <= dist_thresh {
                    let contrib = dweight * (-distance_sq / (2.0 * bandwidth)).exp() * kernel_norm;
                    self.kde_matrix[[i1 as usize, j1 as usize]] += contrib;
                }
            }
        }
    }

    pub fn evaluate_kde(&mut self) -> anyhow::Result<KDEModel> {
        // println!("rust kde_sum = {:?}", self.kde_matrix.sum());
        let mut new_kde_matrix = self.kde_matrix.clone() + KDE_EPSILON;
        new_kde_matrix /= new_kde_matrix.sum();

        Ok(KDEModel {
            width: self.width,
            height: self.height,
            bin_width: self.bin_width,
            data: new_kde_matrix,
        })
    }
}
