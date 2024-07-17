//! The top-level module for the simple 2D KDE crate.  This crate (currently) makes many
//! simplifying assumptions.
//!
//!  - All input locations are integers
//!  - The only supported kernel is a symmetric Gaussian
//!  - XXX
//!

use core::f64::consts::PI;
use ndarray::prelude::*;
use ndarray::Array2;

use std::io;

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
pub struct KDEModel {
    width: usize,
    height: usize,
    bin_width: usize,
    pub data: Array2<f64>,
}

/// Functionality provided by the [KDEModel]
impl KDEModel {
    // The total weight of the model
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
    pub fn new(width: usize, height: usize, bin_width: usize, bandwidth: Option<f64>) -> Self {
        let calc_num_bins = |extent: usize, bw: usize| -> usize {
            if extent % bw == 0 {
                extent / bw
            } else {
                (extent / bw) + 1
            }
        };

        let num_x_bins = calc_num_bins(width, bin_width) + 1;
        let num_y_bins = calc_num_bins(height, bin_width) + 1;

        KDEGrid {
            width,
            height,
            bin_width,
            bandwidth: bandwidth.unwrap_or(1.0),
            data: Array2::zeros((num_x_bins, num_y_bins)),
            kde_matrix: Array2::zeros((num_x_bins, num_y_bins)),
        }
    }

    pub fn from_data_with_binwidth(data: &[u64], weights: &[f64], bin_width: usize) -> Self {
        let n = data.len() / 2;
        let data = Array2::from_shape_vec(((data.len() / 2), 2), data.to_vec()).unwrap();

        let (mut max_x, mut max_y) = (0_u64, 0_u64);
        for i in 0..n {
            max_x = data[[i, 0]].max(max_x);
            max_y = data[[i, 1]].max(max_y);
        }

        let mut grid = Self::new((max_x + 1) as usize, (max_y + 1) as usize, bin_width, None);
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

        let dist_thresh = 10. * bandwidth; // 4 standard deviations

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
        let mut new_kde_matrix = self.kde_matrix.clone() + 2.220446049250313e-16;
        new_kde_matrix /= new_kde_matrix.sum();

        return Ok(KDEModel {
            width: self.width,
            height: self.height,
            bin_width: self.bin_width,
            data: new_kde_matrix,
        });

        let (x_cells, y_cells) = self.data.dim();
        let x_cells = x_cells as i64;
        let y_cells = y_cells as i64;

        let bandwidth = 1.0_f64;
        let bwf = self.bin_width as f64;

        let mut kde_matrix = Array2::<f64>::zeros((x_cells as usize, y_cells as usize));
        let radius_x = bwf; //x_centers[1] - x_centers[0];
        let radius_y = bwf; //y_centers[1] - y_centers[0];

        let half_width = 10_i64;
        eprintln!("6th print");
        for i in 0..x_cells {
            //}(i, &u) in x_centers.iter().enumerate() {
            let u = bwf * i as f64 + ((bwf) / 2.0);
            for j in 0..y_cells {
                let v = bwf * j as f64 + ((bwf) / 2.0);
                //println!("[({}, {})]", u, v);
                let mut sum = 0.0;
                let mut weigh_sum = 0.0;
                for i1 in (0.max(i - half_width))..(x_cells.min(i + half_width)) {
                    let k1 = bwf * i1 as f64 + ((bwf) / 2.0);
                    for j1 in (0.max(j - half_width))..(y_cells.min(j + half_width)) {
                        let k2 = bwf * j1 as f64 + ((bwf) / 2.0);

                        let dx = k1 - u;
                        let dy = k2 - v;
                        let distance_sq = dx * dx + dy * dy;
                        let distance = distance_sq.sqrt();

                        let dweight = self.data[[i1 as usize, j1 as usize]];

                        //if dx.abs() <= (10.0 * radius_x) as f64
                        //&& dy.abs() <= (10.0 * radius_y) as f64
                        if distance <= 200.0 * radius_x {
                            sum += dweight * (-distance_sq / (2.0 * bandwidth.powi(2))).exp();
                            weigh_sum += 1.; //dweight;
                        }
                    }
                }
                kde_matrix[[i as usize, j as usize]] = if weigh_sum > 0.0 {
                    sum / ((2.0 * PI * bandwidth.powi(2)).sqrt() * weigh_sum)
                } else {
                    0.0
                };
            }
        }

        // Normalize kde_matrix
        //eprintln!("kde_matrix: {:?}", kde_matrix);
        let total_sum: f64 = kde_matrix.sum();
        //kde_matrix /= total_sum;
        //eprintln!("kde_matrix2: {:?}", kde_matrix);
        eprintln!("kde summation: {:?}", kde_matrix.sum());
        //println!("{:?}", kde_matrix);

        Ok(KDEModel {
            width: self.width,
            height: self.height,
            bin_width: self.bin_width,
            data: kde_matrix,
        })
    }
}

pub fn kde_computation(
    data: &Vec<f64>,
    weight: &Vec<f64>,
    //store: &mut InMemoryAlignmentStore,
) -> io::Result<()> {
    //eprintln!("1st print");
    //Define data and weight as a 2D array
    let n = data.len() / 2;
    let data = Array2::from_shape_vec(((data.len() / 2), 2), data.to_vec()).unwrap();
    let weights = Array2::from_shape_vec((weight.len(), 1), weight.to_vec()).unwrap();

    //Choose the bandwidth for kde calculation
    let bandwidth: f64 = 1.0;

    //eprintln!("2nd print");
    // Calculate min and max for each dimension
    let x_min = data.column(0).iter().fold(f64::INFINITY, |a, &b| a.min(b)) - 1.0;
    let x_max = data
        .column(0)
        .iter()
        .fold(f64::NEG_INFINITY, |a, &b| a.max(b))
        + 1.0;
    let y_min = data.column(1).iter().fold(f64::INFINITY, |a, &b| a.min(b)) - 1.0;
    let y_max = data
        .column(1)
        .iter()
        .fold(f64::NEG_INFINITY, |a, &b| a.max(b))
        + 1.0;

    //eprintln!("3rd print");
    let num_bins = 4; //2000;
                      //compute the edges of the 2000 bins in x and y axis
    let x_grid = Array::linspace(x_min, x_max, num_bins + 1);
    let y_grid = Array::linspace(y_min, y_max, num_bins + 1);

    //eprintln!("4th print");
    let x_centers = x_grid
        .windows(2)
        .into_iter()
        .map(|window| (window[0] + window[1]) / 2.0)
        .collect::<Vec<f64>>();
    let y_centers = y_grid
        .windows(2)
        .into_iter()
        .map(|window| (window[0] + window[1]) / 2.0)
        .collect::<Vec<f64>>();

    //eprintln!("5th print");
    let mut kde_matrix = Array2::<f64>::zeros((num_bins, num_bins));
    let radius_x = x_centers[1] - x_centers[0];
    let radius_y = y_centers[1] - y_centers[0];

    eprintln!("6th print");
    for (i, &u) in x_centers.iter().enumerate() {
        for (j, &v) in y_centers.iter().enumerate() {
            let mut sum = 0.0;
            let mut weigh_sum = 0.0;
            for k in 0..n {
                let dx = data[[k, 0]] - u;
                let dy = data[[k, 1]] - v;
                let distance = dx * dx + dy * dy;
                if dx.abs() <= (5.0 * radius_x) as f64 && dy.abs() <= (5.0 * radius_y) as f64 {
                    sum += weights[[k, 0]] * (-distance / (2.0 * bandwidth.powi(2))).exp();
                    weigh_sum += weights[[k, 0]]
                }
            }
            kde_matrix[[i, j]] = sum / ((2.0 * PI * bandwidth.powi(2)).sqrt() * weigh_sum);
        }
    }

    // Normalize kde_matrix
    //eprintln!("kde_matrix: {:?}", kde_matrix);
    let total_sum: f64 = kde_matrix.sum();
    kde_matrix /= total_sum;
    eprintln!("kde_matrix2: {:?}", kde_matrix);
    eprintln!("kde ummation: {:?}", kde_matrix.sum());

    eprintln!("7th print");
    //compute the kde values for each datat point
    let mut bin_indices = Vec::with_capacity(n);
    for k in 0..n {
        let x_bin = ((data[[k, 0]] - x_min) / (x_max - x_min) * num_bins as f64).floor() as usize;
        let y_bin = ((data[[k, 1]] - y_min) / (y_max - y_min) * num_bins as f64).floor() as usize;
        bin_indices.push((x_bin, y_bin));
    }

    let mut kde_values = Vec::with_capacity(n);
    for (x_bin, y_bin) in bin_indices {
        kde_values.push(kde_matrix[[x_bin, y_bin]]);
    }

    println!("{:?}", kde_values);

    Ok(())
}
