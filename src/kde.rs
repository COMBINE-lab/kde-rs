use core::f64::consts::PI;
use ndarray::prelude::*;
use ndarray::Array2;

use std::io::{self, BufReader};

/// Assumes that the bin width is the
/// same in both the x and y dimensions.
pub struct KDEGrid {
    width: usize,
    height: usize,
    bin_width: usize,
    data: Array2<f64>,
}

/// Yes, it has the same members as
/// the [KDEGrid] but this is the
/// evaluated density, not the observed
/// weights
pub struct KDEModel {
    width: usize,
    height: usize,
    bin_width: usize,
    data: Array2<f64>,
}

impl std::ops::Index<(usize, usize)> for KDEModel {
    type Output = f64;
    fn index(&self, index: (usize, usize)) -> &Self::Output {
        let bx = index.0 / self.bin_width;
        let by = index.1 / self.bin_width;
        &self.data[[bx, by]]
    }
}

impl KDEGrid {
    pub fn new(width: usize, height: usize, bin_width: usize) -> Self {
        let calc_num_bins = |extent: usize, bw: usize| -> usize {
            if extent % bw == 0 {
                extent / bw
            } else {
                (extent / bw) + 1
            }
        };

        let num_x_bins = calc_num_bins(width, bin_width);
        let num_y_bins = calc_num_bins(height, bin_width);

        KDEGrid {
            width,
            height,
            bin_width,
            data: Array2::zeros((num_x_bins, num_y_bins)),
        }
    }

    #[inline(always)]
    pub fn add_observation(&mut self, x: usize, y: usize, w: f64) {
        let bx = x / self.bin_width;
        let by = y / self.bin_width;
        self.data[[bx, by]] += w;
    }

    pub fn evaluate_kde(&mut self) -> anyhow::Result<KDEModel> {
        let (x_cells, y_cells) = self.data.dim();

        let bandwidth = 1.0_f64;
        let bwf = self.bin_width as f64;
        //eprintln!("5th print");
        let mut kde_matrix = Array2::<f64>::zeros((x_cells, y_cells));
        let radius_x = bwf; //x_centers[1] - x_centers[0];
        let radius_y = bwf; //y_centers[1] - y_centers[0];

        eprintln!("6th print");
        for i in 0..x_cells {
            //}(i, &u) in x_centers.iter().enumerate() {
            let u = bwf * i as f64 + ((bwf) / 2.0);
            for j in 0..y_cells {
                let v = bwf * j as f64 + ((bwf) / 2.0);
                let mut sum = 0.0;
                let mut weigh_sum = 0.0;
                for i1 in 0..x_cells {
                    let k1 = bwf * i1 as f64 + ((bwf) / 2.0);
                    for j1 in 0..y_cells {
                        let k2 = bwf * j1 as f64 + ((bwf) / 2.0);

                        let dx = k1 - u;
                        let dy = k2 - v;
                        let distance = dx * dx + dy * dy;
                        if dx.abs() <= (5.0 * radius_x) as f64
                            && dy.abs() <= (5.0 * radius_y) as f64
                        {
                            sum +=
                                self.data[[i1, j1]] * (-distance / (2.0 * bandwidth.powi(2))).exp();
                            weigh_sum += self.data[[i1, j1]]
                        }
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
        println!("{:?}", kde_matrix);

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

