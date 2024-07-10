use ndarray::{Array2};
use ndarray::prelude::*;
use core::f64::consts::PI;

use std::{
    io::{self, BufReader}
};

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
    let x_max = data.column(0).iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b)) + 1.0;
    let y_min = data.column(1).iter().fold(f64::INFINITY, |a, &b| a.min(b)) - 1.0;
    let y_max = data.column(1).iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b)) + 1.0;


    //eprintln!("3rd print");
    let num_bins = 4; //2000;
    //compute the edges of the 2000 bins in x and y axis
    let x_grid = Array::linspace(x_min, x_max, num_bins + 1);
    let y_grid = Array::linspace(y_min, y_max, num_bins + 1);

    //eprintln!("4th print");
    let x_centers = x_grid.windows(2).into_iter().map(|window| { (window[0] + window[1]) / 2.0 }).collect::<Vec<f64>>();
    let y_centers = y_grid.windows(2).into_iter().map(|window| { (window[0] + window[1]) / 2.0 }).collect::<Vec<f64>>();

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