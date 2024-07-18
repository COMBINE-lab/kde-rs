use ordered_float::OrderedFloat;
use rand::distributions::{Distribution, Uniform};
use rand::prelude::*;
use std::time::Instant;

use kders::kde::GridDimensions;
use ndarray::Array2;
use numpy::IntoPyArray;
use numpy::{PyArray1, PyArray2};
use pyo3::prelude::*;
use pyo3::prepare_freethreaded_python;

use std::io;

pub fn kde_computation_py(
    data: &[f64],
    weight: &[f64],
    bin_width: f64, //store: &mut InMemoryAlignmentStore,
    bandwidth: Option<f64>,
    grid_dim: Option<GridDimensions>,
) -> io::Result<Vec<f64>> {
    // Initialize the Python interpreter
    prepare_freethreaded_python();

    Python::with_gil(|py| {
        // Import the Python module and function
        let module = PyModule::from_code_bound(
            py,
            include_str!("2D_kde_function.py"),
            "2D_kde_function",
            "2D_kde_function.py",
        )?;
        let function = module.getattr("calculate_kde")?;

        // Example 2D data
        let data = Array2::from_shape_vec(((data.len() / 2), 2), data.to_vec()).unwrap();

        // Convert data to PyArray2
        let data_py: pyo3::Bound<'_, PyArray2<f64>> = data.view().to_owned().into_pyarray_bound(py);

        // Dummy weights for demonstration purposes
        let weights = Array2::from_shape_vec((weight.len(), 1), weight.to_vec()).unwrap();

        // Convert weights to PyArray2
        let weights_py: pyo3::Bound<'_, PyArray2<f64>> =
            weights.view().to_owned().into_pyarray_bound(py);

        let (max_x, max_y) = match grid_dim {
            Some(GridDimensions { width, height }) => (Some(width), Some(height)),
            None => (None, None),
        };
        // Call the Python function
        println!("it is before python function");
        let result: Py<PyArray1<f64>> = function
            .call1((data_py, weights_py, bin_width, bandwidth, max_x, max_y))?
            .extract()?;
        //println!("result length: {:?}", result.as_ref().len());
        let r: Vec<f64> = result.extract(py)?;
        Ok(r)
    })
}
fn main() -> anyhow::Result<()> {
    let mut rng = rand::thread_rng();

    let max_val = 1000;

    let die = Uniform::from(10.0..(max_val as f64));
    let mut data = Vec::<f64>::with_capacity(200);
    let mut weights = Vec::<f64>::with_capacity(100);
    let mut weight_sum = 0_f64;

    let kernel_bandwidth = 5_f64;
    let bin_width = 10_usize;

    for _i in 0..1000 {
        let x: f64 = die.sample(&mut rng);
        let y: f64 = die.sample(&mut rng);
        data.push(x.round());
        data.push(y.round());
        let w: f64 = rng.gen();
        weights.push(w.round());
        weight_sum += weights.last().unwrap();
    }

    let gd = kders::kde::GridDimensions {
        width: max_val as usize,
        height: max_val as usize,
    };

    //python kde fucntion
    let py_start = Instant::now();
    let py_res = kde_computation_py(
        &data,
        &weights,
        bin_width as f64,
        Some(kernel_bandwidth),
        Some(gd),
    )?;
    let py_duration = py_start.elapsed();

    // curently have to normalize the weights for the rust impl
    /*
    weights.iter_mut().for_each(|x| {
        *x /= weight_sum;
    });
    */

    //println!("grid \n\n");
    /*
    let mut grid = crate::kde::KDEGrid::from_data_with_binwidth(
        &data.iter().map(|x| *x as u64).collect::<Vec<u64>>(),
        &weights,
        5,
    );
    */

    let rust_start = Instant::now();
    let mut grid = kders::kde::KDEGrid::new(gd, bin_width, Some(kernel_bandwidth));
    for (chunk, w) in data.chunks(2).zip(weights.iter()) {
        grid.add_observation(chunk[0] as usize, chunk[1] as usize, *w);
    }

    let density = grid.get_kde()?;
    //println!("rust kde matrix: {:+e}", density.data);

    let mut lookups = Vec::<f64>::with_capacity(100);
    for chunk in data.chunks(2) {
        //lookups.push(density.query_interp((chunk[0] as usize, chunk[1] as usize)));
        lookups.push(density[(chunk[0] as usize, chunk[1] as usize)]);
    }
    let rust_duration = rust_start.elapsed();
    //println!("py: {:?}", py_res);
    //println!("rust: {:?}", lookups);
    //println!("total denisty of kde is : {}", density.sum());

    let a = py_res
        .iter()
        .map(|x| OrderedFloat(*x))
        .collect::<Vec<OrderedFloat<f64>>>();
    let b = lookups
        .iter()
        .map(|x| OrderedFloat(*x))
        .collect::<Vec<OrderedFloat<f64>>>();

    let (tau_b, significance) = kendalls::tau_b(&a, &b)?;
    println!("tau_b: {}, sig: {}", tau_b, significance);
    println!(
        "time py: {}, time rust: {}",
        py_duration.as_millis(),
        rust_duration.as_millis()
    );
    /*
    println!(
        "diffs = {:?}",
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| {
                let r: f64 = (*x - *y).into();
                r
            })
            .collect::<Vec<f64>>()
    );
    */
    /*
    let mut grid = crate::kde::KDEGrid::new(6, 6, 1);

    grid.add_observation(1, 2, 1.0);
    grid.add_observation(2, 3, 2.0);
    grid.add_observation(3, 4, 3.0);
    grid.add_observation(4, 5, 4.0);

    let density = grid.evaluate_kde()?;

    let mut lookups = Vec::<f64>::with_capacity(4);
    lookups.push(density[(1, 2)]);
    lookups.push(density[(2, 3)]);
    lookups.push(density[(3, 4)]);
    lookups.push(density[(4, 5)]);
    println!("{:?}", lookups);

    println!("total denisty of kde is : {}", density.sum());

    println!("grid 2\n\n");
    let mut grid = crate::kde::KDEGrid::from_data_with_binwidth(
        &data.iter().map(|x| *x as u64).collect::<Vec<u64>>(),
        &weights,
        1,
    );

    let density = grid.evaluate_kde()?;

    let mut lookups = Vec::<f64>::with_capacity(4);
    lookups.push(density[(1, 2)]);
    lookups.push(density[(2, 3)]);
    lookups.push(density[(3, 4)]);
    lookups.push(density[(4, 5)]);
    println!("{:?}", lookups);
    println!("total denisty of kde is : {}", density.sum());
    */
    Ok(())
}
