mod kde;
mod kde_function;
use crate::kde_function::kde_computation_py;

use ordered_float::OrderedFloat;
use rand::distributions::{Distribution, Uniform};
use rand::prelude::*;
use std::time::Instant;

fn main() -> anyhow::Result<()> {
    let mut rng = rand::thread_rng();
    let die = Uniform::from(10.0..500.0);
    let mut data = Vec::<f64>::with_capacity(200);
    let mut weights = Vec::<f64>::with_capacity(100);
    let mut weight_sum = 0_f64;

    let kernel_bandwidth = 5_f64;
    let bin_width = 5_usize;

    for _i in 0..2000 {
        let x: f64 = die.sample(&mut rng);
        let y: f64 = die.sample(&mut rng);
        data.push(x.round());
        data.push(y.round());
        let w: f64 = rng.gen();
        weights.push(w.round());
        weight_sum += weights.last().unwrap();
    }

    let gd = crate::kde::GridDimensions {
        width: 500,
        height: 500,
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
    let mut grid = crate::kde::KDEGrid::new(gd, bin_width, Some(kernel_bandwidth));
    for (chunk, w) in data.chunks(2).zip(weights.iter()) {
        grid.add_observation(chunk[0] as usize, chunk[1] as usize, *w);
    }

    let density = grid.evaluate_kde()?;
    //println!("rust kde matrix: {:+e}", density.data);

    let mut lookups = Vec::<f64>::with_capacity(100);
    for chunk in data.chunks(2) {
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
