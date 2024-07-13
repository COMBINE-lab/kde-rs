mod kde;
mod kde_function;
use crate::kde::kde_computation;
use crate::kde_function::kde_computation_py;

use itertools::*;
use ordered_float::OrderedFloat;
use rand::distributions::{Distribution, Uniform};
use rand::prelude::*;

use std::io::{self, BufReader};

fn main() -> anyhow::Result<()> {
    /*
    let data = vec![1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0, 5.0];
    let weights = vec![1.0, 2.0, 3.0, 4.0];

    //rust kde function
    kde_computation(&data, &weights)?;
    */
    let mut rng = rand::thread_rng();
    let die = Uniform::from(0.0..100.0);
    let mut data = Vec::<f64>::with_capacity(200);
    let mut weights = Vec::<f64>::with_capacity(100);
    for i in 0..200 {
        let x: f64 = die.sample(&mut rng);
        let y: f64 = die.sample(&mut rng);
        data.push(x);
        data.push(y);
        let w: f64 = rng.gen();
        weights.push(w);
    }

    //python kde fucntion
    let py_res = kde_computation_py(&data, &weights)?;

    println!("grid \n\n");
    let mut grid = crate::kde::KDEGrid::from_data_with_binwidth(
        &data.iter().map(|x| *x as u64).collect::<Vec<u64>>(),
        &weights,
        1,
    );

    let density = grid.evaluate_kde()?;

    let mut lookups = Vec::<f64>::with_capacity(100);
    for chunk in data.chunks(2) {
        lookups.push(density[(chunk[0] as usize, chunk[1] as usize)]);
    }
    println!("py: {:?}", py_res);
    println!("rust: {:?}", lookups);
    println!("total denisty of kde is : {}", density.sum());

    use ordered_float::OrderedFloat;
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
