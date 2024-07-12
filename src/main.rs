mod kde;
mod kde_function;
use crate::kde::kde_computation;
use crate::kde_function::kde_computation_py;

use std::io::{self, BufReader};

fn main() -> anyhow::Result<()> {
    let data = vec![1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0, 5.0];

    let weights = vec![1.0, 2.0, 3.0, 4.0];

    //rust kde function
    kde_computation(&data, &weights)?;

    //python kde fucntion
    kde_computation_py(&data, &weights)?;

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

    Ok(())
}
