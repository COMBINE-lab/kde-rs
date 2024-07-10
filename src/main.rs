mod kde;
mod kde_function;
use crate::kde::kde_computation;
use crate::kde_function::kde_computation_py;

use std::{
    io::{self, BufReader}
};

fn main() -> io::Result<()> {

    let data = vec![
        1.0, 2.0,
        2.0, 3.0,
        3.0, 4.0,
        4.0, 5.0 
    ];

    let weights = vec![
        1.0,
        2.0,
        3.0,
        4.0 
    ];

    //rust kde function
    kde_computation(&data, &weights)?;

    //python kde fucntion
    kde_computation_py(&data, &weights)?;

    Ok(())
}
