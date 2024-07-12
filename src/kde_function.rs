use ndarray::{Array1, Array2, ArrayView2};
use numpy::IntoPyArray;
use numpy::{PyArray1, PyArray2};
use pyo3::prelude::*;
use pyo3::prepare_freethreaded_python;
use pyo3::types::PyDict;

use std::io::{self, BufReader};

pub fn kde_computation_py(
    data: &Vec<f64>,
    weight: &Vec<f64>,
    //store: &mut InMemoryAlignmentStore,
) -> io::Result<()> {
    // Initialize the Python interpreter
    prepare_freethreaded_python();

    Python::with_gil(|py| {
        // Import the Python module and function
        let module = PyModule::from_code(
            py,
            include_str!("2D_kde_function.py"),
            "2D_kde_function",
            "2D_kde_function.py",
        )?;
        let function = module.getattr("calculate_kde")?;
        println!("data length is: {:?}", data.len());
        println!("weight length is: {:?}", weight.len());

        // Example 2D data
        let data = Array2::from_shape_vec(((data.len() / 2), 2), data.to_vec()).unwrap();

        // Convert data to PyArray2
        let data_py: &PyArray2<f64> = data.view().to_owned().into_pyarray(py);

        // Dummy weights for demonstration purposes
        let weights = Array2::from_shape_vec((weight.len(), 1), weight.to_vec()).unwrap();

        // Convert weights to PyArray2
        let weights_py: &PyArray2<f64> = weights.view().to_owned().into_pyarray(py);

        // Call the Python function
        println!("it is before python function");
        let result: Py<PyArray1<f64>> = function.call1((data_py, weights_py))?.extract()?;
        //println!("result length: {:?}", result.as_ref().len());

        Ok(())
    })
}

