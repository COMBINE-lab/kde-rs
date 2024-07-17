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
    max_x: Option<usize>,
    max_y: Option<usize>,
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
