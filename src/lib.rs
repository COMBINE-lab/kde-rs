//! The `kde-rs` crate provides an implementation of a 2D kernel density
//! estimator for a restricted set of situations. It focuses on being
//! easy to use, and on being efficient in the scenario for which it
//! has been designed.
//!
//! # Example
//! ```
//! use kders::{kde, kde::GridDimensions};
//! use rand::prelude::*;
//! use rand::{distributions::Uniform, Rng};
//!
//! // create a grid on the range [0, 100]x[0, 100] with bins
//! // of width 5 and a kernel bandwidth of 2.
//! let mut rng = rand::thread_rng();
//! let die = Uniform::from(10.0..100.);
//! let mut g = kde::KDEGrid::new(GridDimensions{ width: 100, height: 100 }, 5, Some(2.0));
//!
//! // add 100 random observations
//! for _i in 0..100 {
//!     let x: f64 = die.sample(&mut rng);
//!     let y: f64 = die.sample(&mut rng);
//!     g.add_observation(x as usize, y as usize, rng.gen())
//! }
//! // obtain the density grid on which we'll evaluate our queries
//!
//! let density = g.get_kde().unwrap();
//! // lookup the density at some points
//! let a = density[(10_usize, 10_usize)];
//! let b = density[(10_usize, 25_usize)];
//! let c = density[(68_usize, 5_usize)];
//! let d = density[(79_usize, 10_usize)];
//! ```

pub mod kde;
