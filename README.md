# kde-rs

An implementation of a *simple*, *restricted* 2D kernel density estimator in Rust.
The `kde-rs` library is developed for a specific use-case, and while we may consider
generalizing it in the future, that is not a current focus.

Currently, the main restrictions are:

 - only 2D estimation is supported
 - only integer sample points --- (x, y) pairs are integers --- are supported (but they can be weighted)
 - the symmetric Gaussian kernel is the only one currently implemented

However, subject to these constraints, `kde-rs` strives to be easy to use, and performant.
