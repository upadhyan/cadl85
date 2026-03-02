pub mod algorithms;
pub mod bitsets;
pub mod caching;
pub(crate) mod cover;
pub mod globals;
pub mod parser;
pub mod reader;
pub mod tree;

#[cfg(feature = "python")]
mod python;

#[cfg(feature = "python")]
use pyo3::prelude::*;

#[cfg(feature = "python")]
#[pymodule]
fn _cadl85(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<python::PyCadl85>()?;
    Ok(())
}
