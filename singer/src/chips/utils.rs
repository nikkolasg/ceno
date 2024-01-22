use frontend::structs::{CellId, CircuitBuilder};
use gkr::structs::Circuit;
use goldilocks::SmallField;

pub(in crate::chips) fn hash_2<F: SmallField>(
    circuit_builder: &mut CircuitBuilder<F>,
    out: CellId,
    in_0: CellId,
    in_1: CellId,
) {
    todo!()
}

pub(in crate::chips) fn hash_n<F: SmallField>(
    circuit_builder: &mut CircuitBuilder<F>,
    out: CellId,
    in_array: &[CellId],
) {
    todo!()
}

/// With denominators of size `den_size`, construct a vector of [denominators,
/// numerators], where numerators are `den_size` 1s padded with zeros.
pub(in crate::chips) fn den_to_frac_circuit<F: SmallField>(den_size: usize) -> Circuit<F> {
    let padded_size = den_size.next_power_of_two();
    let mut circuit_builder = CircuitBuilder::<F>::new();
    let _ = circuit_builder.create_wire_in(den_size);
    let _ = circuit_builder.create_constant_in(padded_size - den_size, 1);
    let _ = circuit_builder.create_constant_in(den_size, 1);
    let _ = circuit_builder.create_constant_in(padded_size - den_size, 0);
    circuit_builder.configure();
    Circuit::new(&circuit_builder)
}

/// Pad with constants.
pub(in crate::chips) fn pad_with_const_circuit<F: SmallField>(
    size: usize,
    constant: i64,
) -> Circuit<F> {
    let mut circuit_builder = CircuitBuilder::<F>::new();
    let padded_size = size.next_power_of_two();
    let _ = circuit_builder.create_wire_in(size);
    let _ = circuit_builder.create_constant_in(padded_size - size, constant);
    circuit_builder.configure();
    Circuit::new(&circuit_builder)
}
