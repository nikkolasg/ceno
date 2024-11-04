pub mod arithmetic;
pub mod expression;
pub mod parallel;
pub mod plonky2_util;
use ff::{Field, PrimeField};
use ff_ext::ExtensionField;
use goldilocks::SmallField;
use itertools::{izip, Either, Itertools};
use multilinear_extensions::{
    mle::{DenseMultilinearExtension, FieldType},
    virtual_poly_v2::ArcMultilinearExtension,
};
use serde::{de::DeserializeOwned, Deserialize, Serialize};
pub mod merkle_tree;
use crate::{util::parallel::parallelize, Error};
pub use plonky2_util::log2_strict;

pub fn ext_to_usize<E: ExtensionField>(x: &E) -> usize {
    let bases = x.as_bases();
    bases[0].to_canonical_u64() as usize
}

pub fn base_to_usize<E: ExtensionField>(x: &E::BaseField) -> usize {
    x.to_canonical_u64() as usize
}

pub fn u32_to_field<E: ExtensionField>(x: u32) -> E::BaseField {
    E::BaseField::from(x as u64)
}

pub trait BitIndex {
    fn nth_bit(&self, nth: usize) -> bool;
}

impl BitIndex for usize {
    fn nth_bit(&self, nth: usize) -> bool {
        (self >> nth) & 1 == 1
    }
}

/// How many bytes are required to store n field elements?
pub fn num_of_bytes<F: PrimeField>(n: usize) -> usize {
    (F::NUM_BITS as usize).next_power_of_two() * n / 8
}

macro_rules! impl_index {
    (@ $name:ty, $field:tt, [$($range:ty => $output:ty),*$(,)?]) => {
        $(
            impl<E: ExtensionField> std::ops::Index<$range> for $name {
                type Output = $output;

                fn index(&self, index: $range) -> &$output {
                    match &self.$field {
                        FieldType::Ext(coeffs) => coeffs.index(index),
                        FieldType::Base(_) => panic!("Cannot index base field"),
                        _ => unreachable!()
                    }
                }
            }

            impl<E: ExtensionField> std::ops::IndexMut<$range> for $name {
                fn index_mut(&mut self, index: $range) -> &mut $output {
                    match &mut self.$field {
                        FieldType::Ext(coeffs) => coeffs.index_mut(index),
                        FieldType::Base(_) => panic!("Cannot index base field"),
                        _ => unreachable!()
                    }
                }
            }
        )*
    };
    (@ $name:ty, $field:tt) => {
        impl_index!(
            @ $name, $field,
            [
                usize => E,
                std::ops::Range<usize> => [E],
                std::ops::RangeFrom<usize> => [E],
                std::ops::RangeFull => [E],
                std::ops::RangeInclusive<usize> => [E],
                std::ops::RangeTo<usize> => [E],
                std::ops::RangeToInclusive<usize> => [E],
            ]
        );
    };
    ($name:ident, $field:tt) => {
        impl_index!(@ $name<E>, $field);
    };
}

pub(crate) use impl_index;

pub fn poly_index_ext<E: ExtensionField>(poly: &DenseMultilinearExtension<E>, index: usize) -> E {
    match &poly.evaluations {
        FieldType::Ext(coeffs) => coeffs[index],
        FieldType::Base(coeffs) => E::from(coeffs[index]),
        _ => unreachable!(),
    }
}

pub fn field_type_index_base<E: ExtensionField>(poly: &FieldType<E>, index: usize) -> E::BaseField {
    match &poly {
        FieldType::Ext(_) => panic!("Cannot get base field from extension field"),
        FieldType::Base(coeffs) => coeffs[index],
        _ => unreachable!(),
    }
}

pub fn field_type_index_ext<E: ExtensionField>(poly: &FieldType<E>, index: usize) -> E {
    match &poly {
        FieldType::Ext(coeffs) => coeffs[index],
        FieldType::Base(coeffs) => E::from(coeffs[index]),
        _ => unreachable!(),
    }
}

pub fn field_type_index_mul_base<E: ExtensionField>(
    poly: &mut FieldType<E>,
    index: usize,
    scalar: &E::BaseField,
) {
    match poly {
        FieldType::Ext(coeffs) => coeffs[index] *= scalar,
        FieldType::Base(coeffs) => coeffs[index] *= scalar,
        _ => unreachable!(),
    }
}

pub fn field_type_index_set_base<E: ExtensionField>(
    poly: &mut FieldType<E>,
    index: usize,
    scalar: &E::BaseField,
) {
    match poly {
        FieldType::Ext(coeffs) => coeffs[index] = E::from(*scalar),
        FieldType::Base(coeffs) => coeffs[index] = *scalar,
        _ => unreachable!(),
    }
}

pub fn field_type_index_set_ext<E: ExtensionField>(
    poly: &mut FieldType<E>,
    index: usize,
    scalar: &E,
) {
    match poly {
        FieldType::Ext(coeffs) => coeffs[index] = *scalar,
        FieldType::Base(_) => panic!("Cannot set base field from extension field"),
        _ => unreachable!(),
    }
}

pub fn poly_iter_ext<E: ExtensionField>(
    poly: &DenseMultilinearExtension<E>,
) -> impl Iterator<Item = E> + '_ {
    field_type_iter_ext(&poly.evaluations)
}

pub fn field_type_iter_ext<E: ExtensionField>(
    evaluations: &FieldType<E>,
) -> impl Iterator<Item = E> + '_ {
    match evaluations {
        FieldType::Ext(coeffs) => Either::Left(coeffs.iter().copied()),
        FieldType::Base(coeffs) => Either::Right(coeffs.iter().map(|x| (*x).into())),
        _ => unreachable!(),
    }
}

pub fn field_type_to_ext_vec<E: ExtensionField>(evaluations: &FieldType<E>) -> Vec<E> {
    match evaluations {
        FieldType::Ext(coeffs) => coeffs.to_vec(),
        FieldType::Base(coeffs) => coeffs.iter().map(|x| (*x).into()).collect(),
        _ => unreachable!(),
    }
}

pub fn field_type_as_ext<E: ExtensionField>(values: &FieldType<E>) -> &Vec<E> {
    match values {
        FieldType::Ext(coeffs) => coeffs,
        FieldType::Base(_) => panic!("Expected base field"),
        _ => unreachable!(),
    }
}

pub fn field_type_iter_base<E: ExtensionField>(
    values: &FieldType<E>,
) -> impl Iterator<Item = &E::BaseField> + '_ {
    match values {
        FieldType::Ext(coeffs) => Either::Left(coeffs.iter().flat_map(|x| x.as_bases())),
        FieldType::Base(coeffs) => Either::Right(coeffs.iter()),
        _ => unreachable!(),
    }
}

pub fn field_type_iter_range_base<'a, E: ExtensionField>(
    values: &'a FieldType<E>,
    range: impl IntoIterator<Item = usize> + 'a,
) -> impl Iterator<Item = &E::BaseField> + 'a {
    match values {
        FieldType::Ext(coeffs) => {
            Either::Left(range.into_iter().flat_map(|i| coeffs[i].as_bases()))
        }
        FieldType::Base(coeffs) => Either::Right(range.into_iter().map(|i| &coeffs[i])),
        _ => unreachable!(),
    }
}

pub fn multiply_poly<E: ExtensionField>(poly: &mut Vec<E>, scalar: &E) {
    for coeff in poly.iter_mut() {
        *coeff *= scalar;
    }
}

/// Resize to the new number of variables, which must be greater than or equal to
/// the current number of variables.
pub fn resize_num_vars<E: ExtensionField>(poly: &mut Vec<E>, num_vars: usize) {
    assert!(num_vars >= log2_strict(poly.len()));
    if num_vars == log2_strict(poly.len()) {
        return;
    }
    poly.resize(1 << num_vars, E::ZERO);
    (log2_strict(poly.len())..1 << num_vars).for_each(|i| poly[i] = poly[i & ((poly.len()) - 1)])
}

pub fn add_polynomial_with_coeff<E: ExtensionField>(
    lhs: &mut Vec<E>,
    rhs: &ArcMultilinearExtension<E>,
    coeff: &E,
) {
    match (lhs.is_empty(), rhs.num_vars() == 0) {
        (_, true) => {}
        (true, false) => {
            *lhs = field_type_to_ext_vec(rhs.evaluations());
            multiply_poly(lhs, coeff);
        }
        (false, false) => {
            if log2_strict(lhs.len()) < rhs.num_vars() {
                resize_num_vars(lhs, rhs.num_vars());
            }
            if rhs.num_vars() < log2_strict(lhs.len()) {
                parallelize(lhs, |(lhs, start)| {
                    for (index, lhs) in lhs.iter_mut().enumerate() {
                        *lhs += *coeff
                            * field_type_index_ext(
                                rhs.evaluations(),
                                (start + index) & ((1 << rhs.num_vars()) - 1),
                            );
                    }
                });
            } else {
                parallelize(lhs, |(lhs, start)| {
                    for (index, lhs) in lhs.iter_mut().enumerate() {
                        *lhs += *coeff * field_type_index_ext(rhs.evaluations(), start + index);
                    }
                });
            }
        }
    }
}

pub fn ext_try_into_base<E: ExtensionField>(x: &E) -> Result<E::BaseField, Error> {
    let bases = x.as_bases();
    if bases[1..].iter().any(|x| *x != E::BaseField::ZERO) {
        Err(Error::ExtensionFieldElementNotFit)
    } else {
        Ok(bases[0])
    }
}

#[cfg(any(test, feature = "benchmark"))]
pub mod test {
    use crate::util::{base_to_usize, u32_to_field};
    use ff::Field;
    type E = goldilocks::GoldilocksExt2;
    type F = goldilocks::Goldilocks;
    use rand::{
        rngs::{OsRng, StdRng},
        CryptoRng, RngCore, SeedableRng,
    };
    use std::{array, iter, ops::Range};

    pub fn std_rng() -> impl RngCore + CryptoRng {
        StdRng::from_seed(Default::default())
    }

    pub fn seeded_std_rng() -> impl RngCore + CryptoRng {
        StdRng::seed_from_u64(OsRng.next_u64())
    }

    pub fn rand_idx(range: Range<usize>, mut rng: impl RngCore) -> usize {
        range.start + (rng.next_u64() as usize % (range.end - range.start))
    }

    pub fn rand_array<F: Field, const N: usize>(mut rng: impl RngCore) -> [F; N] {
        array::from_fn(|_| F::random(&mut rng))
    }

    pub fn rand_vec<F: Field>(n: usize, mut rng: impl RngCore) -> Vec<F> {
        iter::repeat_with(|| F::random(&mut rng)).take(n).collect()
    }

    #[test]
    pub fn test_field_transform() {
        assert_eq!(F::from(2) * F::from(3), F::from(6));
        assert_eq!(base_to_usize::<E>(&u32_to_field::<E>(1u32)), 1);
        assert_eq!(base_to_usize::<E>(&u32_to_field::<E>(10u32)), 10);
    }
}
