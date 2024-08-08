use ff_ext::ExtensionField;
use multilinear_extensions::virtual_poly_v2::{ArcMultilinearExtension, VirtualPolynomialV2};
use sumcheck::structs::IOPProverMessage;

use crate::uint::UInt;

pub struct TowerProver;

#[derive(Clone)]
pub struct TowerProofs<E: ExtensionField> {
    pub proofs: Vec<Vec<IOPProverMessage<E>>>,
    // specs -> layers -> evals
    pub prod_specs_eval: Vec<Vec<Vec<E>>>,
    // specs -> layers -> evals
    pub logup_specs_eval: Vec<Vec<Vec<E>>>,
}

pub struct TowerProverSpec<'a, E: ExtensionField> {
    pub witness: Vec<Vec<ArcMultilinearExtension<'a, E>>>,
}

const VALUE_BIT_WIDTH: usize = 16;
pub type WitnessId = u16;
pub type ChallengeId = u16;
pub type UInt64<E> = UInt<64, VALUE_BIT_WIDTH, E>;
pub type PCUInt<E> = UInt64<E>;
pub type TSUInt<E> = UInt<48, 48, E>;

pub enum ROMType {
    U5, // 2^5=32
}

#[derive(Clone, Debug, Copy)]
pub enum RAMType {
    GlobalState,
    Register,
}

pub struct VirtualPolynomials<'a, E: ExtensionField> {
    pub num_threads: usize,
    pub polys: Vec<VirtualPolynomialV2<'a, E>>,
}

/// A point is a vector of num_var length
pub type Point<F> = Vec<F>;

/// A point and the evaluation of this point.
#[derive(Clone, Debug, PartialEq)]
pub struct PointAndEval<F> {
    pub point: Point<F>,
    pub eval: F,
}

impl<E: ExtensionField> Default for PointAndEval<E> {
    fn default() -> Self {
        Self {
            point: vec![],
            eval: E::ZERO,
        }
    }
}

impl<F: Clone> PointAndEval<F> {
    /// Construct a new pair of point and eval.
    /// Caller gives up ownership
    pub fn new(point: Point<F>, eval: F) -> Self {
        Self { point, eval }
    }

    /// Construct a new pair of point and eval.
    /// Performs deep copy.
    pub fn new_from_ref(point: &Point<F>, eval: &F) -> Self {
        Self {
            point: (*point).clone(),
            eval: eval.clone(),
        }
    }
}
