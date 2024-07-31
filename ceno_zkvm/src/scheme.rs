use ff_ext::ExtensionField;
use sumcheck::structs::IOPProverMessage;

use crate::structs::TowerProofs;

pub mod constants;
pub mod prover;
mod utils;
pub mod verifier;

#[derive(Clone)]
pub struct ZKVMProof<E: ExtensionField> {
    // TODO support >1 opcodes
    pub num_instances: usize,

    // product constraints
    pub record_r_out_evals: Vec<E>,
    pub record_w_out_evals: Vec<E>,
    pub tower_proof: TowerProofs<E>,

    // main constraint and select sumcheck proof
    pub main_sel_sumcheck_proofs: Vec<IOPProverMessage<E>>,
    pub r_records_in_evals: Vec<E>,
    pub w_records_in_evals: Vec<E>,

    pub wits_in_evals: Vec<E>,
}
