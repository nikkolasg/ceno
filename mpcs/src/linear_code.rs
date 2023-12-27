use ff::FromUniformBytes;
use goldilocks::SmallField;
use serde::{Deserialize, Serialize};

use crate::fft::EvaluationDomain;

/// The trait for linear code.
pub trait LinearCode<F> {
    fn encode(&self, message: &[F]) -> Vec<F>;
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
/// Reed-Solomon code.
pub struct ReedSolomonCode<F> {
    pub(crate) message_size_bits: usize,
    pub(crate) generator: F,
    pub(crate) rate_bits: usize,
}

impl<F: SmallField + FromUniformBytes<64>> ReedSolomonCode<F> {
    pub fn new(message_size_bits: usize, rate_bits: usize) -> ReedSolomonCode<F> {
        assert!(message_size_bits + rate_bits <= F::S as usize);
        ReedSolomonCode {
            message_size_bits,
            generator: F::ROOT_OF_UNITY
                .pow(&[1 << (F::S as usize - (message_size_bits + rate_bits))]),
            rate_bits,
        }
    }

    pub fn codeword_length(&self) -> usize {
        self.message_size_bits << self.rate_bits
    }
}

impl<F: SmallField + FromUniformBytes<64>> LinearCode<F> for ReedSolomonCode<F> {
    fn encode(&self, message: &[F]) -> Vec<F> {
        let m = message.len();
        assert!(m == 1 << self.message_size_bits);
        let mut codeword = vec![F::ZERO; self.codeword_length()];
        // Copy the message, repeated 1 << rate_bits times, to the codeword.
        for (i, &x) in message.iter().enumerate() {
            for j in 0..(1 << self.rate_bits) {
                codeword[i + j * m] = x;
            }
        }

        // The message domain is H, generated by m-th root of unity.
        // The codeword domain is L, generated by n-th root of unity, where m = n << rate_bits,
        // then multiplied by F::generator on every element to make it disjoint from H.
        // So L = g * (H || omega H || omega^2 H || ... || omega^(rate-1) H).
        //      = gH || g omega H || ... || g omega^(rate-1) H
        let domain = EvaluationDomain::<F>::new(m).unwrap();
        let mut offset = F::MULTIPLICATIVE_GENERATOR;
        for i in 0..1 << self.rate_bits {
            domain.coset_fft(&mut codeword.as_mut_slice()[i * m..(i + 1) * m], offset);
            offset *= self.generator;
        }
        codeword
    }
}
