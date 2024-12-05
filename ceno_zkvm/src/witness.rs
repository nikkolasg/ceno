use core::assert_eq;
use ff::Field;
use std::{
    array,
    cell::RefCell,
    collections::HashMap,
    mem::{self, MaybeUninit},
    ops::Index,
    slice::{Chunks, ChunksMut},
    sync::Arc,
};

use multilinear_extensions::{
    mle::{DenseMultilinearExtension, IntoMLEs},
    util::create_uninit_vec,
};
use rayon::{
    iter::{IndexedParallelIterator, IntoParallelRefMutIterator, ParallelIterator},
    slice::ParallelSliceMut,
};
use thread_local::ThreadLocal;

use crate::{
    structs::ROMType,
    tables::{AndTable, LtuTable, OpsTable, OrTable, PowTable, XorTable},
    utils::next_pow2_instance_padding,
};

#[macro_export]
macro_rules! set_val {
    ($ins:ident, $field:expr, $val:expr) => {
        $ins[$field.id as usize] = MaybeUninit::new($val.into());
    };
}

#[macro_export]
macro_rules! set_fixed_val {
    ($ins:ident, $field:expr, $val:expr) => {
        $ins[$field.0] = MaybeUninit::new($val);
    };
}

#[derive(Clone)]
pub struct RowMajorMatrix<T: PartialEq + Eq + Sized + Sync + Clone + Send + Copy> {
    // represent 2D in 1D linear memory and avoid double indirection by Vec<Vec<T>> to improve performance
    values: Vec<MaybeUninit<T>>,
    num_padding_rows: usize,
    num_col: usize,
}

impl<T: PartialEq + Eq + Sized + Sync + Clone + Send + Copy> RowMajorMatrix<T> {
    pub fn new(num_rows: usize, num_col: usize) -> Self {
        let num_total_rows = next_pow2_instance_padding(num_rows);
        let num_padding_rows = num_total_rows - num_rows;
        RowMajorMatrix {
            values: create_uninit_vec(num_total_rows * num_col),
            num_padding_rows,
            num_col,
        }
    }

    pub fn num_col(&self) -> usize {
        self.num_col
    }

    pub fn num_instances(&self) -> usize {
        // tracing::info!("num_instances... {} {} {}", self.values.len(), self.num_col, self.num_padding_rows);
        (self.values.len() / self.num_col)
            .checked_sub(self.num_padding_rows)
            .expect("overflow")
    }

    pub fn num_padding_instances(&self) -> usize {
        self.num_padding_rows
    }

    pub fn iter_rows(&self) -> Chunks<MaybeUninit<T>> {
        self.values.chunks(self.num_col)
    }

    pub fn iter_mut(&mut self) -> ChunksMut<MaybeUninit<T>> {
        self.values.chunks_mut(self.num_col)
    }

    pub fn par_iter_mut(&mut self) -> rayon::slice::ChunksMut<MaybeUninit<T>> {
        self.values.par_chunks_mut(self.num_col)
    }

    pub fn par_batch_iter_mut(
        &mut self,
        num_rows: usize,
    ) -> rayon::slice::ChunksMut<MaybeUninit<T>> {
        self.values.par_chunks_mut(num_rows * self.num_col)
    }

    pub fn par_batch_iter_padding_mut(
        &mut self,
        num_rows: usize,
    ) -> rayon::slice::ChunksMut<'_, MaybeUninit<T>> {
        let valid_instance = self.num_instances();
        self.values[valid_instance * self.num_col..]
            .as_mut()
            .par_chunks_mut(num_rows * self.num_col)
    }

    pub fn de_interleaving(mut self) -> Vec<Vec<T>> {
        tracing::debug!("de_interleaving..");
        (0..self.num_col)
            .map(|i| {
                self.values
                    .par_iter_mut()
                    .skip(i)
                    .step_by(self.num_col)
                    .map(|v| unsafe { mem::replace(v, mem::MaybeUninit::uninit()).assume_init() })
                    .collect::<Vec<T>>()
            })
            .collect()
    }

    // TODO: should we consume or clone `self`?
    pub fn chunk_by_num(&self, chunk_rows: usize) -> Vec<Self> {
        let padded_row_num = self.values.len() / self.num_col;
        if padded_row_num <= chunk_rows {
            return vec![self.clone()];
        }
        // padded_row_num and instance_num_per_chunk should both be pow of 2.
        assert_eq!(padded_row_num % chunk_rows, 0);
        let chunk_num = (self.num_instances() + chunk_rows - 1) / chunk_rows;
        let mut result = Vec::new();
        for i in 0..chunk_num {
            let mut values: Vec<_> = self.values
                [(i * chunk_rows * self.num_col)..((i + 1) * chunk_rows * self.num_col)]
                .to_vec();
            let mut num_padding_rows = 0;

            // Only last chunk contains padding rows.
            if i == chunk_num - 1 && self.num_instances() % chunk_rows != 0 {
                let num_rows = self.num_instances() % chunk_rows;
                let num_total_rows = next_pow2_instance_padding(num_rows);
                num_padding_rows = num_total_rows - num_rows;
                values.truncate(num_total_rows * self.num_col);
            };

            tracing::info!(
                "chunk_by_num {i}th chunk: num_rows {chunk_rows}, num_padding_rows {num_padding_rows}"
            );
            result.push(Self {
                num_col: self.num_col,
                num_padding_rows,
                values,
            });
        }
        assert_eq!(
            self.num_instances(),
            result
                .iter()
                .enumerate()
                .map(|(idx, c)| {
                    tracing::info!("{idx}chunk num_instances: {}", c.num_instances());
                    c.num_instances()
                })
                .sum::<usize>()
        );
        result
    }
}

impl<F: Field> RowMajorMatrix<F> {
    pub fn into_mles<E: ff_ext::ExtensionField<BaseField = F>>(
        self,
    ) -> Vec<DenseMultilinearExtension<E>> {
        tracing::info!("before de_interleaving");
        self.de_interleaving().into_mles()
    }
}

impl<F: Field> Index<usize> for RowMajorMatrix<F> {
    type Output = [MaybeUninit<F>];

    fn index(&self, idx: usize) -> &Self::Output {
        &self.values[self.num_col * idx..][..self.num_col]
    }
}

/// A lock-free thread safe struct to count logup multiplicity for each ROM type
/// Lock-free by thread-local such that each thread will only have its local copy
/// struct is cloneable, for internallly it use Arc so the clone will be low cost
#[derive(Clone, Default, Debug)]
#[allow(clippy::type_complexity)]
pub struct LkMultiplicity {
    multiplicity: Arc<ThreadLocal<RefCell<[HashMap<u64, usize>; mem::variant_count::<ROMType>()]>>>,
}

impl LkMultiplicity {
    /// assert within range
    #[inline(always)]
    pub fn assert_ux<const C: usize>(&mut self, v: u64) {
        match C {
            16 => self.increment(ROMType::U16, v),
            14 => self.increment(ROMType::U14, v),
            8 => self.increment(ROMType::U8, v),
            5 => self.increment(ROMType::U5, v),
            _ => panic!("Unsupported bit range"),
        }
    }

    /// Track a lookup into a logic table (AndTable, etc).
    pub fn logic_u8<OP: OpsTable>(&mut self, a: u64, b: u64) {
        self.increment(OP::ROM_TYPE, OP::pack(a, b));
    }

    /// lookup a AND b
    pub fn lookup_and_byte(&mut self, a: u64, b: u64) {
        self.logic_u8::<AndTable>(a, b)
    }

    /// lookup a OR b
    pub fn lookup_or_byte(&mut self, a: u64, b: u64) {
        self.logic_u8::<OrTable>(a, b)
    }

    /// lookup a XOR b
    pub fn lookup_xor_byte(&mut self, a: u64, b: u64) {
        self.logic_u8::<XorTable>(a, b)
    }

    /// lookup a < b as unsigned byte
    pub fn lookup_ltu_byte(&mut self, a: u64, b: u64) {
        self.logic_u8::<LtuTable>(a, b)
    }

    pub fn lookup_pow2(&mut self, v: u64) {
        self.logic_u8::<PowTable>(2, v)
    }

    /// Fetch instruction at pc
    pub fn fetch(&mut self, pc: u32) {
        self.increment(ROMType::Instruction, pc as u64);
    }

    /// merge result from multiple thread local to single result
    pub fn into_finalize_result(self) -> [HashMap<u64, usize>; mem::variant_count::<ROMType>()] {
        Arc::try_unwrap(self.multiplicity)
            .unwrap()
            .into_iter()
            .fold(array::from_fn(|_| HashMap::new()), |mut x, y| {
                x.iter_mut().zip(y.borrow().iter()).for_each(|(m1, m2)| {
                    for (key, value) in m2 {
                        *m1.entry(*key).or_insert(0) += value;
                    }
                });
                x
            })
    }

    fn increment(&mut self, rom_type: ROMType, key: u64) {
        let multiplicity = self
            .multiplicity
            .get_or(|| RefCell::new(array::from_fn(|_| HashMap::new())));
        (*multiplicity.borrow_mut()[rom_type as usize]
            .entry(key)
            .or_default()) += 1;
    }
}

#[cfg(test)]
mod tests {
    use std::thread;

    use crate::{structs::ROMType, witness::LkMultiplicity};

    #[test]
    fn test_lk_multiplicity_threads() {
        // TODO figure out a way to verify thread_local hit/miss in unittest env
        let lkm = LkMultiplicity::default();
        let thread_count = 20;
        // each thread calling assert_byte once
        for _ in 0..thread_count {
            let mut lkm = lkm.clone();
            thread::spawn(move || lkm.assert_ux::<8>(8u64))
                .join()
                .unwrap();
        }
        let res = lkm.into_finalize_result();
        // check multiplicity counts of assert_byte
        assert_eq!(res[ROMType::U8 as usize][&8], thread_count);
    }
}
