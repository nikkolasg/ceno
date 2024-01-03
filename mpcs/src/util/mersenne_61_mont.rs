use halo2_curves::ff::{PrimeField};
use serde::{Serialize, Deserialize};
use ff::{PrimeFieldBits,BatchInvert};
use std::ops::{Shr, BitAnd};
use rand::RngCore;
use std::fmt::{Display,Formatter,Debug};
use core::{iter::{Product,Sum}, ops::{Add,Mul,AddAssign, Sub, SubAssign, Neg, Div, DivAssign, MulAssign}};
use subtle::{ConstantTimeEq,Choice,ConditionallySelectable,CtOption};
use rand::SeedableRng;
use crate::util::{BigUint, {arithmetic::{modulus,Field}}};

#[derive(PrimeField,Serialize,Deserialize,Hash)]
#[PrimeFieldModulus = "2305843009213693951"]
#[PrimeFieldGenerator = "7"]
#[PrimeFieldReprEndianness = "little"]
pub struct Mersenne61Mont([u64;1]);
