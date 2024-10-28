use std::fmt;
use std::ops::{Add, Sub, Mul, MulAssign, Div};
use pretty::{BoxAllocator, Pretty, DocAllocator, DocBuilder};
use crate::context::{Ctx, Set};
use crate::traits::{Specializable, SpecializeError, Normalizable};
use crate::bin::{Bin, Lin};
use thiserror::Error;

use arbitrary::{Arbitrary, Unstructured};

/// Represents a type-level, signed dyadic monoterm,
/// eg: 3* a * b^2 * c^3 * 2^(d+e+f+2)
///     -2 * x * 2^y
#[derive(Debug, Clone, PartialEq, Eq, Ord, PartialOrd, Hash)]
pub struct DyadicMono<Id> {
    mult: i32,          // 3
    terms: Ctx<Id, u8>, //a * b^2 * c^3
    bin: Bin<Id>        // 2^(d+e+f+2)
}

impl<Id: Ord> DyadicMono<Id> {
    pub fn lit(mult: i32) -> Self where Id: Ord {
        let (p, r) = Bin::log2(mult);
        DyadicMono { mult: r, terms: Ctx::new(), bin: p }
    }
    pub fn var(v: Id) -> Self where Id: Ord {
        DyadicMono { mult:1, terms: Ctx::from([(v, 1)]), bin: Bin::default() }
    }
    pub fn term(v: Id, exp: u8) -> Self {
        DyadicMono { mult: 1, terms: Ctx::from([(v, exp)]), bin: Bin::default() }
    }
    pub fn bin(b: Bin<Id>) -> Self where Id: Ord {
        DyadicMono { mult: 1, terms: Ctx::new(), bin: b }
    }
    pub fn neg(self) -> Self {
        DyadicMono { mult: -self.mult, terms: self.terms, bin: self.bin }
    }
    /// Doubling a term
    pub fn double(self) -> Self where Id: Clone {
        DyadicMono { mult: self.mult, terms: self.terms, bin: self.bin.double() }
    }
    /// Halving a term could fail (ex: 3*a^2*2^c)
    pub fn half(self) -> Option<Self> {
        if self.mult % 2 == 0 { // broken invariant, normalize
            let mut d = DyadicMono { mult: self.mult / 2, terms: self.terms, bin: self.bin };
            d.normalize();
            Some(d)
        } else if let Some(b) = self.bin.half() {
            Some(DyadicMono { mult: self.mult, terms: self.terms, bin: b })
        } else {
            None
        }
    }
    /// Multiplication by a signed literal
    pub fn mul_lit(&mut self, other: i32) {
        let (p, r) = Bin::log2(other);
        self.mult *= r;
        self.bin *= p;
    }
    /// Multiplication by a bin [normalizes]
    pub fn mul_bin(self, other: Bin<Id>) -> Self where Id: Clone {
        let mut res = self.clone();
        res.bin *= other;
        res
    }
    /// Division by a bin (with remainder)
    pub fn div_bin(self, other: Bin<Id>) -> (Self, Bin<Id>) where Id: Clone {
        let mut res = self.clone();
        let (q, r) = res.bin.div(other);
        res.bin = q;
        (res, r)
    }
}

impl<T: Ord> Default for DyadicMono<T> {
    fn default() -> Self {
        DyadicMono::bin(Bin::default())
    }
}

/// Multiplication for monoterms [normalizes]
impl<T: Ord> MulAssign for DyadicMono<T> {
    fn mul_assign(&mut self, other: Self) {
        for (k, v) in other.terms.into_iter() {
            self.terms.insert_with(k, v, &|x, y| x + y);
        }
        // No normalization needed
        // if a % 2 == 1, b % 2 == 1, then a * b % 2 == 1
        self.mult *= other.mult;
        self.bin *= other.bin;
    }
}

impl<T: Ord + Clone> Mul for DyadicMono<T> {
    type Output = Self;
    fn mul(self, other: Self) -> Self {
        let mut res = self.clone();
        res *= other;
        res
    }
}

impl<T: Ord> From<Bin<T>> for DyadicMono<T> {
    fn from(b: Bin<T>) -> Self {
        DyadicMono::bin(b)
    }
}

impl<'a> From<&'a str> for DyadicMono<&'a str> {
    fn from(s: &'a str) -> Self {
        DyadicMono::var(s)
    }
}

impl<T: Ord> From<i32> for DyadicMono<T> {
    fn from(l: i32) -> Self {
        DyadicMono::lit(l)
    }
}

impl<T: fmt::Display + Ord + Clone> Specializable<T> for DyadicMono<T> {
    // ex: 12*a^2*b^3*2^(c+1) -> a = 2 -> 48*b^3*2^(c+3)
    fn specialize(&mut self, id: T, val: u8) -> Result<(), SpecializeError<T>> {
        if let Some(v) = self.terms.remove(&id) {
            self.mult *= val.pow(v as u32) as i32;
            self.bin.specialize(id, val)?;
            self.normalize();
            Ok(())
        } else {
            Err(SpecializeError::VarNotFound(id))
        }
    }

    fn free_vars(&self) -> Set<&T> {
        self.terms.keys().union(self.bin.free_vars())
    }
}

impl<T: Ord> Normalizable for DyadicMono<T> {
    // ex: 12 * a * b^2 * 2^(c+1) -> 3 * a * b^2 * 2^(c + 3)
    fn normalize(&mut self) {
        let (p, r) = Bin::log2(self.mult);
        self.bin *= p;
        self.mult = r;
    }
}

////////////////////////////////////////////////////////////////////////////////////////
/* Pretty Formatting & Display */
////////////////////////////////////////////////////////////////////////////////////////
impl<'a, D, A, T> Pretty<'a, D, A> for DyadicMono<T>
where
    D: DocAllocator<'a, A>,
    T: Pretty<'a, D, A> + Clone,
    D::Doc: Clone,
    A: 'a + Clone,
{
    fn pretty(self, allocator: &'a D) -> DocBuilder<'a, D, A> {
        allocator.text(format!("{}*", self.mult))
            .append(allocator.intersperse(self.terms.into_iter()
                    .filter(|(_, v)| *v != 0)
                    .map(|(k, v)|
                        if v == 1 {
                            k.pretty(allocator)
                        } else {
                            k.pretty(allocator).append(allocator.text(format!("^{}", v)))
                        })
                    , "*"))
            .append(self.bin.pretty(allocator))
    }
}

/// Display instance calls the pretty printer
impl<'a, T> fmt::Display for DyadicMono<T>
where
    T: Pretty<'a, BoxAllocator, ()> + Clone + Ord,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        <DyadicMono<T> as Pretty<'_, BoxAllocator, ()>>::pretty(self.clone(), &BoxAllocator)
            .1
            .render_fmt(100, f)
    }
}

/// Arbitrary instance for DyadicMono
impl<'a, T: Ord + Arbitrary<'a>> Arbitrary<'a> for DyadicMono<T> {
    fn arbitrary(u: &mut Unstructured<'a>) -> arbitrary::Result<Self> {
        Ok(DyadicMono {
            mult : u.int_in_range(0..=9)?,
            terms : Ctx::arbitrary(u)?,
            bin : Bin::arbitrary(u)?
        })
    }
}

////////////////////////////////////////////////////////////////////////////////////////
/* Unit Tests for DyadicMono */
////////////////////////////////////////////////////////////////////////////////////////
#[test]
fn test_dyadicmono_mul() {
    // 2 * X * 3 * 2^Y = 6 * X * 2^Y
    let a = DyadicMono::lit(2) * DyadicMono::var("X");
    let b = DyadicMono::lit(3) * DyadicMono::bin(Bin::var("Y"));
    assert_eq!(
        a * b,
        DyadicMono { mult: 3, terms: Ctx::from([("X", 1)]), bin: Bin::var("Y") * Bin::lit(1) }
    );
}

#[test]
fn test_specialize() {
    let l = DyadicMono::from("x") * DyadicMono::from("y") * DyadicMono::from(2);

    let mut l1 = l.clone();
    l1.specialize("x", 2).unwrap();
    // 2*x*y -> x = 2 -> 4*y
    assert_eq!(l1, DyadicMono::var("y") * DyadicMono::lit(4));

    let mut l2 = l.clone();
    l2.specialize("y", 3).unwrap();
    // 2*x*y -> y = 3 -> 6*x
    assert_eq!(l2, DyadicMono::var("x") * DyadicMono::lit(6));

    let mut l3 = l.clone();
    assert_eq!(l3.specialize("z", 3),
        Err(SpecializeError::VarNotFound("z")));
}

/// Finally a dyadic number
/// (DyadicMono + ... + DyadicMono) / Bin
/// Dyadic(Set(), Bin::lit(0)) = Dyadic(Set(DyadicMono::lit(0)), Bin::lit(0)) = 0
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Dyadic<T> { numer: Set<DyadicMono<T>>, denom: Bin<T> }

impl<T: Ord> Default for Dyadic<T> {
    fn default() -> Self {
        Dyadic { numer: Set::new(), denom: Bin::default() }
    }
}

impl<T: Ord> Dyadic<T> {
    pub fn lit(i: i32) -> Self {
        Dyadic {
            numer: Set::singleton(DyadicMono::lit(i)),
            denom: Bin::default()
        }
    }
    pub fn var(v: T) -> Self {
        Dyadic {
            numer: Set::singleton(DyadicMono::var(v)),
            denom: Bin::default()
        }
    }
    // p / q
    pub fn frac(p: DyadicMono<T>, q: Bin<T>) -> Self {
        Dyadic {
            numer: Set::singleton(p),
            denom: q
        }
    }
    // -p
    pub fn neg(self) -> Self {
        Dyadic {
            numer: self.numer.into_iter().map(|v| v.neg()).collect(),
            denom: self.denom
        }
    }

    // 2*p
    pub fn double(self) -> Self where T: Clone {
        if let Some(p) = self.denom.clone().half() {
            // First try halving the denominator
            Dyadic { numer: self.numer, denom: p }
        } else {
            // Otherwise multiply all the numerator terms by 2
            Dyadic {
                numer: self.numer.into_iter().map(|v| v.double()).collect(),
                denom: self.denom
            }
        }
    }

    // p / 2
    pub fn half(self) -> Self where T: Clone {
        Dyadic { numer: self.numer, denom: self.denom.double() }
    }
    // p / b
    pub fn div_bin(&mut self, other: Bin<T>) {
        self.denom *= other;
    }
}

// TODO: Remove Debug and assert
impl<T: Ord + Clone + fmt::Debug> Add for Dyadic<T> {
    type Output = Dyadic<T>;
    fn add(self, other: Self) -> Self::Output {
        // Compute denominator as the LCM of the two denominators
        let denom = self.denom.lcm(other.denom.clone());

        // Compute the left multiplicative factor (remainder is 0 by LCM property)
        let (lm, lr) = denom.clone().div(self.denom);
        assert_eq!(lr, Bin::default());
        // Compute the right multiplicative factor
        let (rm, rr) = denom.clone().div(other.denom);
        assert_eq!(rr, Bin::default());

        // Multiply each term by the multiplicative factor
        let mut numer: Set<DyadicMono<T>> = self.numer.into_iter().map(|v| v.mul_bin(lm.clone())).collect();
        let r = other.numer.into_iter().map(|v| v.mul_bin(rm.clone()));

        // If a + b + b = a + 2b
        numer.append_with(r, &|v| v.double());
        // Return l / denom
        Dyadic { numer, denom }
    }
}

impl<T: Ord + Clone + fmt::Debug> Sub for Dyadic<T> {
    type Output = Dyadic<T>;
    fn sub(self, other: Self) -> Self::Output {
        self + other.neg()
    }
}

impl<T: Ord + Clone + fmt::Debug> Mul for Dyadic<T> {
    type Output = Dyadic<T>;
    fn mul(self, other: Self) -> Self::Output {
        let mut terms = Set::new();
        for l in self.numer.into_iter() {
            for r in other.numer.clone().into_iter() {
                terms.insert_with(l.clone().mul(r), |v| v.double());
            }
        }
        // a/b * c/d = ac / bd
        Dyadic { numer: terms, denom: self.denom * other.denom }
    }
}

impl<T: Ord + fmt::Display + Clone> Specializable<T> for Dyadic<T> {
    fn specialize(&mut self, id: T, val: u8) -> Result<(), SpecializeError<T>> {
        let mut numer = Set::new();
        for v in self.numer.iter() {
            let mut vv = v.clone();
            vv.specialize(id.clone(), val)?;
            numer.insert(vv);
        }
        self.denom.specialize(id, val)?;
        self.numer = numer;
        Ok(())
    }

    fn free_vars(&self) -> Set<&T> {
        self.numer.iter().flat_map(|x|x.free_vars()).collect::<Set<&T>>().union(self.denom.free_vars())
    }
}

impl<T: Ord + Clone> Normalizable for Dyadic<T> {
    fn normalize(&mut self) {
        let mut acc = Bin::lit(0);
        for v in self.numer.iter() {
            let mut t = v.clone();
            t.normalize();
            acc = t.bin.gcd(acc);
        }

        let mut numer = Set::new();
        for v in self.numer.iter() {
            numer.insert(v.clone().div_bin(acc.clone()));
        }

        self.numer = numer;
        self.denom *= acc;
    }
}
