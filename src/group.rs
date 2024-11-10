use std::fmt;
use std::ops::{Add, AddAssign, Sub, SubAssign, Mul, MulAssign, Div, DivAssign};
use pretty::{BoxAllocator, Pretty, DocAllocator, DocBuilder};
use crate::context::{Set, Ctx};
use crate::traits::Eval;
use thiserror::Error;
use crate::bin::Bin;

/// Algebraic Groups of binary numbers for vector sizes.
/// Example:
///
/// ```zippel
/// a: [F; 2^n] + b: [F; 2^(3+m)] : [F; 2^n + 2^(3+m)]
/// ```
/// BinGroup eg: 3*2^n + 4*2^(3-m) /
#[derive(Debug, Clone, PartialEq, Eq, Ord, PartialOrd, Hash)]
pub struct BinGroup<T>(Ctx<Bin<T>, i32>);

impl<T: Ord> BinGroup<T> {
    pub fn lit(i: i32) -> Self {
        if i == 0 {
            BinGroup::zero()
        } else {
            let (p, q) = Bin::log2(i);
            BinGroup(Ctx::singleton(p, q))
        }
    }

    pub fn var(v: T) -> Self {
        BinGroup(Ctx::singleton(Bin::var(v), 1))
    }
    pub fn bin(b: Bin<T>) -> Self {
        BinGroup(Ctx::singleton(b, 1))
    }
    /// Unit of addition
    pub fn zero() -> Self {
        BinGroup(Ctx::new())
    }
    /// Unit of multiplication (2^0)
    pub fn one() -> Self {
        BinGroup(Ctx::singleton(Bin::default(), 1))
    }
    /// -p
    pub fn neg(self) -> Self {
        BinGroup(self.0.into_iter().map(|(k, v)| (k, -v)).collect())
    }
    /// Maximum element
    pub fn max() -> Self {
        BinGroup(Ctx::singleton(Bin::max(), i32::MAX))
    }
    /// Minimum element
    pub fn min() -> Self {
        BinGroup(Ctx::singleton(Bin::max(), i32::MIN))
    }
}

//////////////////////////////////////////////////////////////////////////////////////////////
/// Addition of BinGroup numbers
//////////////////////////////////////////////////////////////////////////////////////////////
impl<T: Ord> AddAssign<i32> for BinGroup<T> {
    fn add_assign(&mut self, other: i32) {
        let (bin, r) = Bin::log2(other);
        if let Some(v) = self.0.remove(&bin) {
            let (mut p, q) = Bin::log2(v+r);
            p *= bin;
            self.0.insert(p, q)
        } else {
            self.0.insert(bin, r)
        }
    }
}

impl<T: Ord> AddAssign<Bin<T>> for BinGroup<T> {
    fn add_assign(&mut self, other: Bin<T>) {
        if let Some(v) = self.0.remove(&other) {
            let (mut p, q) = Bin::log2(v+1);
            p *= other;
            self.0.insert(p, q)
        } else {
            self.0.insert(other, 1)
        }
    }
}

impl<T: Ord> AddAssign for BinGroup<T> {
    fn add_assign(&mut self, other: Self) {
        for (bin, mult) in other.0.into_iter() {
            if let Some(v) = self.0.remove(&bin) {
                let (mut p, q) = Bin::log2(v+mult);
                p *= bin;
                self.0.insert(p, q)
            } else {
                self.0.insert(bin, mult)
            }
        }
        // Normalization
        self.0.retain(|_, v| *v != 0);
    }
}

impl<T: Ord + Clone> Add for BinGroup<T> {
    type Output = BinGroup<T>;
    fn add(self, other: Self) -> Self::Output {
        let mut res = self.clone();
        res += other;
        res
    }
}

impl<T: Ord + Clone> Add<Bin<T>> for BinGroup<T> {
    type Output = BinGroup<T>;
    fn add(self, other: Bin<T>) -> Self::Output {
        let mut res = self.clone();
        res += other;
        res
    }
}

impl<T: Ord + Clone> Add for &BinGroup<T> {
    type Output = BinGroup<T>;
    fn add(self, other: Self) -> Self::Output {
        self.clone() + other.clone()
    }
}

impl<T: Ord + Clone> Add<&Bin<T>> for &BinGroup<T> {
    type Output = BinGroup<T>;
    fn add(self, other: &Bin<T>) -> Self::Output {
        self.clone() + other.clone()
    }
}

//////////////////////////////////////////////////////////////////////////////////////////////
/// Subtraction of BinGroup numbers
//////////////////////////////////////////////////////////////////////////////////////////////
impl<T: Ord> SubAssign<i32> for BinGroup<T> {
    fn sub_assign(&mut self, other: i32) {
        self.add_assign(-other)
    }
}

impl<T: Ord> SubAssign<Bin<T>> for BinGroup<T> {
    fn sub_assign(&mut self, other: Bin<T>) {
        if let Some(v) = self.0.remove(&other) {
            let (mut p, q) = Bin::log2(v-1);
            p *= other;
            self.0.insert(p, q)
        } else {
            self.0.insert(other, -1)
        }
    }
}

impl<T: Ord> SubAssign for BinGroup<T> {
    fn sub_assign(&mut self, other: Self) {
        *self += other.neg();
    }
}

impl<T: Ord + Clone> Sub for BinGroup<T> {
    type Output = BinGroup<T>;
    fn sub(self, other: Self) -> Self::Output {
        let mut res = self.clone();
        res -= other;
        res
    }
}

impl<T: Ord + Clone> Sub<Bin<T>> for BinGroup<T> {
    type Output = BinGroup<T>;
    fn sub(self, other: Bin<T>) -> Self::Output {
        let mut res = self.clone();
        res -= other;
        res
    }
}

impl<T: Ord + Clone> Sub for &BinGroup<T> {
    type Output = BinGroup<T>;
    fn sub(self, other: Self) -> Self::Output {
        self.clone() - other.clone()
    }
}

impl<T: Ord + Clone> Sub<&Bin<T>> for &BinGroup<T> {
    type Output = BinGroup<T>;
    fn sub(self, other: &Bin<T>) -> Self::Output {
        self.clone() - other.clone()
    }
}

//////////////////////////////////////////////////////////////////////////////////////////////
/// Multiplication of BinGroup numbers
//////////////////////////////////////////////////////////////////////////////////////////////
impl<T: Ord + Clone> MulAssign<i32> for BinGroup<T> {
    fn mul_assign(&mut self, other: i32) {
        let (p, r) = Bin::log2(other);
        for (bin, v) in std::mem::take(&mut self.0) {
            self.0.insert(bin * p.clone(), v * r);
        }
    }
}

impl<T: Ord + Clone> MulAssign<Bin<T>> for BinGroup<T> {
    fn mul_assign(&mut self, other: Bin<T>) {
        for (bin, v) in std::mem::take(&mut self.0) {
            self.0.insert(bin * other.clone(), v);
        }
    }
}

impl<T: Ord + Clone> MulAssign for BinGroup<T> {
    fn mul_assign(&mut self, other: Self) {
        for (b1, x1) in std::mem::take(&mut self.0) {
            for (b2, x2) in other.0.iter() {
                let (p, q) = Bin::log2(x1*x2);
                let b = &b1 * &b2 * p;
                if let Some(v) = self.0.remove(&b) {
                    let (b3, r) = Bin::log2(v+q);
                    self.0.insert(b * b3, r);
                } else {
                    self.0.insert(b, q);
                }
            }
        }
    }
}

impl<T: Ord + Clone> Mul for BinGroup<T> {
    type Output = BinGroup<T>;
    fn mul(self, other: Self) -> Self::Output {
        let mut res = self.clone();
        res *= other;
        res
    }
}

impl<T: Ord + Clone> Mul<Bin<T>> for BinGroup<T> {
    type Output = BinGroup<T>;
    fn mul(self, other: Bin<T>) -> Self::Output {
        let mut res = self.clone();
        res *= other;
        res
    }
}

impl<T: Ord + Clone> Mul for &BinGroup<T> {
    type Output = BinGroup<T>;
    fn mul(self, other: Self) -> Self::Output {
        self.clone() * other.clone()
    }
}

impl<T: Ord + Clone> Mul<&Bin<T>> for &BinGroup<T> {
    type Output = BinGroup<T>;
    fn mul(self, other: &Bin<T>) -> Self::Output {
        self.clone() * other.clone()
    }
}

//////////////////////////////////////////////////////////////////////////////////////////////
/// Division of BinGroup numbers by binary powers
//////////////////////////////////////////////////////////////////////////////////////////////
impl<T: Ord + Clone> DivAssign<Bin<T>> for BinGroup<T> {
    fn div_assign(&mut self, other: Bin<T>) {
        *self *= other.inv();
    }
}

impl<T: Ord + Clone> Div<Bin<T>> for BinGroup<T> {
    type Output = BinGroup<T>;
    fn div(self, other: Bin<T>) -> Self::Output {
        let mut res = self.clone();
        res /= other;
        res
    }
}

impl<T: Ord + Clone> Div<&Bin<T>> for &BinGroup<T> {
    type Output = BinGroup<T>;
    fn div(self, other: &Bin<T>) -> Self::Output {
        self.clone() / other.clone()
    }
}

//////////////////////////////////////////////////////////////////////////////////////////////
/// Pretty printing, display and Arbitrary for BinGroup
//////////////////////////////////////////////////////////////////////////////////////////////
impl<'a, D, A, T> Pretty<'a, D, A> for BinGroup<T>
where
    D: DocAllocator<'a, A>,
    T: Pretty<'a, D, A> + Clone + Ord,
    D::Doc: Clone,
    A: 'a + Clone,
{
    fn pretty(self, allocator: &'a D) -> DocBuilder<'a, D, A> {
        if self.0.is_empty() {
            allocator.text("0")
        } else {
            allocator.intersperse(
                self.0.into_iter()
                    .map(|(k, v)|
                        allocator.text(format!("{}*", v)).append(k.pretty(allocator))),
                " + ")
        }
    }
}

/// Display instance calls the pretty printer
impl<'a, T> fmt::Display for BinGroup<T>
where
    T: Pretty<'a, BoxAllocator, ()> + Clone + Ord,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        <BinGroup<T> as Pretty<'_, BoxAllocator, ()>>::pretty(self.clone(), &BoxAllocator)
            .1
            .render_fmt(100, f)
    }
}

#[cfg(test)] use arbitrary::{Arbitrary, Unstructured};

/// Arbitrary instance for BinGroup
#[cfg(test)]
impl<'a, T: Ord + Clone + Arbitrary<'a>> Arbitrary<'a> for BinGroup<T> {
    fn arbitrary(u: &mut Unstructured<'a>) -> arbitrary::Result<Self> {
        let n = u.int_in_range(0..=6)?;
        let mut numer = Ctx::new();
        for _ in 0..=n {
            let bin = Bin::arbitrary(u)?;
            let mult = 2*u.int_in_range(-6..=6)? + 1;
            numer.insert(bin, mult);
        }
        Ok(BinGroup(numer))
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum NegativeSizeError<Id: Ord + Clone + fmt::Display> {
    #[error("Sized type {0} evaluates to a negative size: {1}.")]
    NegativeSize(BinGroup<Id>, i32),
}

impl<Id: Ord + Clone + fmt::Display> Eval<Id, i32> for BinGroup<Id> {
    type ReflectError = NegativeSizeError<Id>;
    fn specialize(&mut self, id: &Id, val: i32) {
        for (mut bin, v) in std::mem::take(&mut self.0) {
            bin.specialize(id, val);
            self.0.insert(bin, v);
        }
    }

    fn free_vars(&self) -> Set<&Id> {
        self.0.iter().flat_map(|(x, _)| x.free_vars()).collect::<Set<&Id>>()
    }

    fn reflect(&self) -> Result<i32, Self::ReflectError> {
        let mut size = 0;
        for (bin, v) in self.0.iter() {
            size += bin.reflect().unwrap() as i32 * *v;
        }
        if size < 0 {
            Err(NegativeSizeError::NegativeSize(self.clone(), size))
        } else {
            Ok(size as i32)
        }
    }
}

////////////////////////////////////////////////////////////////////////////////////////
/* Unit Tests for BinGroup */
////////////////////////////////////////////////////////////////////////////////////////

#[test]
fn test_bingroup_add_unit() {
    let a = BinGroup::var("a") / Bin::var("n");
    let b = BinGroup::var("b") / Bin::var("m");
    assert_eq!(&a + &b,
        BinGroup(Ctx::from([
            (Bin::var("a") * Bin::var("n").inv(), 1),
            (Bin::var("b") * Bin::var("m").inv(), 1)
        ])));
}

#[test]
fn test_bingroup_sub_unit() {
    let a = BinGroup::var("a") / Bin::var("n");
    let b = BinGroup::var("b") / Bin::var("m");
    assert_eq!(&a - &b,
        BinGroup(Ctx::from([
            (Bin::var("a") * Bin::var("n").inv(), 1),
            (Bin::var("b") * Bin::var("m").inv(), -1)
        ])));

    // Cancellativity
    let x = BinGroup::var("a") * BinGroup::lit(2);
    assert_eq!(&x - &x, BinGroup::<&str>::zero());
}

#[test]
fn test_bingroup_mul_unit() {
    // (6 + 2^X) * (8 / 2^Y) = 3*2^{4-Y} + 1*2^(X+3-Y)
    let a = BinGroup::lit(6) + Bin::var("X");
    let b = BinGroup::lit(8) / Bin::var("Y");
    assert_eq!(
        a * b,
        BinGroup(Ctx::from([
            (Bin::lit(4) / Bin::var("Y"), 3),
            (Bin::var("X") * Bin::lit(3) / Bin::var("Y"), 1)
        ])));
}

#[test]
fn test_bingroup_specialize_unit() {
    let l = BinGroup::var("x") * BinGroup::var("y") * BinGroup::lit(2);

    let mut l1 = l.clone();
    l1.specialize(&"x", 2);
    // 2^x* 2^y * 2 -> x = 2 -> 8 * 2^y
    assert_eq!(l1, BinGroup::var("y") * BinGroup::lit(8));

    let mut l2 = l.clone();
    l2.specialize(&"y", 3);
    // 2^x * 2^y * 2 -> y = 3 -> 16 * 2^x
    assert_eq!(l2, BinGroup::var("x") * BinGroup::lit(16));

    let mut l3 = l.clone();
    l3.specialize(&"z", 3);
    assert_eq!(l3, l);
}

////////////////////////////////////////////////////////////////////////////////////////
/* Prop tests */
////////////////////////////////////////////////////////////////////////////////////////
#[cfg(test)] use arbtest::arbtest;
#[cfg(test)] use crate::id::Id;

#[test]
fn test_bingroup_add_prop() {
    // Associativity of addition for bingroup numbers
    arbtest(|u| {
        let a = u.arbitrary::<BinGroup<Id>>()?;
        let b = u.arbitrary::<BinGroup<Id>>()?;
        let c = u.arbitrary::<BinGroup<Id>>()?;
        assert_eq!(&a + &(&b + &c), &(&a + &b) + &c);
        Ok(())
    });

    // Commutativity of addition
    arbtest(|u| {
        let a = u.arbitrary::<BinGroup<Id>>()?;
        let b = u.arbitrary::<BinGroup<Id>>()?;
        assert_eq!(&a + &b, &b + &a);
        Ok(())
    });

    // Unit of addition
    arbtest(|u| {
        let a = u.arbitrary::<BinGroup<Id>>()?;
        assert_eq!(&a + &BinGroup::zero(), a);
        assert_eq!(&BinGroup::zero() + &a, a);
        Ok(())
    });
}

#[test]
fn test_bingroup_sub_prop() {
    // Cancellativity
    arbtest(|u| {
        let a = u.arbitrary::<BinGroup<Id>>()?;
        println!("a = {}", a);
        assert_eq!(&a - &a, BinGroup::<Id>::zero());
        Ok(())
    });

    // Unit of addition
    arbtest(|u| {
        let a = u.arbitrary::<BinGroup<Id>>()?;
        assert_eq!(&a - &BinGroup::zero(), a);
        assert_eq!(&BinGroup::zero() - &a, a.clone().neg());
        Ok(())
    });

    // Negate twice (idempotence)
    arbtest(|u| {
        let a = u.arbitrary::<BinGroup<Id>>()?;
        assert_eq!(a, a.clone().neg().neg());
        Ok(())
    });
}

#[test]
fn test_bingroup_mul_prop() {
    // Associativity of addition for bingroup numbers
    arbtest(|u| {
        let a = u.arbitrary::<BinGroup<Id>>()?;
        let b = u.arbitrary::<BinGroup<Id>>()?;
        let c = u.arbitrary::<BinGroup<Id>>()?;
        assert_eq!(&a * &(&b * &c), &(&a * &b) * &c);
        Ok(())
    });

    // Commutativity of addition
    arbtest(|u| {
        let a = u.arbitrary::<BinGroup<Id>>()?;
        let b = u.arbitrary::<BinGroup<Id>>()?;
        assert_eq!(&a * &b, &b * &a);
        Ok(())
    });

    // Unit of addition
    arbtest(|u| {
        let a = u.arbitrary::<BinGroup<Id>>()?;
        assert_eq!(&a * &BinGroup::one(), a);
        assert_eq!(&BinGroup::one() * &a, a);
        Ok(())
    });
}
