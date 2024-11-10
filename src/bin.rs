use std::fmt;
use std::ops::{Mul, MulAssign, Div, DivAssign};
use crate::context::Set;
use crate::lin::Lin;
use crate::traits::Eval;
use pretty::{DocAllocator, DocBuilder, BoxAllocator, Pretty};

////////////////////////////////////////////////////////////////////////////////////////
// Implementing Bin (2^Lin) operations
////////////////////////////////////////////////////////////////////////////////////////
/// Represents a type-level power of two (2^Lin)
#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub struct Bin<Id> { pub exp: Lin<Id> }

/// Identity element of multiplication is 2^0
impl<T: Ord> Default for Bin<T> {
    fn default() -> Self {
        Bin { exp: Lin::default() }
    }
}

impl<T: Ord> Bin<T> {
    pub fn lit(a: i32) -> Self {
        Bin { exp: Lin::lit(a) }
    }
    pub fn var(v: T) -> Self {
        Bin { exp: Lin::var(v) }
    }
    pub fn one() -> Self {
        Bin { exp: Lin::zero() }
    }
    /// Partial order extends to [Bin] as [2^-] is monotone
    pub fn leq(&self, other: &Self) -> bool where T: Clone {
        self.exp.leq(&other.exp)
    }
    /// Logarithm with "remainder"
    /// ex: log2(12) = (2, 3)    [means 2^2 * 3]
    ///     log(-72) = (3, -9)   [means 2^3 * (-9)]
    pub fn log2(u: i32) -> (Bin<T>, i32) {
        let mut exp = 0;
        let mut um = u.abs();

        while um % 2 == 0 && um > 0 {
            exp += 1;
            um /= 2;
        }
        (Bin { exp: Lin::lit(exp) }, if u > 0 { um } else { -um })
    }

    pub fn inv(self) -> Self where T: Clone {
        Bin { exp : self.exp.neg() }
    }
    /// Least common multiple of two exponents of two
    pub fn lub(&self, other: &Self) -> Self where T: Clone {
        Bin { exp: self.exp.clone().lub(other.exp.clone()) }
    }

    /// Greatest common divisor of two exponents of two
    pub fn glb(&self, other: &Self) -> Self where T: Clone {
        Bin { exp : self.exp.clone().glb(other.exp.clone()) }
    }
    /// Biggest element
    pub fn max() -> Self {
        Bin { exp: Lin::lit(i32::MAX) }
    }
    /// Smallest binary power is 1
    pub fn min() -> Self {
        Bin::one()
    }
}

/// Multiplication of powers of two is equivalent to adding the exponents
impl<T: Ord> MulAssign for Bin<T> {
    fn mul_assign(&mut self, other: Self) {
        self.exp += other.exp;
    }
}

impl<T: Ord + Clone> Mul for Bin<T> {
    type Output = Bin<T>;
    fn mul(self, a: Self) -> Self::Output {
        Bin { exp: self.exp + a.exp }
    }
}

impl<T: Ord + Clone> Mul for &Bin<T> {
    type Output = Bin<T>;
    fn mul(self, a: Self) -> Self::Output {
        self.clone() * a.clone()
    }
}
/// Division of powers of two is equivalent to subtracting the exponents
impl<T: Ord> DivAssign for Bin<T> {
    fn div_assign(&mut self, a: Self) {
        self.exp -= a.exp;
    }
}
impl<T: Ord + Clone> Div for Bin<T> {
    type Output = Bin<T>;
    fn div(self, a: Self) -> Self::Output {
        Bin { exp: self.exp - a.exp }
    }
}

impl<T: Ord + Clone> Div for &Bin<T> {
    type Output = Bin<T>;
    fn div(self, a: Self) -> Bin<T> {
        self.clone() / a.clone()
    }
}

/// Specialize a bin power by substituting a variable with a literal
impl<Id: Ord + Clone + fmt::Display> Eval<Id, i32> for Bin<Id> {
    type ReflectError = ();
    fn specialize(&mut self, id: &Id, val: i32) {
        self.exp.specialize(id, val)
    }
    fn free_vars(&self) -> Set<&Id> {
        self.exp.free_vars()
    }
    fn reflect(&self) -> Result<i32, ()> {
        Ok(1 << self.exp.reflect()?)
    }
}

////////////////////////////////////////////////////////////////////////////////////////
// Pretty Formatting, Display & Arbitrary for Bin
////////////////////////////////////////////////////////////////////////////////////////
impl<'a, D, A, T> Pretty<'a, D, A> for Bin<T>
where
    D: DocAllocator<'a, A>,
    D::Doc: Clone,
    A: 'a + Clone,
    T: Pretty<'a, D, A> + Clone + Ord
{
    fn pretty(self, allocator: &'a D) -> DocBuilder<'a, D, A> {
        allocator.text("2^(")
            .append(self.exp.pretty(allocator))
            .append(allocator.text(")"))
    }
}

/// Display instance calls the pretty printer
impl<'a, T> fmt::Display for Bin<T>
where
    T: Pretty<'a, BoxAllocator, ()> + Clone + Ord
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        <Bin<T> as Pretty<'_, BoxAllocator, ()>>::pretty(self.clone(), &BoxAllocator)
            .1
            .render_fmt(100, f)
    }
}

/// Arbitrary instance for Bin
#[cfg(test)] use arbitrary::{Arbitrary, Unstructured};
#[cfg(test)]
impl<'a, T: Ord + Clone + Arbitrary<'a>> Arbitrary<'a> for Bin<T> {
    fn arbitrary(u: &mut Unstructured<'a>) -> arbitrary::Result<Self> {
        Ok(Bin { exp: Lin::arbitrary(u)? })
    }
}

////////////////////////////////////////////////////////////////////////////////////////
// Unit Tests for Bin
////////////////////////////////////////////////////////////////////////////////////////
#[test]
fn test_bin_mul() {
    assert_eq!(
        // 2^1 * 2^2 * 2^x = 2^(3 + x)
        Bin::lit(1) * Bin::lit(2) * Bin::var("x"),
        Bin::var("x") * Bin::lit(3)
    )
}

#[test]
fn test_bin_div() {
    let a = Bin::lit(3) * Bin::lit(2) * Bin::var("x");
    let b = Bin::lit(2) * Bin::var("y") * Bin::var("x");
    assert_eq!(a / b, (Bin::lit(3) * Bin::var("y").inv()));
}

#[test]
fn test_bin_lub() {
    let a = Bin::lit(3) * Bin::lit(2) * Bin::var("x");
    let b = Bin::lit(2) * Bin::var("y") * Bin::var("x");
    assert_eq!(a.lub(&b), Bin::lit(3) * Bin::lit(2) * Bin::var("x") * Bin::var("y"));
}

#[test]
fn test_bin_log2() {
    assert_eq!(Bin::<&str>::log2(12), (Bin::lit(2), 3));
    assert_eq!(Bin::<&str>::log2(-96), (Bin::lit(5), -3));
}

#[test]
fn test_bin_specialize() {
    let l = Bin::var("x") * Bin::var("y") * Bin::lit(1);

    let mut l1 = l.clone();
    l1.specialize(&"x", 2);
    assert_eq!(l1, Bin::var("y") * Bin::lit(3));

    let mut l2 = l.clone();
    l2.specialize(&"y", 2);
    assert_eq!(
        l2,
        Bin::var("x") * Bin::lit(3)
    );

    let mut l3 = l.clone();
    l3.specialize(&"z", 2);
    assert_eq!(l, l3);
}

#[cfg(test)] use arbtest::arbtest;
#[cfg(test)] use crate::id::Id;

#[test]
fn test_bin_mul_prop() {
    // Commutativity
    arbtest(|u| {
        let a = u.arbitrary::<Bin<Id>>()?;
        let b = u.arbitrary::<Bin<Id>>()?;
        assert_eq!(&a * &b, &b * &a);
        Ok(())
    });
    // Associativity
    arbtest(|u| {
        let a = u.arbitrary::<Bin<Id>>()?;
        let b = u.arbitrary::<Bin<Id>>()?;
        let c = u.arbitrary::<Bin<Id>>()?;
        assert_eq!(&a * &(&b * &c), &(&a * &b) * &c);
        Ok(())
    });
    // Units
    arbtest(|u| {
        let a = u.arbitrary::<Bin<Id>>()?;
        assert_eq!(&a * &Bin::default(), &Bin::default() * &a);
        Ok(())
    });

    // Double and half
    arbtest(|u| {
        let a = u.arbitrary::<Bin<Id>>()?;
        assert_eq!(a.clone() * Bin::lit(1) / a.clone(), Bin::lit(1));
        assert_eq!(a.clone() * Bin::lit(1) / Bin::lit(1), a);
        Ok(())
    });
}

#[test]
fn test_bin_div_prop() {
    // Cancellativity
    arbtest(|u| {
        let a = u.arbitrary::<Bin<Id>>()?;
        assert_eq!(&a / &a, Bin::default());
        Ok(())
    });
    // Unit and Division
    arbtest(|u| {
        let a = u.arbitrary::<Bin<Id>>()?;
        assert_eq!(&Bin::default() / &a, a.clone().inv());
        assert_eq!(&a / &Bin::default(), a);
        Ok(())
    });
    // Least-common multiple divides evenly
    arbtest(|u| {
        let a = u.arbitrary::<Bin<Id>>()?;
        let b = u.arbitrary::<Bin<Id>>()?;
        let lub = a.lub(&b);
        assert!(a.leq(&lub));
        assert!(b.leq(&lub));
        Ok(())
    });
}

#[test]
fn test_bin_leq_prop() {
    // Reflexivity
    arbtest(|u| {
        let a = u.arbitrary::<Bin<Id>>()?;
        assert!(a.leq(&a));
        Ok(())
    });
    // Terms less than their product
    arbtest(|u| {
        let a = u.arbitrary::<Bin<Id>>()?;
        let b = u.arbitrary::<Bin<Id>>()?;
        assert!(a.leq(&(&a * &b)), "{} <= ({} * {} = {})", a, a, b, &a * &b);
        assert!(b.leq(&(&a * &b)), "{} <= ({} * {} = {})", b, a, b, &a * &b);
        Ok(())
    });
    // Div less than terms
    arbtest(|u| {
        let a = u.arbitrary::<Bin<Id>>()?;
        let b = u.arbitrary::<Bin<Id>>()?;
        assert!((&a / &b).leq(&a), "({} / {} = {}) <= {}", a, b, &a / &b, a);
        Ok(())
    });
}
