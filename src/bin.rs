use std::fmt;
use std::ops::{Add, AddAssign, Sub, Mul, MulAssign, Div};
use crate::context::{Set, Ctx};
use crate::traits::{Specializable, SpecializeError, Normalizable};
use pretty::{DocAllocator, DocBuilder, BoxAllocator, Pretty};
use arbitrary::{Arbitrary, Result, Unstructured};

/// Represents a type-level positive, linear expression
/// ex: 2a + 3b + 4c + 5
/// The invariant is that multipliers are non-zero
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct Lin<Id>(Ctx<Id, u8>, u8);

/// Represents a type-level power of two (2^Lin)
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct Bin<Id> { exp: Lin<Id> }

/// Identity element of addition is 0
impl<T : Ord> Default for Lin<T> {
    fn default() -> Self {
        Lin::lit(0)
    }
}

/// Identity element of multiplication is 2^0
impl<T: Ord> Default for Bin<T> {
    fn default() -> Self {
        Bin { exp: Lin::default() }
    }
}

impl<T: Ord> Lin<T> {
    /// Create a linear constant
    pub fn lit(a: u8) -> Self {
        Lin(Ctx::new(), a)
    }

    /// Create a linear variable
    pub fn var(v: T) -> Self {
        Lin(Ctx::from([(v, 1)]), 0)
    }

    /// Create a linear variable with multiplier
    pub fn term(v: T, a: u8) -> Self {
        if a == 0 {
            Lin::default()
        } else {
            Lin(Ctx::from([(v, a)]), 0)
        }
    }

    /// Define a partial order for linear, positive expressions
    /// true:  2*a + b <= 3*a + b + c
    ///        {} <= a
    /// false: 4*a <= 3*a + b + c
    /// None:  b <= c
    pub fn leq(&self, other: &Self) -> Option<bool> {
        let mut le = true;
        for (k, v) in self.0.iter() {
            if let Some(vr) = other.0.get(&k) {
                if v > vr {
                    le = false;
                }
            } else {
                return None;
            }
        }
        Some(le && self.1 <= other.1)
    }
}

impl<T: Ord> AddAssign for Lin<T> {
    /// Add two linear terms
    fn add_assign(&mut self, other: Self) {
        self.0.append_with(other.0.into_iter(), &|a, b| a + b);
        self.1 += other.1;
    }
}

impl<T: Ord + Clone> Add for Lin<T> {
    type Output = Lin<T>;
    /// Add two linear terms
    fn add(self, other: Self) -> Self::Output {
        let mut c = self.clone();
        c += other;
        c
    }
}

impl<T: Ord + Clone> Sub for Lin<T> {
    type Output = (Lin<T>, Lin<T>);
    /// Subtract two linear terms (with remainder)
    fn sub(self, other: Self) -> Self::Output {
        let mut n: u8 = self.1;
        let mut m: u8 = other.1;
        if n < m { // 3 - 4 = (0 ,1)
            m -= n;
            n = 0;
        } else {   // 4 - 1 = (3, 0)
            n -= m;
            m = 0;
        }
        let mut nvars = self.0.clone();
        let mut mvars = other.0.clone();
        for (k, mx) in mvars.iter_mut() {
            if let Some(nx) = nvars.get_mut(k) {
                if *nx < *mx {
                    *mx -= *nx;
                    *nx = 0;
                } else {
                    *nx -= *mx;
                    *mx = 0;
                }
            }
        }
        nvars.retain(|_, v| *v > 0);
        mvars.retain(|_, v| *v > 0);
       (Lin(nvars, n), Lin(mvars, m))
    }
}

/// Remove all zero elements (0*a + c = c)
impl<T: Ord + Clone> Normalizable for Lin<T> {
    fn normalize(&mut self) {
        self.0.retain(|_, v| *v > 0);
    }
}

/// Specialize a linear term by substituting a variable
impl<T: Ord + fmt::Display + Clone> Specializable<T> for Lin<T> {
    fn specialize(&mut self, id: T, val: u8) -> Result<(), SpecializeError<T>> {
        if let Some(v) = self.0.remove(&id) {
            self.1 += v * val;
            Ok(())
        } else {
            Err(SpecializeError::VarNotFound(id))
        }
    }

    fn free_vars(&self) -> Set<&T> {
        self.0.keys()
    }
}

////////////////////////////////////////////////////////////////////////////////////////
// Implementing Bin (2^Lin) operations
////////////////////////////////////////////////////////////////////////////////////////
impl<T: Ord> Bin<T> {
    pub fn lit(a: u8) -> Self {
        Bin { exp: Lin::lit(a) }
    }
    pub fn var(v: T) -> Self {
        Bin{ exp: Lin::var(v) }
    }
    pub fn double(self) -> Self where T: Clone {
        Bin { exp: self.exp + Lin::lit(1) }
    }
    /// Halving could fail, ex: 2^a / 2 = None
    pub fn half(self) -> Option<Self> {
        if self.exp.1 > 0 {
            Some(Bin { exp: Lin(self.exp.0, self.exp.1 - 1) })
        } else {
            None
        }
    }
    /// Partial order extends to [Bin] as [2^-] is monotone
    pub fn leq(&self, other: &Self) -> Option<bool> {
        self.exp.leq(&other.exp)
    }
    /// Logarithm with remainder
    /// ex: log2(9) = (3, 1)
    ///     log(-129) = (7, -1)
    pub fn log2(u: i32) -> (Bin<T>, i32) {
        let mut exp = 0;
        let mut um = u.abs();

        while um % 2 == 0 && um > 0 {
            exp += 1;
            um /= 2;
        }
        (Bin { exp: Lin::lit(exp) }, if u > 0 { um } else { -um })
    }

    /// Least common multiple of two exponents of two
    pub fn lcm(&self, other: Self) -> Self where T: Clone {
        Bin { exp: Lin(
            self.exp.0.union_with(other.exp.0, &|a, b| std::cmp::max(a, b)),
            std::cmp::max(self.exp.1, other.exp.1)
        )}
    }

    /// Greatest common divisor of two exponents of two
    pub fn gcd(&self, other: Self) -> Self where T: Clone {
        Bin { exp : Lin(
            self.exp.0.intersection_with(other.exp.0, &|a, b| std::cmp::min(a, b)),
            std::cmp::min(self.exp.1, other.exp.1)
        )}
    }

    /// Minimum (partial) of two exponents of two
    pub fn min(self, other: Self) -> Option<Self> {
        if self.leq(&other)? {
            Some(self)
        } else {
            Some(other)
        }
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

/// Division of powers of two is equivalent to subtracting the exponents
impl<T: Ord + Clone> Div for Bin<T> {
    type Output = (Bin<T>, Bin<T>);

    fn div(self, a: Self) -> Self::Output {
        let (q, r) = self.exp - a.exp;
        (Bin { exp: q }, Bin { exp: r })
    }
}

/// Specialize a bin power by substituting a variable with a literal
impl<T: Ord + fmt::Display + Clone> Specializable<T> for Bin<T> {
    fn specialize(&mut self, id: T, val: u8) -> Result<(), SpecializeError<T>> {
        self.exp.specialize(id, val)
    }
    fn free_vars(&self) -> Set<&T> {
        self.exp.0.keys()
    }
}
/// Remove all zero elements (0*a + c = c)
impl<T: Ord + Clone> Normalizable for Bin<T> {
    fn normalize(&mut self) {
        self.exp.normalize();
    }
}
////////////////////////////////////////////////////////////////////////////////////////
// Pretty Formatting, Display & Arbitrary for Lin and  Bin
////////////////////////////////////////////////////////////////////////////////////////
impl<'a, D, A, T> Pretty<'a, D, A> for Lin<T>
where
    D: DocAllocator<'a, A>,
    D::Doc: Clone,
    A: 'a + Clone,
    T: Pretty<'a, D, A> + Clone
{
    fn pretty(self, allocator: &'a D) -> DocBuilder<'a, D, A> {
        allocator.intersperse(self.0.into_iter()
                    .map(|(k, v)| allocator.text(v.to_string()).append(k.pretty(allocator))), "+")
    }
}

/// Display instance calls the pretty printer
impl<'a, T> fmt::Display for Lin<T>
where
    T: Pretty<'a, BoxAllocator, ()> + Clone
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        <Lin<T> as Pretty<'_, BoxAllocator, ()>>::pretty(self.clone(), &BoxAllocator)
            .1
            .render_fmt(100, f)
    }
}

/// Arbitrary instance for Lin
impl<'a, T: Ord + Arbitrary<'a>> Arbitrary<'a> for Lin<T> {
    fn arbitrary(u: &mut Unstructured<'a>) -> Result<Self> {
        Ok(Lin(Ctx::arbitrary(u)?, u.int_in_range(0..=9)?))
    }
}

impl<'a, D, A, T> Pretty<'a, D, A> for Bin<T>
where
    D: DocAllocator<'a, A>,
    D::Doc: Clone,
    A: 'a + Clone,
    T: Pretty<'a, D, A> + Clone
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
    T: Pretty<'a, BoxAllocator, ()> + Clone
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        <Bin<T> as Pretty<'_, BoxAllocator, ()>>::pretty(self.clone(), &BoxAllocator)
            .1
            .render_fmt(100, f)
    }
}

/// Arbitrary instance for Bin
impl<'a, T: Ord + Arbitrary<'a>> Arbitrary<'a> for Bin<T> {
    fn arbitrary(u: &mut Unstructured<'a>) -> Result<Self> {
        Ok(Bin { exp: Lin::arbitrary(u)? })
    }
}

////////////////////////////////////////////////////////////////////////////////////////
// Unit Tests for Lin
////////////////////////////////////////////////////////////////////////////////////////
#[test]
fn test_lin_add() {
    assert_eq!(
        Lin::lit(1) + Lin::lit(2) + Lin::var("x"),
        Lin::var("x") + Lin::lit(3)
    )
}

#[test]
fn test_lin_sub() {
    assert_eq!((Lin::lit(3) + Lin::lit(2) + Lin::var("x"))
        - (Lin::lit(2) + Lin::var("y") + Lin::var("x")),
        (Lin::lit(3), Lin::var("y")));
}

#[test]
fn test_leq_lin() {
    assert_eq!(
        Lin::leq(
            &(Lin::lit(2) + Lin::var("a")),
            &(Lin::term("a", 2) + Lin::var("b") + Lin::lit(4))
        ),
        Some(true)
    );
    assert_eq!(
        Lin::leq(
            &(Lin::lit(2) + Lin::var("c")),
            &(Lin::term("a", 2) + Lin::var("b") + Lin::lit(4))
        ),
        None
    );
    assert_eq!(
        Lin::leq(
            &(Lin::term("a", 3) + Lin::var("c")),
            &(Lin::term("a", 2) + Lin::var("b") + Lin::lit(4))
        ),
        Some(false)
    );
}

#[test]
fn test_lin_specialize() {
    let mut l = Lin::var("x") + Lin::var("y") + Lin::lit(1);

    let mut l1 = l.clone();
    l1.specialize("x", 2).unwrap();
    assert_eq!(l1, Lin::var("y") + Lin::lit(3));

    let mut l2 = l.clone();
    l2.specialize("y", 2).unwrap();
    assert_eq!(l2, Lin::var("x") + Lin::lit(3));

    let mut l3 = l.clone();
    assert_eq!(l3.specialize("z", 2),
        Err(SpecializeError::VarNotFound("z")));
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
    assert_eq!(a / b, (Bin::lit(3), Bin::var("y")));
}

#[test]
fn test_bin_lcm() {
    let a = Bin::lit(3) * Bin::lit(2) * Bin::var("x");
    let b = Bin::lit(2) * Bin::var("y") * Bin::var("x");
    assert_eq!(a.lcm(b), Bin::lit(3) * Bin::lit(2) * Bin::var("x") * Bin::var("y"));
}

#[test]
fn test_bin_log2() {
    assert_eq!(Bin::<&str>::log2(9), (Bin::lit(3), 1));
    assert_eq!(Bin::<&str>::log2(-129), (Bin::lit(7), -1));
}

#[test]
fn test_bin_min() {
    let a = Bin::lit(3) * Bin::var("y") * Bin::var("x");
    let b = Bin::lit(2) * Bin::var("x");
    assert_eq!(a.min(b.clone()), Some(b));
}

#[test]
fn test_bin_specialize() {
    let l = Bin::var("x") * Bin::var("y") * Bin::lit(1);

    let mut l1 = l.clone();
    l1.specialize("x", 2).unwrap();
    assert_eq!(l1, Bin::var("y") * Bin::lit(3));

    let mut l2 = l.clone();
    l2.specialize("y", 2).unwrap();
    assert_eq!(
        l2,
        Bin::var("x") * Bin::lit(3)
    );

    let mut l3 = l.clone();
    assert_eq!(
        l3.specialize("z", 2),
        Err(SpecializeError::VarNotFound("z"))
    );
}
