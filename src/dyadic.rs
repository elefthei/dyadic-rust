use std::fmt;
use std::ops::{Add, Sub, Mul, MulAssign, Div};
use pretty::{BoxAllocator, Pretty, DocAllocator, DocBuilder};
use crate::context::{Ctx, Set};
use crate::traits::{Specializable, Normalizable};
use crate::bin::Bin;

/// Represents a type-level, signed dyadic monoterm,
/// eg: 3* a * b^2 * c^3 * 2^(d+e+f+2)
///     -2 * x * 2^y
#[derive(Debug, Clone, PartialEq, Eq, Ord, PartialOrd, Hash)]
pub struct Mono<Id> {
    mult: i32,          // 3
    terms: Ctx<Id, u8>, // a * b^2 * c^3
    bin: Bin<Id>        // 2^(d+e+f+2)
}

impl<Id: Ord> Mono<Id> {
    pub fn lit(mult: i32) -> Self where Id: Ord {
        let (p, r) = Bin::log2(mult);
        Mono { mult: r, terms: Ctx::new(), bin: p }
    }
    pub fn var(v: Id) -> Self where Id: Ord {
        Mono { mult:1, terms: Ctx::from([(v, 1)]), bin: Bin::default() }
    }
    pub fn term(v: Id, exp: u8) -> Self {
        Mono { mult: 1, terms: Ctx::from([(v, exp)]), bin: Bin::default() }
    }
    pub fn bin(b: Bin<Id>) -> Self where Id: Ord {
        Mono { mult: 1, terms: Ctx::new(), bin: b }
    }
    pub fn neg(self) -> Self {
        Mono { mult: -self.mult, terms: self.terms, bin: self.bin }
    }
    pub fn termbin(&self) -> (&Ctx<Id, u8>, &Bin<Id>) {
        (&self.terms, &self.bin)
    }
    /// Doubling a term
    pub fn double(self) -> Self where Id: Clone {
        Mono { mult: self.mult, terms: self.terms, bin: self.bin.double() }
    }
    /// Halving a term could fail (ex: 3*a^2*2^c)
    pub fn half(self) -> Option<Self> {
        if self.mult % 2 == 0 && self.mult > 0 { // broken invariant, normalize
            let mut d = Mono { mult: self.mult / 2, terms: self.terms, bin: self.bin };
            d.normalize();
            Some(d)
        } else if let Some(b) = self.bin.half() {
            Some(Mono { mult: self.mult, terms: self.terms, bin: b })
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
    pub fn div_bin(&self, other: &Bin<Id>) -> (Self, Bin<Id>) where Id: Clone {
        let mut res = self.clone();
        let (q, r) = res.bin.div(other.clone());
        res.bin = q;
        (res, r)
    }
}

impl<T: Ord> Default for Mono<T> {
    fn default() -> Self {
        Mono::bin(Bin::default())
    }
}

/// Multiplication for monoterms [normalizes]
impl<T: Ord> MulAssign for Mono<T> {
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

impl<T: Ord + Clone> Mul for Mono<T> {
    type Output = Self;
    fn mul(self, other: Self) -> Self::Output {
        let mut res = self.clone();
        res *= other;
        res
    }
}

impl<T: Ord + Clone> Mul for &Mono<T> {
    type Output = Mono<T>;
    fn mul(self, other: Self) -> Self::Output {
        self.clone() * other.clone()
    }
}

impl<T: Ord> From<Bin<T>> for Mono<T> {
    fn from(b: Bin<T>) -> Self {
        Mono::bin(b)
    }
}

impl<'a> From<&'a str> for Mono<&'a str> {
    fn from(s: &'a str) -> Self {
        Mono::var(s)
    }
}

impl<T: Ord> From<i32> for Mono<T> {
    fn from(l: i32) -> Self {
        Mono::lit(l)
    }
}

impl<T: fmt::Display + Ord + Clone> Specializable<T> for Mono<T> {
    // ex: 12*a^2*b^3*2^(c+1) -> a = 2 -> 48*b^3*2^(c+3)
    fn specialize(&mut self, id: &T, val: u8) {
        if let Some(v) = self.terms.remove(&id) {
            self.mult *= val.pow(v as u32) as i32;
            // We already performed [id] substitution so don't fail
            self.bin.specialize(id, val);
            self.normalize();
        }
        // substitute denominator
        self.bin.specialize(id, val)
    }

    fn free_vars(&self) -> Set<&T> {
        self.terms.keys().union(self.bin.free_vars())
    }
}

impl<T: Ord> Normalizable for Mono<T> {
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
impl<'a, D, A, T> Pretty<'a, D, A> for Mono<T>
where
    D: DocAllocator<'a, A>,
    T: Pretty<'a, D, A> + Clone + Ord,
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
            .append(allocator.text("*"))
            .append(self.bin.pretty(allocator))
    }
}

/// Display instance calls the pretty printer
impl<'a, T> fmt::Display for Mono<T>
where
    T: Pretty<'a, BoxAllocator, ()> + Clone + Ord,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        <Mono<T> as Pretty<'_, BoxAllocator, ()>>::pretty(self.clone(), &BoxAllocator)
            .1
            .render_fmt(100, f)
    }
}

#[cfg(test)] use arbitrary::{Unstructured, Arbitrary};

/// Arbitrary instance for Mono
#[cfg(test)]
impl<'a, T: Ord + Clone + Arbitrary<'a>> Arbitrary<'a> for Mono<T> {
    fn arbitrary(u: &mut Unstructured<'a>) -> arbitrary::Result<Self> {
        Ok(Mono {
            mult : u.int_in_range(0..=9)?,
            terms : Ctx::arbitrary(u)?,
            bin : Bin::arbitrary(u)?
        })
    }
}

/// Finally a dyadic number
/// (Mono + ... + Mono) / Bin
/// Dyadic(Set(), Bin::lit(0)) = Dyadic(Set(Mono::lit(0)), Bin::lit(0)) = 0
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Dyadic<T> { numer: Set<Mono<T>>, denom: Bin<T> }

impl<T: Ord> Dyadic<T> {
    pub fn lit(i: i32) -> Self {
        Dyadic {
            numer: Set::singleton(Mono::lit(i)),
            denom: Bin::default()
        }
    }
    pub fn var(v: T) -> Self {
        Dyadic {
            numer: Set::singleton(Mono::var(v)),
            denom: Bin::default()
        }
    }
    // Unit of addition
    pub fn unit_add() -> Self {
        Dyadic { numer: Set::new(), denom: Bin::default() }
    }
    // Unit of multiplication (2^0 / 2^0)
    pub fn unit_mul() -> Self {
        Dyadic { numer: Set::singleton(Mono::bin(Bin::default())), denom: Bin::default() }
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
    pub fn div_bin(&self, other: &Bin<T>) -> Self
    where
        T: Clone
    {
        let mut s = self.clone();
        s.denom *= other.clone();
        s
    }
}

impl<T: Ord + Clone> Add for Dyadic<T> {
    type Output = Dyadic<T>;
    fn add(self, other: Self) -> Self::Output {
        // Compute denominator as the LCM of the two denominators
        let denom = self.denom.lcm(&other.denom);

        // Compute the left multiplicative factor (remainder is 0 by LCM property)
        let (lm, _) = denom.clone().div(self.denom);
        // Compute the right multiplicative factor
        let (rm, _) = denom.clone().div(other.denom);

        // Multiply each term by the multiplicative factor
        let mut numer: Set<Mono<T>> = self.numer.into_iter().map(|v| v.mul_bin(lm.clone())).collect();
        let r = other.numer.into_iter().map(|v| v.mul_bin(rm.clone()));

        // If a + b + b = a + 2b and a - b + a = a
        for i in r {
            let term = i.termbin();
            let same = numer.extract_if(|v| term == v.termbin());
            let mut mult = 0;
            for v in same.iter() {
                mult += v.mult;
            }
            numer.insert_with(
                Mono { mult, terms: term.0.clone(), bin: term.1.clone() },
                |v| v.double()
            );
        }

        // Return numer / denom
        Dyadic { numer, denom }
    }
}

impl<T: Ord + Clone> Add for &Dyadic<T> {
    type Output = Dyadic<T>;
    fn add(self, other: Self) -> Self::Output {
        self.clone() + other.clone()
    }
}

impl<T: Ord + Clone> Sub for Dyadic<T> {
    type Output = Dyadic<T>;
    fn sub(self, other: Self) -> Self::Output {
        self + other.neg()
    }
}

impl<T: Ord + Clone> Sub for &Dyadic<T> {
    type Output = Dyadic<T>;
    fn sub(self, other: Self) -> Self::Output {
        self.clone() - other.clone()
    }
}

impl<T: Ord + Clone> Mul for Dyadic<T> {
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

impl<T: Ord + Clone> Mul for &Dyadic<T> {
    type Output = Dyadic<T>;
    fn mul(self, other: Self) -> Self::Output {
        self.clone() * other.clone()
    }
}

impl<'a, D, A, T> Pretty<'a, D, A> for Dyadic<T>
where
    D: DocAllocator<'a, A>,
    T: Pretty<'a, D, A> + Clone + Ord,
    D::Doc: Clone,
    A: 'a + Clone,
{
    fn pretty(self, allocator: &'a D) -> DocBuilder<'a, D, A> {
        if self.numer.is_empty() {
            return allocator.text("0 / ").append(self.denom.pretty(allocator));
        } else {
            let num = allocator.intersperse(self.numer.into_iter().map(|v| v.pretty(allocator)), " + ");
            return num.append(allocator.text(" / ")).append(self.denom.pretty(allocator));
        }
    }
}

/// Display instance calls the pretty printer
impl<'a, T> fmt::Display for Dyadic<T>
where
    T: Pretty<'a, BoxAllocator, ()> + Clone + Ord,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        <Dyadic<T> as Pretty<'_, BoxAllocator, ()>>::pretty(self.clone(), &BoxAllocator)
            .1
            .render_fmt(100, f)
    }
}

/// Arbitrary instance for Dyadic
#[cfg(test)]
impl<'a, T: Ord + Clone + Arbitrary<'a>> Arbitrary<'a> for Dyadic<T> {
    fn arbitrary(u: &mut Unstructured<'a>) -> arbitrary::Result<Self> {
        let numer = Set::arbitrary(u)?;
        let denom = Bin::arbitrary(u)?;
        Ok(Dyadic { numer, denom })
    }
}
impl<T: Ord + fmt::Display + Clone> Specializable<T> for Dyadic<T> {
    fn specialize(&mut self, id: &T, val: u8) {
        self.numer.modify(|x| x.specialize(id, val));
        self.denom.specialize(id, val);
    }

    fn free_vars(&self) -> Set<&T> {
        self.numer.iter().flat_map(|x|x.free_vars()).collect::<Set<&T>>().union(self.denom.free_vars())
    }
}

impl<T: Ord + Clone> Normalizable for Dyadic<T> {
    fn normalize(&mut self) {
        // 1. Normalize numerator
        self.numer.modify(|x| x.normalize());
        self.numer.extract_if(|x| x.mult == 0);

        // 2. Take denominator and normalize it
        let mut acc = self.denom.clone();
        acc.normalize();

        // 3. Compute greatest-common divisor of all binary multipliers and denominator
        for v in self.numer.iter() {
            acc = acc.gcd(&v.bin);
        }

        // Divide both numerator and denominator by the GCD
        self.numer.modify(|x| *x = x.div_bin(&acc).0);
        self.denom = self.denom.clone().div(acc).0;
    }
}

////////////////////////////////////////////////////////////////////////////////////////
/* Unit Tests for Mono */
////////////////////////////////////////////////////////////////////////////////////////
#[test]
fn test_mono_mul() {
    // 2 * X * 3 * 2^Y = 6 * X * 2^Y
    let a = Mono::lit(2) * Mono::var("X");
    let b = Mono::lit(3) * Mono::bin(Bin::var("Y"));
    assert_eq!(
        a * b,
        Mono { mult: 3, terms: Ctx::from([("X", 1)]), bin: Bin::var("Y") * Bin::lit(1) }
    );
}

#[test]
fn test_mono_div_bin() {
    // 2 * (2^X * 2^2)
    let a = Mono::lit(2) * Mono::bin(Bin::var("X") * Bin::lit(2));
    let b = Bin::var("X") * Bin::lit(1);
    let c = Bin::var("Y") * Bin::lit(3);
    // 2*(2^X * 2^2) / (2*2^X) = 2^2
    assert_eq!(
        a.div_bin(&b),
        (Mono { mult: 1, terms: Ctx::new(), bin: Bin::lit(2) }, Bin::default())
    );
    // 2*(2^X * 2^2) / (2^Y * 2^3) = 2^x /  2^Y
    assert_eq!(
        a.div_bin(&c),
        (Mono { mult: 1, terms: Ctx::new(), bin: Bin::var("X") }, Bin::var("Y"))
    );
}

#[test]
fn test_mono_specialize() {
    let l = Mono::from("x") * Mono::from("y") * Mono::from(2);

    let mut l1 = l.clone();
    l1.specialize(&"x", 2);
    // 2*x*y -> x = 2 -> 4*y
    assert_eq!(l1, Mono::var("y") * Mono::lit(4));

    let mut l2 = l.clone();
    l2.specialize(&"y", 3);
    // 2*x*y -> y = 3 -> 6*x
    assert_eq!(l2, Mono::var("x") * Mono::lit(6));

    let mut l3 = l.clone();
    l3.specialize(&"z", 3);
    assert_eq!(l3, l);
}
////////////////////////////////////////////////////////////////////////////////////////
/* Unit Tests for Dyadic */
////////////////////////////////////////////////////////////////////////////////////////
#[test]
fn test_dyadic_mul() {
    // 2 * X * 4 / 2^Y = X / 2^(Y-3)
    let a = Dyadic::lit(2) * Dyadic::var("X");
    let b = Dyadic::lit(4).div_bin(&Bin::var("Y"));
    assert_eq!(
        a * b,
        Dyadic {
            numer: Set::singleton(Mono { mult: 1, terms: Ctx::from([("X", 1)]), bin: Bin::lit(3) }),
            denom: Bin::var("Y")
        });
}

#[test]
fn test_dyadic_specialize() {
    let l = Dyadic::var("x") * Dyadic::var("y") * Dyadic::lit(2);

    let mut l1 = l.clone();
    l1.specialize(&"x", 2);
    // 2*x*y -> x = 2 -> 4*y
    assert_eq!(l1, Dyadic::var("y") * Dyadic::lit(4));

    let mut l2 = l.clone();
    l2.specialize(&"y", 3);
    // 2*x*y -> y = 3 -> 6*x
    assert_eq!(l2, Dyadic::var("x") * Dyadic::lit(6));

    let mut l3 = l.clone();
    l3.specialize(&"z", 3);
    assert_eq!(l3, l);
}

#[test]
fn test_dyadic_normalize() {
    let l =
        Dyadic { numer:
            Set::from([
                Mono::var("x") * Mono::var("y") * Mono::lit(4),
                Mono::var("x") * Mono::lit(4)
            ]),
                denom: Bin::lit(2)
        };

    let mut l1 = l.clone();
    l1.normalize();
    assert_eq!(l1, Dyadic { numer: Set::from([Mono::var("x") * Mono::var("y"), Mono::var("x")]), denom: Bin::lit(0) });
}

////////////////////////////////////////////////////////////////////////////////////////
/* Prop tests */
////////////////////////////////////////////////////////////////////////////////////////
#[cfg(test)] use arbtest::arbtest;
#[cfg(test)] use crate::id::Id;

// Match two expressions, like `assert_eqn!(a, b)` modulo beta-equivalence
#[cfg(test)]
#[macro_export]
macro_rules! assert_eqn {
    ($left:expr, $right:expr) => ({
        if !Normalizable::eqn(&$left, &$right) {
            let mut l = $left.clone();
            let mut r = $right.clone();
            l.normalize();
            r.normalize();
            panic!(
                "Assertion failed: `assert_eqn!({}, {})`\n  left (pre): `{}`,\n  right (pre): `{}`,\n  left (post): `{}`,\n  right (post): `{}`",
                stringify!($left), stringify!($right), $left, $right, l, r
            );
        }
    });
}

#[test]
fn test_mono_mul_prop() {
    // Associativity of multiplication for monoterms
    arbtest(|u| {
        let a = u.arbitrary::<Mono<Id>>()?;
        let b = u.arbitrary::<Mono<Id>>()?;
        let c = u.arbitrary::<Mono<Id>>()?;
        assert_eqn!(&a * &(&b * &c), &(&a * &b) * &c);
        Ok(())
    });

    // Commutativity
    arbtest(|u| {
        let a = u.arbitrary::<Mono<Id>>()?;
        let b = u.arbitrary::<Mono<Id>>()?;
        assert_eqn!(&a * &b, &b * &a);
        Ok(())
    });

    // Unit
    arbtest(|u| {
        let a = u.arbitrary::<Mono<Id>>()?;
        assert_eqn!(&a * &Mono::default(), a);
        assert_eqn!(&Mono::default() * &a, a);
        Ok(())
    });

    // Double and half
    arbtest(|u| {
        let a = u.arbitrary::<Mono<Id>>()?;
        assert_eqn!(a.clone().double().half().unwrap(), a);
        Ok(())
    });
}

#[test]
fn test_dyadic_add_prop() {
    // Associativity of addition for dyadic numbers
    arbtest(|u| {
        let a = u.arbitrary::<Dyadic<Id>>()?;
        let b = u.arbitrary::<Dyadic<Id>>()?;
        let c = u.arbitrary::<Dyadic<Id>>()?;
        assert_eqn!(&a + &(&b + &c), &(&a + &b) + &c);
        Ok(())
    });

    // Commutativity of addition
    arbtest(|u| {
        let a = u.arbitrary::<Dyadic<Id>>()?;
        let b = u.arbitrary::<Dyadic<Id>>()?;
        assert_eqn!(&a + &b, &b + &a);
        Ok(())
    });

    // Unit of addition
    arbtest(|u| {
        let a = u.arbitrary::<Dyadic<Id>>()?;
        assert_eqn!(&a + &Dyadic::unit_add(), a);
        assert_eqn!(&Dyadic::unit_add() + &a, a);
        Ok(())
    });
}

#[test]
fn test_dyadic_sub_prop() {
    // Cancellativity
    arbtest(|u| {
        let a = u.arbitrary::<Dyadic<Id>>()?;
        assert_eqn!(&a - &a, Dyadic::<Id>::unit_add());
        Ok(())
    });

    // Unit of addition
    arbtest(|u| {
        let a = u.arbitrary::<Dyadic<Id>>()?;
        assert_eqn!(&a - &Dyadic::unit_add(), a);
        assert_eqn!(&Dyadic::unit_add() - &a, a.clone().neg());
        Ok(())
    });

    // Negate twice (idempotence)
    arbtest(|u| {
        let a = u.arbitrary::<Dyadic<Id>>()?;
        assert_eqn!(a, a.clone().neg().neg());
        Ok(())
    });
}

#[test]
fn test_dyadic_mul_prop() {
    // Associativity of addition for dyadic numbers
    arbtest(|u| {
        let a = u.arbitrary::<Dyadic<Id>>()?;
        let b = u.arbitrary::<Dyadic<Id>>()?;
        let c = u.arbitrary::<Dyadic<Id>>()?;
        assert_eqn!(&a * &(&b * &c), &(&a * &b) * &c);
        Ok(())
    });

    // Commutativity of addition
    arbtest(|u| {
        let a = u.arbitrary::<Dyadic<Id>>()?;
        let b = u.arbitrary::<Dyadic<Id>>()?;
        assert_eqn!(&a * &b, &b * &a);
        Ok(())
    });

    // Unit of addition
    arbtest(|u| {
        let a = u.arbitrary::<Dyadic<Id>>()?;
        assert_eqn!(&a * &Dyadic::unit_mul(), a);
        assert_eqn!(&Dyadic::unit_mul() * &a, a);
        Ok(())
    });

    // Double and half
    arbtest(|u| {
        let a = u.arbitrary::<Dyadic<Id>>()?;
        assert_eqn!(a.clone().double().half(), a);
        Ok(())
    });
}
