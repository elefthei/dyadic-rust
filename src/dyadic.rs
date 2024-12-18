use std::fmt;
use std::ops::{Add, AddAssign, Sub, SubAssign, Mul, MulAssign, Div, DivAssign};
use pretty::{BoxAllocator, Pretty, DocAllocator, DocBuilder};
use crate::context::{Ctx, Set};
use crate::traits::{Specializable, Normalizable};
use crate::bin::Bin;

#[cfg(test)]
use crate::assert_eqn;

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
        if exp == 0 {
            Mono::lit(1)
        } else {
            Mono { mult: 1, terms: Ctx::from([(v, exp)]), bin: Bin::default() }
        }
    }
    pub fn bin(b: Bin<Id>) -> Self where Id: Ord {
        Mono { mult: 1, terms: Ctx::new(), bin: b }
    }
    pub fn log2(i: i32) -> Self {
        let (p, r) = Bin::log2(i);
        Mono { mult: r, terms: Ctx::new(), bin: p }
    }
    pub fn neg(self) -> Self {
        Mono { mult: -self.mult, terms: self.terms, bin: self.bin }
    }
    /// Doubling a term
    pub fn double(self) -> Self where Id: Clone {
        Mono { mult: self.mult, terms: self.terms, bin: self.bin.double() }
    }
    /// Halving a term could fail (ex: 3*a^2*2^c)
    pub fn half(self) -> Option<Self> {
        if self.mult % 2 == 0 && self.mult > 0 { // broken invariant
            Some(Mono { mult: self.mult / 2, terms: self.terms, bin: self.bin })
        } else if let Some(b) = self.bin.half() {
            Some(Mono { mult: self.mult, terms: self.terms, bin: b })
        } else {
            None
        }
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

/// Multiplication of monoterms [normalizes]
impl<T: Ord> MulAssign<Mono<T>> for Mono<T> {
    fn mul_assign(&mut self, other: Self) {
        for (k, v) in other.terms.into_iter().filter(|(_, v)| *v > 0) {
            self.terms.insert_with(k, v, &|x, y| x + y);
        }
        // No normalization needed
        // if a % 2 == 1, b % 2 == 1, then a * b % 2 == 1
        self.mult *= other.mult;
        self.bin *= other.bin;
    }
}

/// Multiplication of monoterms to bin [normalizes]
impl<T: Ord> MulAssign<Bin<T>> for Mono<T> {
    fn mul_assign(&mut self, other: Bin<T>) {
        self.bin *= other;
    }
}

/// Multiplication of monoterms to integers [normalizes]
impl <T: Ord> MulAssign<i32> for Mono<T> {
    fn mul_assign(&mut self, other: i32) {
        let (p, r) = Bin::log2(self.mult * other);
        self.mult = r;
        self.bin *= p;
    }
}

impl<T: Ord + Clone> Mul<Mono<T>> for Mono<T> {
    type Output = Self;
    fn mul(self, other: Self) -> Self::Output {
        let mut res = self.clone();
        res *= other;
        res
    }
}

impl<T: Ord + Clone> Mul<Bin<T>> for Mono<T> {
    type Output = Self;
    fn mul(self, other: Bin<T>) -> Self::Output {
        let mut res = self.clone();
        res *= other;
        res
    }
}

impl<T: Ord + Clone> Mul<i32> for Mono<T> {
    type Output = Self;
    fn mul(self, other: i32) -> Self::Output {
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

impl<T: Ord + Clone> Mul<&Bin<T>> for &Mono<T> {
    type Output = Mono<T>;
    fn mul(self, other: &Bin<T>) -> Self::Output {
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

impl<T: fmt::Display + Ord + Clone> Specializable<T, u8> for Mono<T> {
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
        self.terms.retain(|_, v| *v > 0);
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
        if self.terms.is_empty() {
            allocator.text(format!("{}", self.mult)).append(allocator.text("*")).append(self.bin.pretty(allocator))
        } else {
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
        let mut mono = Mono {
            mult : u.int_in_range(0..=9)?,
            terms : Ctx::arbitrary(u)?,
            bin : Bin::arbitrary(u)?
        };
        mono.normalize();
        Ok(mono)
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

    pub fn bin(b: Bin<T>) -> Self {
        Dyadic {
            numer: Set::singleton(Mono::bin(b)),
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

//////////////////////////////////////////////////////////////////////////////////////////////
/// Addition of dyadic numbers
//////////////////////////////////////////////////////////////////////////////////////////////
impl<T: Ord + Clone> AddAssign<Mono<T>> for Dyadic<T> {
    fn add_assign(&mut self, other: Mono<T>) {
        // a / 2^n + b = (a + 2^n * b) / 2^n
        let mono = &other * &self.denom;
        // if not normalized, manually combine terms
        let join = self.numer.extract_if(&|v: &Mono<T>| v.terms == mono.terms && v.bin == mono.bin);
        // Sum multipliers
        let mult = join.into_iter().fold(mono.mult, |acc, x| acc + x.mult);
        if mult != 0 {
            self.numer.insert(Mono { mult, terms: other.terms, bin: other.bin });
        }
    }
}

impl<T: Ord + Clone> AddAssign<Bin<T>> for Dyadic<T> {
    fn add_assign(&mut self, other: Bin<T>) {
        *self += Mono::bin(other);
    }
}

impl<T: Ord + Clone> AddAssign<i32> for Dyadic<T> {
    fn add_assign(&mut self, other: i32) {
        *self += Mono::lit(other);
    }
}

impl<T: Ord + Clone> AddAssign for Dyadic<T> {
    fn add_assign(&mut self, other: Self) {
        // Compute denominator as the LCM of the two denominators
        let lcm = self.denom.lcm(&other.denom);

        // Compute the left multiplicative factor (remainder is 0 by LCM property)
        let (lm, _) = &lcm / &self.denom;
        // Compute the right multiplicative factor
        let (rm, _) = &lcm / &other.denom;

        // Make a reverse map of all mono terms to their multipliers
        let mut hm: Ctx<(&Ctx<T, u8>, Bin<T>), i32> = Ctx::new();

        // Multiply each term by the multiplicative factor and denominator is the lcd
        for l in self.numer.iter() {
            hm.insert_with((&l.terms, &l.bin * &lm), l.mult, &|a, b| a + b);
        }
        for r in other.numer.iter() {
            hm.insert_with((&r.terms, &r.bin * &rm), r.mult, &|a, b| a + b);
        }
        self.numer = hm.into_iter().map(|((t, b), m)| Mono { terms: t.clone(), bin: b, mult: m }).collect();
        self.denom = lcm;
    }
}

impl<T: Ord + Clone> Add for Dyadic<T> {
    type Output = Dyadic<T>;
    fn add(self, other: Self) -> Self::Output {
        let mut res = self.clone();
        res += other;
        res
    }
}

impl<T: Ord + Clone> Add<Mono<T>> for Dyadic<T> {
    type Output = Dyadic<T>;
    fn add(self, other: Mono<T>) -> Self::Output {
        let mut res = self.clone();
        res += other;
        res
    }
}

impl<T: Ord + Clone> Add<Bin<T>> for Dyadic<T> {
    type Output = Dyadic<T>;
    fn add(self, other: Bin<T>) -> Self::Output {
        let mut res = self.clone();
        res += other;
        res
    }
}

impl<T: Ord + Clone> Add for &Dyadic<T> {
    type Output = Dyadic<T>;
    fn add(self, other: Self) -> Self::Output {
        self.clone() + other.clone()
    }
}

impl<T: Ord + Clone> Add<&Mono<T>> for &Dyadic<T> {
    type Output = Dyadic<T>;
    fn add(self, other: &Mono<T>) -> Self::Output {
        self.clone() + other.clone()
    }
}

impl<T: Ord + Clone> Add<&Bin<T>> for &Dyadic<T> {
    type Output = Dyadic<T>;
    fn add(self, other: &Bin<T>) -> Self::Output {
        self.clone() + other.clone()
    }
}

//////////////////////////////////////////////////////////////////////////////////////////////
/// Subtraction of dyadic numbers
//////////////////////////////////////////////////////////////////////////////////////////////
impl<T: Ord + Clone> SubAssign<Mono<T>> for Dyadic<T> {
    fn sub_assign(&mut self, other: Mono<T>) {
        *self += other.neg()
    }
}

impl<T: Ord + Clone> SubAssign<Bin<T>> for Dyadic<T> {
    fn sub_assign(&mut self, other: Bin<T>) {
        *self += Mono::bin(other).neg();
    }
}

impl<T: Ord + Clone> SubAssign<i32> for Dyadic<T> {
    fn sub_assign(&mut self, other: i32) {
        *self += Mono::lit(-other);
    }
}

impl<T: Ord + Clone> SubAssign for Dyadic<T> {
    fn sub_assign(&mut self, other: Self) {
        *self += other.neg();
    }
}

impl<T: Ord + Clone> Sub for Dyadic<T> {
    type Output = Dyadic<T>;
    fn sub(self, other: Self) -> Self::Output {
        let mut res = self.clone();
        res -= other;
        res
    }
}

impl<T: Ord + Clone> Sub<Mono<T>> for Dyadic<T> {
    type Output = Dyadic<T>;
    fn sub(self, other: Mono<T>) -> Self::Output {
        let mut res = self.clone();
        res -= other;
        res
    }
}

impl<T: Ord + Clone> Sub<Bin<T>> for Dyadic<T> {
    type Output = Dyadic<T>;
    fn sub(self, other: Bin<T>) -> Self::Output {
        let mut res = self.clone();
        res -= other;
        res
    }
}

impl<T: Ord + Clone> Sub for &Dyadic<T> {
    type Output = Dyadic<T>;
    fn sub(self, other: Self) -> Self::Output {
        self.clone() - other.clone()
    }
}

impl<T: Ord + Clone> Sub<&Mono<T>> for &Dyadic<T> {
    type Output = Dyadic<T>;
    fn sub(self, other: &Mono<T>) -> Self::Output {
        self.clone() - other.clone()
    }
}

impl<T: Ord + Clone> Sub<&Bin<T>> for &Dyadic<T> {
    type Output = Dyadic<T>;
    fn sub(self, other: &Bin<T>) -> Self::Output {
        self.clone() - other.clone()
    }
}

//////////////////////////////////////////////////////////////////////////////////////////////
/// Multiplication of dyadic numbers
//////////////////////////////////////////////////////////////////////////////////////////////
impl<T: Ord + Clone> MulAssign<Mono<T>> for Dyadic<T> {
    fn mul_assign(&mut self, other: Mono<T>) {
        self.numer.modify(|v| *v *= other.clone());
        self.normalize();
    }
}

impl<T: Ord + Clone> MulAssign<Bin<T>> for Dyadic<T> {
    fn mul_assign(&mut self, other: Bin<T>) {
        // Calculate GCD of binary terms
        let gcd = self.denom.gcd(&other);
        // Multiply numerator by [other / GCD]
        let (mult, _) = other / gcd.clone();
        self.numer.modify(|x| *x *= mult.clone());
        // Denominator is [denom / GCD]
        (self.denom, _) = self.denom.clone() / gcd;
    }
}

impl<T: Ord + Clone> MulAssign<i32> for Dyadic<T> {
    fn mul_assign(&mut self, other: i32) {
        let mono= Mono::<T>::log2(other);
        *self *= mono;
    }
}

impl<T: Ord + Clone> MulAssign for Dyadic<T> {
    fn mul_assign(&mut self, other: Self) {
        let mut terms = Set::new();
        for l in self.numer.iter() {
            for r in other.numer.iter() {
                terms.insert_with(l * r, |v| v.double());
            }
        }
        self.numer = terms;
        self.denom *= other.denom;
    }
}

impl<T: Ord + Clone> Mul for Dyadic<T> {
    type Output = Dyadic<T>;
    fn mul(self, other: Self) -> Self::Output {
        let mut res = self.clone();
        res *= other;
        res
    }
}

impl<T: Ord + Clone> Mul<Mono<T>> for Dyadic<T> {
    type Output = Dyadic<T>;
    fn mul(self, other: Mono<T>) -> Self::Output {
        let mut res = self.clone();
        res *= other;
        res
    }
}

impl<T: Ord + Clone> Mul<Bin<T>> for Dyadic<T> {
    type Output = Dyadic<T>;
    fn mul(self, other: Bin<T>) -> Self::Output {
        let mut res = self.clone();
        res *= other;
        res
    }
}

impl<T: Ord + Clone> Mul for &Dyadic<T> {
    type Output = Dyadic<T>;
    fn mul(self, other: Self) -> Self::Output {
        self.clone() * other.clone()
    }
}

impl<T: Ord + Clone> Mul<&Mono<T>> for &Dyadic<T> {
    type Output = Dyadic<T>;
    fn mul(self, other: &Mono<T>) -> Self::Output {
        self.clone() * other.clone()
    }
}

impl<T: Ord + Clone> Mul<&Bin<T>> for &Dyadic<T> {
    type Output = Dyadic<T>;
    fn mul(self, other: &Bin<T>) -> Self::Output {
        self.clone() * other.clone()
    }
}

//////////////////////////////////////////////////////////////////////////////////////////////
/// Division of dyadic numbers by binary powers
//////////////////////////////////////////////////////////////////////////////////////////////
impl<T: Ord> DivAssign<Bin<T>> for Dyadic<T> {
    fn div_assign(&mut self, other: Bin<T>) {
        self.denom *= other;
    }
}

impl<T: Ord + Clone> Div<Bin<T>> for Dyadic<T> {
    type Output = Dyadic<T>;
    fn div(self, other: Bin<T>) -> Self::Output {
        let mut res = self.clone();
        res /= other;
        res
    }
}

impl<T: Ord + Clone> Div<&Bin<T>> for &Dyadic<T> {
    type Output = Dyadic<T>;
    fn div(self, other: &Bin<T>) -> Self::Output {
        self.clone() / other.clone()
    }
}

//////////////////////////////////////////////////////////////////////////////////////////////
/// Pretty printing, display and Arbitrary for Dyadic
//////////////////////////////////////////////////////////////////////////////////////////////
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
        let mut d = Dyadic { numer, denom };
        d.normalize();
        Ok(d)
    }
}
impl<T: Ord + fmt::Display + Clone> Specializable<T, u8> for Dyadic<T> {
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
        // 1. Normalize numerator and remove zero terms
        self.numer.modify(|x| x.normalize());
        self.numer.retain(|x| x.mult > 0);

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

        // Make a reverse map of all mono terms to their multipliers
        let mut hm: Ctx<(&Ctx<T, u8>, Bin<T>), i32> = Ctx::new();

        // Cancel out terms
        for l in self.numer.iter() {
            hm.insert_with((&l.terms, l.bin.clone()), l.mult, &|a, b| a + b);
        }
        self.numer = hm.into_iter().map(|((t, b), m)| Mono { terms: t.clone(), bin: b, mult: m }).collect();
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
fn test_dyadic_add_comm_unit() {
    let a = Dyadic::lit(2)*Dyadic::var("X")*Dyadic::bin(Bin::lit(2)* Bin::var("y"));
    let b = Dyadic::lit(3)*Dyadic::var("Y")*Dyadic::bin(Bin::lit(0)* Bin::var("x"));
    assert_eqn!(&a + &b, &b + &a);
}

#[test]
fn test_dyadic_add_lcm_unit() {
    let a = Dyadic::var("a").div_bin(&Bin::var("n"));
    let b = Dyadic::var("b").div_bin(&Bin::var("m"));
    assert_eqn!(&a + &b,
        Dyadic { numer: Set::from([
            Mono { mult: 1, terms: Ctx::from([("a", 1)]), bin: Bin::var("m") },
            Mono { mult: 1, terms: Ctx::from([("b", 1)]), bin: Bin::var("n") }
        ]), denom: Bin::var("n") * Bin::var("m")
    });
}

#[test]
fn test_dyadic_sub_lcm_unit() {
    let a = Dyadic::var("a").div_bin(&Bin::var("n"));
    let b = Dyadic::var("b").div_bin(&Bin::var("m"));
    assert_eqn!(&a - &b,
        Dyadic { numer: Set::from([
            Mono { mult: 1, terms: Ctx::from([("a", 1)]), bin: Bin::var("m") },
            Mono { mult: -1, terms: Ctx::from([("b", 1)]), bin: Bin::var("n") }
        ]), denom: Bin::var("n") * Bin::var("m")
    });
}

#[test]
fn test_dyadic_sub_cancel_unit() {
    let a = Dyadic {
        numer: Set::from([Mono::<Id>::term(Id::from('a'), 7)]),
        denom: Bin::default(),
    };
    assert_eqn!(&a - &a, Dyadic::<Id>::unit_add());
}

#[test]
fn test_dyadic_mul_unit() {
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
fn test_dyadic_specialize_unit() {
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
fn test_dyadic_normalize_unit() {
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
