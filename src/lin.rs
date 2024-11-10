use std::fmt;
use std::ops::{Add, AddAssign, Sub, SubAssign, MulAssign, Mul};
use crate::context::{Set, Ctx};
use crate::traits::Eval;
use pretty::{DocAllocator, DocBuilder, BoxAllocator, Pretty};
use std::iter::Sum;

/// Represents a type-level linear expression, with the invariant that multipliers are non-zero
/// ex: 2a + 3b - 4c - 5
#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub struct Lin<Id> { pub terms: Ctx<Id, i32>, pub c: i32 }

/// Specialize a linear term by substituting a variable
impl<Id: Ord + Clone + fmt::Display> Eval<Id, i32> for Lin<Id> {
    type ReflectError = ();
    fn specialize(&mut self, id: &Id, val: i32) {
        if let Some(v) = self.terms.remove(id) {
            self.c += v * val;
        }
    }
    fn reflect(&self) -> Result<i32, ()> {
        Ok(self.c)
    }
    fn free_vars(&self) -> Set<&Id> {
        self.terms.keys()
    }
}

/// Identity element of addition is 0
impl<Id : Ord> Default for Lin<Id> {
    fn default() -> Self {
        Lin::lit(0)
    }
}

impl<Id: Ord> Sum for Lin<Id> {
    fn sum<I>(terms: I) -> Self
    where
        I: IntoIterator<Item = Self>,
    {
        let mut sum = Lin::lit(0);
        for term in terms {
            sum += term;
        }
        sum
    }
}

impl <Id: Ord> Lin<Id> {
    pub fn lit(a: i32) -> Self {
        Lin { terms: Ctx::new(), c: a }
    }
    pub fn var(v: Id) -> Self {
        Lin { terms: Ctx::from([(v, 1)]), c: 0 }
    }
    pub fn zero() -> Self {
        Lin::lit(0)
    }
    pub fn term(v: Id, a: i32) -> Self {
        if a == 0 {
            Lin::lit(0)
        } else {
            Lin { terms: Ctx::from([(v, a)]), c: 0 }
        }
    }
    pub fn neg(&self) -> Self where Id: Clone {
        let mut terms = self.terms.clone();
        for (_, v) in terms.iter_mut() {
            *v = -*v;
        }
        Lin { terms, c: -self.c }
    }
    /// Define a partial order for linear expressions with positive variable assignments
    /// true:  2*a + b <= 3*a + b + c
    ///        {} <= a
    /// false: 4*a <= 3*a + b + c
    ///        b <= c
    pub fn leq(&self, other: &Self) -> bool where Id: Clone {
        (other - self).terms.values().all(|v| *v >= 0) && (other - self).c >= 0
    }

    /// Least-upper bound of two linear terms
    pub fn lub(self, other: Self) -> Self where Id: Clone {
        Lin {
            terms: self.terms.union_with(other.terms.clone(), &|a, b| std::cmp::max(a, b)),
            c: std::cmp::max(self.c, other.c)
        }
    }

    /// Greatest-lower bound of two linear terms
    pub fn glb(self, other: Self) -> Self where Id: Clone {
        Lin {
            terms: self.terms.intersection_with(other.terms.clone(), &|a, b| std::cmp::min(a, b)),
            c: std::cmp::min(self.c, other.c)
        }
    }
}

impl<Id: Ord> AddAssign for Lin<Id> {
    /// Add two linear terms
    fn add_assign(&mut self, other: Self) {
        self.terms.append_with(other.terms.into_iter(), &|a, b| a + b);
        self.terms.retain(|_, v| *v != 0);
        self.c += other.c;
    }
}

impl<Id: Ord + Clone> Add for Lin<Id> {
    type Output = Lin<Id>;
    /// Add two linear terms
    fn add(self, other: Self) -> Self::Output {
        let mut c = self.clone();
        c += other;
        c
    }
}

impl<Id: Ord + Clone> Add for &Lin<Id> {
    type Output = Lin<Id>;
    /// Add two linear terms
    fn add(self, other: Self) -> Self::Output {
        let mut c = self.clone();
        c += other.clone();
        c
    }
}

impl<Id: Ord> SubAssign for Lin<Id> {
    /// Subtract two linear terms
    fn sub_assign(&mut self, other: Self) {
        self.terms.append_with(other.terms.into_iter().map(|(k, v)| (k, -v)), &|a, b| a + b);
        self.terms.retain(|_, v| *v != 0);
        self.c -= other.c;
    }
}

impl<Id: Ord + Clone> Sub for Lin<Id> {
    type Output = Lin<Id>;
    /// Subtract two linear terms
    fn sub(self, other: Self) -> Self::Output {
        let mut c = self.clone();
        c -= other.clone();
        c
    }
}

impl<Id: Ord + Clone> Sub for &Lin<Id> {
    type Output = Lin<Id>;
    /// Add two linear terms
    fn sub(self, other: Self) -> Self::Output {
        let mut c = self.clone();
        c -= other.clone();
        c
    }
}

impl <Id: Ord> MulAssign<i32> for Lin<Id> {
    /// Multiply a linear term by a scalar
    fn mul_assign(&mut self, rhs: i32) {
        for (_, v) in self.terms.iter_mut() {
            *v *= rhs;
        }
        self.c *= rhs;
    }
}

impl<Id: Ord + Clone> Mul<i32> for Lin<Id> {
    type Output = Lin<Id>;
    /// Subtract two linear terms
    fn mul(self, other: i32) -> Self::Output {
        let mut c = self.clone();
        c *= other;
        c
    }
}

impl<Id: Ord + Clone> Mul<i32> for &Lin<Id> {
    type Output = Lin<Id>;
    /// Subtract two linear terms
    fn mul(self, other: i32) -> Self::Output {
        self.clone() * other
    }
}

impl<Id: Ord> From<i32> for Lin<Id> {
    fn from(a: i32) -> Self {
        Lin::lit(a)
    }
}

////////////////////////////////////////////////////////////////////////////////////////
/// Pretty Formatting, Display & Arbitrary for Lin
////////////////////////////////////////////////////////////////////////////////////////
impl<'a, D, A, Id> Pretty<'a, D, A> for Lin<Id>
where
    D: DocAllocator<'a, A>,
    D::Doc: Clone,
    A: 'a + Clone,
    Id: Pretty<'a, D, A> + Clone + Ord
{
    fn pretty(self, allocator: &'a D) -> DocBuilder<'a, D, A> {
        if self.terms.is_empty() {
            allocator.text(format!("{}", self.c))
        } else {
            allocator.text(format!("{}", self.c))
                .append(allocator.concat(
                        self.terms.into_iter()
                            .map(|(k, v)|
                                if v == 0 {
                                    allocator.nil()
                                } else if v == 1 {
                                    allocator.text("+").append(k.pretty(allocator))
                                } else if v < 0 {
                                    allocator.text("-")
                                        .append(allocator.text(format!("{}", -v)).append(k.pretty(allocator)))
                                } else {
                                    allocator.text("+")
                                        .append(allocator.text(format!("{}", v)).append(k.pretty(allocator)))
                                })
                            .collect::<Vec<_>>()
                        )
                )
        }
    }
}

/// Display instance calls the pretty printer
impl<'a, Id> fmt::Display for Lin<Id>
where
    Id: Pretty<'a, BoxAllocator, ()> + Clone + Ord
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        <Lin<Id> as Pretty<'_, BoxAllocator, ()>>::pretty(self.clone(), &BoxAllocator)
            .1
            .render_fmt(100, f)
    }
}

/// Arbitrary instance for Lin
#[cfg(test)] use arbitrary::{Arbitrary, Unstructured};
#[cfg(test)]
impl<'a, Id: Ord + Clone + Arbitrary<'a>> Arbitrary<'a> for Lin<Id> {
    fn arbitrary(u: &mut Unstructured<'a>) -> arbitrary::Result<Self> {
        let n = u.int_in_range(0..=10)?;
        let mut ctx = Ctx::new();
        for _ in 0..=n {
            ctx.insert(Id::arbitrary(u)?, u.int_in_range(1..=9)?);
        }
        Ok(Lin { terms: ctx, c: u.int_in_range(0..=9)? })
    }
}

////////////////////////////////////////////////////////////////////////////////////////
/// Unit Tests for Lin
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
    assert_eq!(Lin::from(3) + Lin::from(2) + Lin::var("x") - (Lin::from(2) + Lin::var("y") + Lin::var("x")),
        Lin::from(3) - Lin::var("y"));
}

#[test]
fn test_leq_lin() {
    assert_eq!(
        Lin::leq(
            &(Lin::lit(2) + Lin::var("a")),
            &(Lin::term("a", 2) + Lin::var("b") + Lin::lit(4))
        ),
        true
    );
    assert_eq!(
        Lin::leq(
            &(Lin::lit(2) + Lin::var("c")),
            &(Lin::term("a", 2) + Lin::var("b") + Lin::lit(4))
        ),
        false
    );
    assert_eq!(
        Lin::leq(
            &(Lin::term("a", 3) + Lin::var("b")),
            &(Lin::term("a", 2) + Lin::var("b") + Lin::lit(4))
        ),
        false
    );
}

#[test]
fn test_lin_eval() {
    let l = Lin::var("x") + Lin::var("y") + Lin::lit(1);

    let mut l1 = l.clone();
    l1.specialize(&"x", 2);
    assert_eq!(l1, Lin::var("y") + Lin::lit(3));

    let mut l2 = l.clone();
    l2.specialize(&"y", 2);
    assert_eq!(l2, Lin::var("x") + Lin::lit(3));

    let mut l3 = l.clone();
    l3.specialize(&"z", 2);
    assert_eq!(l, l3);

    assert_eq!(l.clone().eval(Ctx::from([("x", 2), ("y", 3)])), Ok(6));
}

#[cfg(test)] use arbtest::arbtest;
#[cfg(test)] use crate::id::Id;

#[test]
fn test_lin_add_prop() {
    // Associativity
    arbtest(|u| {
        let a = u.arbitrary::<Lin<Id>>()?;
        let b = u.arbitrary::<Lin<Id>>()?;
        let c = u.arbitrary::<Lin<Id>>()?;
        assert_eq!(&a + &(&b + &c), &(&a + &b) + &c);
        Ok(())
    });

    // Commutativity
    arbtest(|u| {
        let a = u.arbitrary::<Lin<Id>>()?;
        let b = u.arbitrary::<Lin<Id>>()?;
        assert_eq!(&a + &b, &b + &a);
        Ok(())
    });

    // Unit
    arbtest(|u| {
        let a = u.arbitrary::<Lin<Id>>()?;
        assert_eq!(&a + &Lin::default(), a);
        assert_eq!(&Lin::default() + &a, a);
        Ok(())
    });
}

#[test]
fn test_lin_sub_prop() {
    // Cancelativity
    arbtest(|u| {
        let a = u.arbitrary::<Lin<Id>>()?;
        assert_eq!(&a - &a, Lin::default());
        Ok(())
    });
    // Subtraction is the inverse of addition
    arbtest(|u| {
        let a = u.arbitrary::<Lin<Id>>()?;
        let b = u.arbitrary::<Lin<Id>>()?;
        assert_eq!(&a + &b - a, b);
        Ok(())
    });
    // Unit with subtraction
    arbtest(|u| {
        let a = u.arbitrary::<Lin<Id>>()?;
        assert_eq!(&a - &Lin::default(), a.clone());
        assert_eq!(&Lin::default() - &a, a.neg());
        Ok(())
    });
}

#[test]
fn test_lin_leq_prop() {
    // Reflexivity
    arbtest(|u| {
        let a = u.arbitrary::<Lin<Id>>()?;
        assert!(a.leq(&a));
        Ok(())
    });
    // Leq and addition
    arbtest(|u| {
        let a = u.arbitrary::<Lin<Id>>()?;
        let b = u.arbitrary::<Lin<Id>>()?;
        // a <= a + b
        assert!(a.leq(&(&a + &b)));
        assert!(b.leq(&(&a + &b)));
        Ok(())
    });
}
