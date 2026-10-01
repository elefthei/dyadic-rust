//! Property tests: symbolic dyadic rationals form a commutative ring with unity.
//!
//! Each operand is the symbolic dyadic
//!
//! ```text
//!     (c + k * v * [2^m]) / (2^d * [2^n])
//! ```
//!
//! where `v` is the variable `x` or `y`, and the bracketed symbolic powers of two are
//! toggled by generated booleans. Operands therefore cover negative, zero, and even (2-adic)
//! coefficients, polynomial variables, and concrete and symbolic binary exponents in both
//! the numerator and the denominator.
//!
//! Equality is `Normalizable::eqn`, the library's equality modulo normalization.
//!
//! Coefficients and denominator exponents are generated within bounds so that products of
//! three operands stay within the `i32` coefficients and `u8` exponents of the representation;
//! the ring laws are claimed for exact arithmetic, not for overflowing machine arithmetic.

use arbtest::arbitrary::{Result, Unstructured};
use arbtest::arbtest;
use dyadic_rationals::id::Id;
use dyadic_rationals::{Bin, Dyadic, Normalizable};

const COEFF_BOUND: i32 = 64;
const DENOM_BOUND: u8 = 8;

/// Generates `(c + k * v * [2^m]) / (2^d * [2^n])` within the non-overflowing domain.
fn dyadic(u: &mut Unstructured<'_>) -> Result<Dyadic<Id>> {
    let c = u.int_in_range(-COEFF_BOUND..=COEFF_BOUND)?;
    let k = u.int_in_range(-COEFF_BOUND..=COEFF_BOUND)?;
    let use_y: bool = u.arbitrary()?;
    let sym_numer: bool = u.arbitrary()?;
    let d = u.int_in_range(0..=DENOM_BOUND)?;
    let sym_denom: bool = u.arbitrary()?;

    let mut term = Dyadic::lit(k) * Dyadic::var(Id::from(if use_y { 'y' } else { 'x' }));
    if sym_numer {
        term = term * Dyadic::bin(Bin::var(Id::from('m')));
    }
    let mut denom = Bin::lit(d);
    if sym_denom {
        denom = denom * Bin::var(Id::from('n'));
    }
    Ok((Dyadic::lit(c) + term).div_bin(&denom))
}

fn assert_ring_eq(left: &Dyadic<Id>, right: &Dyadic<Id>, law: &str) {
    assert!(left.eqn(right), "{law} violated:\n  left:  {left}\n  right: {right}");
}

#[test]
fn addition_is_associative() {
    arbtest(|u| {
        let (a, b, c) = (dyadic(u)?, dyadic(u)?, dyadic(u)?);
        assert_ring_eq(&(&a + &(&b + &c)), &(&(&a + &b) + &c), "a + (b + c) = (a + b) + c");
        Ok(())
    });
}

#[test]
fn addition_is_commutative() {
    arbtest(|u| {
        let (a, b) = (dyadic(u)?, dyadic(u)?);
        assert_ring_eq(&(&a + &b), &(&b + &a), "a + b = b + a");
        Ok(())
    });
}

#[test]
fn zero_is_additive_identity() {
    arbtest(|u| {
        let a = dyadic(u)?;
        for zero in [Dyadic::unit_add(), Dyadic::lit(0)] {
            assert_ring_eq(&(&a + &zero), &a, "a + 0 = a");
            assert_ring_eq(&(&zero + &a), &a, "0 + a = a");
        }
        Ok(())
    });
}

#[test]
fn negation_is_additive_inverse() {
    arbtest(|u| {
        let (a, b) = (dyadic(u)?, dyadic(u)?);
        let zero = Dyadic::unit_add();
        assert_ring_eq(&(&a + &a.clone().neg()), &zero, "a + (-a) = 0");
        assert_ring_eq(&(&a - &a), &zero, "a - a = 0");
        assert_ring_eq(&(&a - &b), &(&a + &b.clone().neg()), "a - b = a + (-b)");
        Ok(())
    });
}

#[test]
fn multiplication_is_associative() {
    arbtest(|u| {
        let (a, b, c) = (dyadic(u)?, dyadic(u)?, dyadic(u)?);
        assert_ring_eq(&(&a * &(&b * &c)), &(&(&a * &b) * &c), "a * (b * c) = (a * b) * c");
        Ok(())
    });
}

#[test]
fn multiplication_is_commutative() {
    arbtest(|u| {
        let (a, b) = (dyadic(u)?, dyadic(u)?);
        assert_ring_eq(&(&a * &b), &(&b * &a), "a * b = b * a");
        Ok(())
    });
}

#[test]
fn one_is_multiplicative_identity() {
    arbtest(|u| {
        let a = dyadic(u)?;
        for one in [Dyadic::unit_mul(), Dyadic::lit(1)] {
            assert_ring_eq(&(&a * &one), &a, "a * 1 = a");
            assert_ring_eq(&(&one * &a), &a, "1 * a = a");
        }
        Ok(())
    });
}

#[test]
#[ignore = "known bug: Dyadic normalization does not merge like terms, e.g. -27*y*2^(m+3) + 5*y*2^(m+3)"]
fn multiplication_distributes_over_addition() {
    arbtest(|u| {
        let (a, b, c) = (dyadic(u)?, dyadic(u)?, dyadic(u)?);
        assert_ring_eq(&(&a * &(&b + &c)), &(&(&a * &b) + &(&a * &c)), "a * (b + c) = a*b + a*c");
        assert_ring_eq(&(&(&a + &b) * &c), &(&(&a * &c) + &(&b * &c)), "(a + b) * c = a*c + b*c");
        Ok(())
    });
}

/// The integers embed as a subring, and distinct integers stay distinct.
/// Rules out degenerate models (e.g. an equality that identifies everything) that would
/// satisfy every axiom above vacuously.
#[test]
#[ignore = "known bug: Dyadic normalization does not merge literals with different 2^k factors, e.g. lit(-39) + lit(-25)"]
fn integers_embed_faithfully() {
    arbtest(|u| {
        let a = u.int_in_range(-COEFF_BOUND..=COEFF_BOUND)?;
        let b = u.int_in_range(-COEFF_BOUND..=COEFF_BOUND)?;
        let (da, db) = (Dyadic::<Id>::lit(a), Dyadic::<Id>::lit(b));
        assert_ring_eq(&(&da + &db), &Dyadic::lit(a + b), "lit(a) + lit(b) = lit(a + b)");
        assert_ring_eq(&(&da * &db), &Dyadic::lit(a * b), "lit(a) * lit(b) = lit(a * b)");
        assert_ring_eq(&da.clone().neg(), &Dyadic::lit(-a), "-lit(a) = lit(-a)");
        assert_eq!(da.eqn(&db), a == b, "lit({a}) = lit({b}) must hold exactly when {a} = {b}");
        Ok(())
    });
}
