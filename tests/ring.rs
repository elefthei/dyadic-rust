//! Property tests: symbolic dyadic rationals form a commutative ring with unity, and the other
//! operations (specialization, doubling, halving, scaling by powers of two) respect it.
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
//! Coefficients and denominator exponents are kept within bounds so that products of
//! three operands stay within the `i32` coefficients and `u8` exponents of the representation;
//! the laws are claimed for exact arithmetic, not for overflowing machine arithmetic.

use arbtest::arbitrary::{Result, Unstructured};
use arbtest::arbtest;
use dyadic_rationals::id::Id;
use dyadic_rationals::{Bin, Dyadic, Normalizable, Specializable};

const COEFF_BOUND: i32 = 64;
const DENOM_BOUND: u8 = 8;
const SPEC_BOUND: u8 = 8;
const SPEC_VARS: [char; 4] = ['x', 'y', 'm', 'n'];

////////////////////////////////////////////////////////////////////////////////////////
// Operands
////////////////////////////////////////////////////////////////////////////////////////

/// Generates `(c + k * v * [2^m]) / (2^d * [2^n])` within the non-overflowing domain.
fn dyadic(u: &mut Unstructured<'_>) -> Result<Dyadic<Id>> {
    let c = u.int_in_range(-COEFF_BOUND..=COEFF_BOUND)?;
    let k = u.int_in_range(-COEFF_BOUND..=COEFF_BOUND)?;
    let use_y: bool = u.arbitrary()?;
    let sym_numer: bool = u.arbitrary()?;
    let d = u.int_in_range(0..=DENOM_BOUND)?;
    let sym_denom: bool = u.arbitrary()?;
    Ok(operand(c, k, use_y, sym_numer, d, sym_denom))
}

/// Builds `(c + k * v * [2^m]) / (2^d * [2^n])`.
fn operand(c: i32, k: i32, use_y: bool, sym_numer: bool, d: u8, sym_denom: bool) -> Dyadic<Id> {
    let mut term = Dyadic::lit(k) * Dyadic::var(Id::from(if use_y { 'y' } else { 'x' }));
    if sym_numer {
        term = term * Dyadic::bin(Bin::var(Id::from('m')));
    }
    (Dyadic::lit(c) + term).div_bin(&power(d, sym_denom))
}

/// Builds `2^d * [2^n]`.
fn power(d: u8, sym: bool) -> Bin<Id> {
    let mut b = Bin::lit(d);
    if sym {
        b = b * Bin::var(Id::from('n'));
    }
    b
}

fn assert_ring_eq(left: &Dyadic<Id>, right: &Dyadic<Id>, law: &str) {
    assert!(left.eqn(right), "{law} violated:\n  left:  {left}\n  right: {right}");
}

////////////////////////////////////////////////////////////////////////////////////////
// Laws
////////////////////////////////////////////////////////////////////////////////////////

fn check_add_assoc(a: &Dyadic<Id>, b: &Dyadic<Id>, c: &Dyadic<Id>) {
    assert_ring_eq(&(a + &(b + c)), &(&(a + b) + c), "a + (b + c) = (a + b) + c");
}

fn check_add_comm(a: &Dyadic<Id>, b: &Dyadic<Id>) {
    assert_ring_eq(&(a + b), &(b + a), "a + b = b + a");
}

fn check_zero_identity(a: &Dyadic<Id>) {
    for zero in [Dyadic::unit_add(), Dyadic::lit(0)] {
        assert_ring_eq(&(a + &zero), a, "a + 0 = a");
        assert_ring_eq(&(&zero + a), a, "0 + a = a");
    }
}

fn check_negation(a: &Dyadic<Id>, b: &Dyadic<Id>) {
    let zero = Dyadic::unit_add();
    assert_ring_eq(&(a + &a.clone().neg()), &zero, "a + (-a) = 0");
    assert_ring_eq(&(a - a), &zero, "a - a = 0");
    assert_ring_eq(&(a - b), &(a + &b.clone().neg()), "a - b = a + (-b)");
}

fn check_mul_assoc(a: &Dyadic<Id>, b: &Dyadic<Id>, c: &Dyadic<Id>) {
    assert_ring_eq(&(a * &(b * c)), &(&(a * b) * c), "a * (b * c) = (a * b) * c");
}

fn check_mul_comm(a: &Dyadic<Id>, b: &Dyadic<Id>) {
    assert_ring_eq(&(a * b), &(b * a), "a * b = b * a");
}

fn check_one_identity(a: &Dyadic<Id>) {
    for one in [Dyadic::unit_mul(), Dyadic::lit(1)] {
        assert_ring_eq(&(a * &one), a, "a * 1 = a");
        assert_ring_eq(&(&one * a), a, "1 * a = a");
    }
}

fn check_distributivity(a: &Dyadic<Id>, b: &Dyadic<Id>, c: &Dyadic<Id>) {
    assert_ring_eq(&(a * &(b + c)), &(&(a * b) + &(a * c)), "a * (b + c) = a*b + a*c");
    assert_ring_eq(&(&(a + b) * c), &(&(a * c) + &(b * c)), "(a + b) * c = a*c + b*c");
}

/// The integers embed as a subring, and distinct integers stay distinct.
/// Rules out degenerate models (e.g. an equality that identifies everything) that would
/// satisfy every axiom above vacuously.
fn check_integer_embedding(a: i32, b: i32) {
    let (da, db) = (Dyadic::<Id>::lit(a), Dyadic::<Id>::lit(b));
    assert_ring_eq(&(&da + &db), &Dyadic::lit(a + b), "lit(a) + lit(b) = lit(a + b)");
    assert_ring_eq(&(&da * &db), &Dyadic::lit(a * b), "lit(a) * lit(b) = lit(a * b)");
    assert_ring_eq(&da.clone().neg(), &Dyadic::lit(-a), "-lit(a) = lit(-a)");
    assert_eq!(da.eqn(&db), a == b, "lit({a}) = lit({b}) must hold exactly when {a} = {b}");
}

/// Every `i32` literal is already in normal form, is fixed by `* 1` and double negation, and
/// distinct literals stay distinct. Unlike the laws above, this covers the full `i32` range,
/// since none of these operations can leave it.
fn check_integer_literals(a: i32, b: i32) {
    let (da, db) = (Dyadic::<Id>::lit(a), Dyadic::<Id>::lit(b));
    assert!(da.is_normal(), "lit({a}) is not in normal form: {da}");
    assert_ring_eq(&(&da * &Dyadic::lit(1)), &da, "lit(a) * 1 = lit(a)");
    assert_ring_eq(&da.clone().neg().neg(), &da, "-(-lit(a)) = lit(a)");
    assert_eq!(da.eqn(&db), a == b, "lit({a}) = lit({b}) must hold exactly when {a} = {b}");
}

/// Substituting `v := val` is a ring homomorphism and respects equality modulo normalization.
fn check_specialization(a: &Dyadic<Id>, b: &Dyadic<Id>, v: char, val: u8) {
    let id = Id::from(v);
    let spec = |e: &Dyadic<Id>| {
        let mut e = e.clone();
        e.specialize(&id, val);
        e
    };
    let mut normal = a.clone();
    normal.normalize();
    assert_ring_eq(&spec(&normal), &spec(a), "normalize(a)[v := val] = a[v := val]");
    assert_ring_eq(&spec(&(a + b)), &(&spec(a) + &spec(b)), "(a + b)[v := val] = a[v := val] + b[v := val]");
    assert_ring_eq(&spec(&(a * b)), &(&spec(a) * &spec(b)), "(a * b)[v := val] = a[v := val] * b[v := val]");
}

/// `double` and `half` multiply and divide by two, and are mutually inverse.
fn check_doubling(a: &Dyadic<Id>) {
    assert_ring_eq(&a.clone().double(), &(a + a), "double(a) = a + a");
    assert_ring_eq(&a.clone().double().half(), a, "half(double(a)) = a");
    assert_ring_eq(&a.clone().half().double(), a, "double(half(a)) = a");
}

/// Multiplying and dividing by a power of two agree with the dyadic `Dyadic::bin(b)`.
fn check_bin_scaling(a: &Dyadic<Id>, b: &Bin<Id>) {
    let db = Dyadic::bin(b.clone());
    assert_ring_eq(&(a.clone() * b.clone()), &(a * &db), "a * b = a * bin(b)");
    assert_ring_eq(&(&a.div_bin(b) * &db), a, "(a / b) * bin(b) = a");
    assert_ring_eq(&(a / b), &a.div_bin(b), "a / b = div_bin(a, b)");
}

/// Adding and subtracting a power of two agree with the dyadic `Dyadic::bin(b)`.
fn check_bin_addition(a: &Dyadic<Id>, b: &Bin<Id>) {
    let db = Dyadic::bin(b.clone());
    assert_ring_eq(&(a.clone() + b.clone()), &(a + &db), "a + b = a + bin(b)");
    assert_ring_eq(&(a.clone() - b.clone()), &(a - &db), "a - b = a - bin(b)");
}

/// Normalization is idempotent and preserves equality modulo normalization.
fn check_normal_form(a: &Dyadic<Id>) {
    let mut normal = a.clone();
    normal.normalize();
    assert!(normal.is_normal(), "normalize is not idempotent on {a}: {normal}");
    assert_ring_eq(&normal, a, "normalize(a) = a");
}

////////////////////////////////////////////////////////////////////////////////////////
// arbtest property tests
////////////////////////////////////////////////////////////////////////////////////////

#[test]
fn addition_is_associative() {
    arbtest(|u| {
        check_add_assoc(&dyadic(u)?, &dyadic(u)?, &dyadic(u)?);
        Ok(())
    });
}

#[test]
fn addition_is_commutative() {
    arbtest(|u| {
        check_add_comm(&dyadic(u)?, &dyadic(u)?);
        Ok(())
    });
}

#[test]
fn zero_is_additive_identity() {
    arbtest(|u| {
        check_zero_identity(&dyadic(u)?);
        Ok(())
    });
}

#[test]
fn negation_is_additive_inverse() {
    arbtest(|u| {
        check_negation(&dyadic(u)?, &dyadic(u)?);
        Ok(())
    });
}

#[test]
fn multiplication_is_associative() {
    arbtest(|u| {
        check_mul_assoc(&dyadic(u)?, &dyadic(u)?, &dyadic(u)?);
        Ok(())
    });
}

#[test]
fn multiplication_is_commutative() {
    arbtest(|u| {
        check_mul_comm(&dyadic(u)?, &dyadic(u)?);
        Ok(())
    });
}

#[test]
fn one_is_multiplicative_identity() {
    arbtest(|u| {
        check_one_identity(&dyadic(u)?);
        Ok(())
    });
}

#[test]
fn multiplication_distributes_over_addition() {
    arbtest(|u| {
        check_distributivity(&dyadic(u)?, &dyadic(u)?, &dyadic(u)?);
        Ok(())
    });
}

#[test]
fn integers_embed_faithfully() {
    arbtest(|u| {
        let a = u.int_in_range(-COEFF_BOUND..=COEFF_BOUND)?;
        let b = u.int_in_range(-COEFF_BOUND..=COEFF_BOUND)?;
        check_integer_embedding(a, b);
        Ok(())
    });
}

#[test]
fn integer_literals_are_canonical() {
    arbtest(|u| {
        check_integer_literals(u.arbitrary()?, u.arbitrary()?);
        Ok(())
    });
}

#[test]
fn specialization_is_a_homomorphism() {
    arbtest(|u| {
        let (a, b) = (dyadic(u)?, dyadic(u)?);
        let v = *u.choose(&SPEC_VARS)?;
        let val = u.int_in_range(0..=SPEC_BOUND)?;
        check_specialization(&a, &b, v, val);
        Ok(())
    });
}

#[test]
fn doubling_and_halving_are_inverse() {
    arbtest(|u| {
        check_doubling(&dyadic(u)?);
        Ok(())
    });
}

#[test]
fn scaling_by_powers_of_two() {
    arbtest(|u| {
        let a = dyadic(u)?;
        let b = power(u.int_in_range(0..=DENOM_BOUND)?, u.arbitrary()?);
        check_bin_scaling(&a, &b);
        Ok(())
    });
}

#[test]
fn adding_powers_of_two() {
    arbtest(|u| {
        let a = dyadic(u)?;
        let b = power(u.int_in_range(0..=DENOM_BOUND)?, u.arbitrary()?);
        check_bin_addition(&a, &b);
        Ok(())
    });
}

#[test]
fn normalization_is_idempotent() {
    arbtest(|u| {
        check_normal_form(&dyadic(u)?);
        Ok(())
    });
}
