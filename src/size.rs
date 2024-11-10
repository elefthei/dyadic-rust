use crate::context::Set;
use pretty::{Pretty, DocBuilder, DocAllocator, BoxAllocator};
use crate::group::BinGroup;
use std::fmt;
use itertools::Itertools;

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub enum Size<Id> {
    // Rules for Max:
    // - Idempotence: max(a,a) = a
    // - Absorption: max(a,min(a,b)) = a
    // - Distributivity over addition: max(a+c, b+c) = max(a,b) + c
    Max(Set<Size<Id>>),

    // Rules for Min:
    // - Idempotence: min(a,a) = a
    // - Absorption: min(a,max(a,b)) = a
    // - Distributivity over addition: min(a+c, b+c) = min(a,b) + c
    Min(Set<Size<Id>>),

    Term(BinGroup<Id>),
}

impl<Id: Clone + Ord> Size<Id> {
    pub fn var(id: Id) -> Self {
        Size::Term(BinGroup::var(id))
    }

    pub fn lit(value: i32) -> Self {
        Size::Term(BinGroup::lit(value))
    }

    // Rules for negation:
    // - Distributivity over max: -max(a,b) = min(-a,-b)
    // - Distributivity over min: -min(a,b) = max(-a,-b)
    // - Double negation: -(-a) = a
    // - Negation of constant: -(constant) = -constant
    pub fn neg(form: Size<Id>) -> Self {
        match form {
            Size::Max(terms) => Size::Min(terms.into_iter().map(Size::neg).collect()),
            Size::Min(terms) => Size::Max(terms.into_iter().map(Size::neg).collect()),
            Size::Term(term) => Size::Term(term.neg())
        }
    }

    // Rules for addition:
    // - Identity: a + 0 = a
    // - Commutativity: a + b = b + a
    // - Associativity: (a + b) + c = a + (b + c)
    // - Distributivity of max over addition: max(a,b) + c = max(a+c, b+c)
    // - Distributivity of min over addition: min(a,b) + c = min(a+c, b+c)
    pub fn add<I>(forms: I) -> Self
    where
        I: IntoIterator<Item = Size<Id>>,
    {
        let forms: Vec<_> = forms.into_iter().collect();

        match forms.len() {
            0 => Size::Term(BinGroup::zero()),  // Identity element for addition
            1 => forms.into_iter().next().unwrap(),
            _ => {
                // Convert each form into its constituents for distribution
                let term_sets: Vec<Vec<Size<Id>>> = forms.into_iter()
                    .map(|form| match form {
                        // Distribute addition over max: (max(a,b) + c) = max(a+c, b+c)
                        Size::Max(terms) => terms.into_iter().collect(),
                        // Preserve min terms for later distribution
                        Size::Min(terms) => vec![Size::Min(terms)],
                        term => vec![term],
                    })
                    .collect();

                // Use cartesian product to implement distribution
                let distributed = term_sets.into_iter()
                    .multi_cartesian_product()
                    .map(|terms| {
                        terms.into_iter().fold(Size::Term(BinGroup::zero()), |acc, term| {
                            match (acc, term) {
                                // Combine terms into a single Add
                                (Size::Term(mut t1), Size::Term(t2)) => {
                                    t1 += t2;
                                    Size::Term(t1)
                                },
                                // Distribute addition over min: (min(a,b) + c) = min(a+c, b+c)
                                (Size::Term(l1), Size::Min(terms)) => {
                                    Size::Min(terms.into_iter()
                                        .map(|t| Size::add([Size::Term(l1.clone()), t]))
                                        .collect())
                                },
                                // Distribute addition over max: (max(a,b) + c) = max(a+c, b+c)
                                (Size::Term(l1), Size::Max(terms)) => {
                                    Size::Max(terms.into_iter()
                                        .map(|t| Size::add([Size::Term(l1.clone()), t]))
                                        .collect())
                                },
                                _ => unreachable!("First term should always be Add")
                            }
                        })
                    }).collect();

                Size::Max(distributed)
            }
        }
    }

    // Rules for subtraction:
    // a - b = a + (-b)
    pub fn sub<I>(forms: I) -> Self
    where
        I: IntoIterator<Item = Size<Id>>,
    {
        let mut forms = forms.into_iter();
        if let Some(first) = forms.next() {
            Size::add([first].into_iter().chain(forms.map(Size::neg)))
        } else {
            Size::Term(BinGroup::zero())
        }
    }

    // Rules for max:
    // - Identity: max(a) = a
    // - Idempotence: max(a,a) = a
    // - Absorption with min: max(a,min(a,b)) = a
    // - Associativity: max(max(a,b),c) = max(a,max(b,c))
    // - Commutativity: max(a,b) = max(b,a)
    pub fn max<I>(forms: I) -> Self
    where
        I: IntoIterator<Item = Self>,
    {
        let mut terms: Set<_> = forms.into_iter()
            .flat_map(|form| match form {
                // Flatten nested max (associativity): max(max(a,b),c) = max(a,b,c)
                Size::Max(inner) => inner,
                other => Set::from([other]),
            }).collect();

        let cloned = terms.clone();

        // Absorption: max(a, min(a,b)) = a
        terms.retain(|form| {
            if let Size::Min(min_terms) = form {
                !cloned.iter().any(|other| min_terms.contains(other))
            } else {
                true
            }
        });

        match terms.len() {
            0 => Size::Term(BinGroup::lit(i32::MIN)),  // Identity for max
            1 => terms.into_iter().next().unwrap(),
            _ => Size::Max(terms),  // BTreeSet handles idempotence and commutativity
        }
    }

    // Rules for min:
    // - Identity: min(a) = a
    // - Idempotence: min(a,a) = a
    // - Absorption with max: min(a,max(a,b)) = a
    // - Associativity: min(min(a,b),c) = min(a,min(b,c))
    // - Commutativity: min(a,b) = min(b,a)
    pub fn min<I>(forms: I) -> Self
    where
        I: IntoIterator<Item = Self>,
    {
        let mut terms: Set<_> = forms.into_iter()
            .flat_map(|form| match form {
                // Flatten nested max (associativity): max(max(a,b),c) = max(a,b,c)
                Size::Min(inner) => inner,
                other => Set::from([other]),
            }).collect();

        let cloned = terms.clone();
        // Absorption: min(a, max(a,b)) = a
        terms.retain(|form| {
            if let Size::Max(max_terms) = form {
                !cloned.iter().any(|other| max_terms.contains(other))
            } else {
                true
            }
        });

        match terms.len() {
            0 => Size::Term(BinGroup::lit(i32::MAX)),  // Identity for max
            1 => terms.into_iter().next().unwrap(),
            _ => Size::Min(terms),  // BTreeSet handles idempotence and commutativity
        }
    }

    pub fn zero() -> Self {
        Size::Term(BinGroup::zero())
    }
}

//////////////////////////////////////////////////////////////////////////////////////////////
/// Pretty printing, display and Arbitrary for BinGroup
//////////////////////////////////////////////////////////////////////////////////////////////
impl<'a, D, A, T> Pretty<'a, D, A> for Size<T>
where
    D: DocAllocator<'a, A>,
    T: Pretty<'a, D, A> + Clone + Ord,
    D::Doc: Clone,
    A: 'a + Clone,
{
    fn pretty(self, allocator: &'a D) -> DocBuilder<'a, D, A> {
        match self {
            Size::Max(terms) =>
                allocator.concat([
                    allocator.text("max {"),
                    allocator.intersperse(terms.into_iter().map(|term| term.pretty(allocator)),
                        allocator.text(", ")),
                    allocator.text("}")
                ]),
            Size::Min(terms) =>
                allocator.concat([
                    allocator.text("min {"),
                    allocator.intersperse(terms.into_iter().map(|term| term.pretty(allocator)),
                        allocator.text(", ")),
                    allocator.text("}")
                ]),
            Size::Term(term) => term.pretty(allocator),
        }
    }
}

/// Display instance calls the pretty printer
impl<'a, T> fmt::Display for Size<T>
where
    T: Pretty<'a, BoxAllocator, ()> + Clone + Ord,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        <Size<T> as Pretty<'_, BoxAllocator, ()>>::pretty(self.clone(), &BoxAllocator)
            .1
            .render_fmt(100, f)
    }
}

#[test]
fn test_size_idempotence() {
    let a = Size::var("a");
    // max(a,a) = a
    let max_aa = Size::max([a.clone(), a.clone()]);
    assert_eq!(max_aa, a);
}

#[test]
fn test_size_absorption() {
    let a = Size::var("a");
    let b = Size::var("b");
    // min(a, max(a,b)) = a
    let min_a_maxab =
        Size::min([a.clone(), Size::max([a.clone(), b.clone()])]);
    assert_eq!(min_a_maxab, a);

    // max(a,min(a,b)) = a
    let max_a_minab =
        Size::max([a.clone(), Size::min([a.clone(), b.clone()])]);
    assert_eq!(max_a_minab, a);
}

#[test]
fn test_size_distributivity() {
    let a = Size::var("a");
    let b = Size::var("b");
    let c = Size::var("c");
    // max(a,b) + c = max(a+c, b+c)
    let max_ab = Size::max([a.clone(), b.clone()]);
    let max_ab_plus_c = Size::add([max_ab, c.clone()]);
    let a_plus_c = Size::add([a, c.clone()]);
    let b_plus_c = Size::add([b, c]);
    let expected = Size::max([a_plus_c, b_plus_c]);
    assert_eq!(max_ab_plus_c, expected);
}

#[test]
fn test_size_add_distribution() {
    let a = Size::var("a");
    let b = Size::var("b");
    let c = Size::var("c");
    let d = Size::var("d");

    // (max(a,b) + max(c,d)) = max(a+c, a+d, b+c, b+d)
    let max_ab = Size::max([a.clone(), b.clone()]);
    let max_cd = Size::max([c.clone(), d.clone()]);
    let sum = Size::add([max_ab, max_cd]);

    let ac = Size::add([a.clone(), c.clone()]);
    let ad = Size::add([a, d.clone()]);
    let bc = Size::add([b.clone(), c]);
    let bd = Size::add([b, d]);

    let expected = Size::max([ac, ad, bc, bd]);
    assert_eq!(sum, expected);
}

#[test]
fn test_size_negation() {
    let a = Size::var("a");
    let b = Size::var("b");
    // -max(a,b) = min(-a,-b)
    let max_ab = Size::max([a.clone(), b.clone()]);
    let neg_max = Size::neg(max_ab);
    let neg_a = Size::neg(a);
    let neg_b = Size::neg(b);
    let expected = Size::min([neg_a, neg_b]);
    assert_eq!(neg_max, expected);
}

#[test]
fn test_size_n_ary_add() {
    let a = Size::var("a");
    let b = Size::var("b");
    let c = Size::var("c");
    let sum = Size::add([a.clone(), b.clone(), c.clone()]);

    // Associativity: (a + b) + c = a + (b + c)
    let sum2 = Size::add([Size::add([a.clone(), b.clone()]), c.clone()]);
    assert_eq!(sum, sum2);

    // Commutativity: a + b + c = c + b + a
    let sum3 = Size::add([c, b, a]);
    assert_eq!(sum, sum3);
}

#[test]
fn test_size_id() {
    let a = Size::var("x");
    let b = Size::var("y");
    let max_ab = Size::max([a.clone(), b.clone()]);
    let neg_max = Size::neg(max_ab);
    let neg_a = Size::neg(a);
    let neg_b = Size::neg(b);
    let expected = Size::min([neg_a, neg_b]);
    assert_eq!(neg_max, expected);
}
