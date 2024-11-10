use crate::context::Set;
use pretty::{Pretty, DocBuilder, DocAllocator, BoxAllocator};
use crate::lin::Lin;
use std::fmt;
use itertools::Itertools;

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub enum MinMaxGroup<Id> {
    // Rules for Max:
    // - Idempotence: max(a,a) = a
    // - Absorption: max(a,min(a,b)) = a
    // - Distributivity over addition: max(a+c, b+c) = max(a,b) + c
    Max(Set<MinMaxGroup<Id>>),

    // Rules for Min:
    // - Idempotence: min(a,a) = a
    // - Absorption: min(a,max(a,b)) = a
    // - Distributivity over addition: min(a+c, b+c) = min(a,b) + c
    Min(Set<MinMaxGroup<Id>>),

    Term(Lin<Id>),
}

impl<Id: Clone + Ord> MinMaxGroup<Id> {
    pub fn var(id: Id) -> Self {
        MinMaxGroup::Term(Lin::var(id))
    }

    pub fn lit(value: i32) -> Self {
        MinMaxGroup::Term(Lin::lit(value))
    }

    // Rules for negation:
    // - Distributivity over max: -max(a,b) = min(-a,-b)
    // - Distributivity over min: -min(a,b) = max(-a,-b)
    // - Double negation: -(-a) = a
    // - Negation of constant: -(constant) = -constant
    pub fn neg(form: MinMaxGroup<Id>) -> Self {
        match form {
            MinMaxGroup::Max(terms) => MinMaxGroup::Min(terms.into_iter().map(MinMaxGroup::neg).collect()),
            MinMaxGroup::Min(terms) => MinMaxGroup::Max(terms.into_iter().map(MinMaxGroup::neg).collect()),
            MinMaxGroup::Term(term) => MinMaxGroup::Term(term.neg())
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
        I: IntoIterator<Item = MinMaxGroup<Id>>,
    {
        let forms: Vec<_> = forms.into_iter().collect();

        match forms.len() {
            0 => MinMaxGroup::Term(Lin::zero()),  // Identity element for addition
            1 => forms.into_iter().next().unwrap(),
            _ => {
                // Convert each form into its constituents for distribution
                let term_sets: Vec<Vec<MinMaxGroup<Id>>> = forms.into_iter()
                    .map(|form| match form {
                        // Distribute addition over max: (max(a,b) + c) = max(a+c, b+c)
                        MinMaxGroup::Max(terms) => terms.into_iter().collect(),
                        // Preserve min terms for later distribution
                        MinMaxGroup::Min(terms) => vec![MinMaxGroup::Min(terms)],
                        term => vec![term],
                    })
                    .collect();

                // Use cartesian product to implement distribution
                let distributed = term_sets.into_iter()
                    .multi_cartesian_product()
                    .map(|terms| {
                        terms.into_iter().fold(MinMaxGroup::Term(Lin::zero()), |acc, term| {
                            match (acc, term) {
                                // Combine terms into a single Add
                                (MinMaxGroup::Term(mut t1), MinMaxGroup::Term(t2)) => {
                                    t1 += t2;
                                    MinMaxGroup::Term(t1)
                                },
                                // Distribute addition over min: (min(a,b) + c) = min(a+c, b+c)
                                (MinMaxGroup::Term(l1), MinMaxGroup::Min(terms)) => {
                                    MinMaxGroup::Min(terms.into_iter()
                                        .map(|t| MinMaxGroup::add([MinMaxGroup::Term(l1.clone()), t]))
                                        .collect())
                                },
                                // Distribute addition over max: (max(a,b) + c) = max(a+c, b+c)
                                (MinMaxGroup::Term(l1), MinMaxGroup::Max(terms)) => {
                                    MinMaxGroup::Max(terms.into_iter()
                                        .map(|t| MinMaxGroup::add([MinMaxGroup::Term(l1.clone()), t]))
                                        .collect())
                                },
                                _ => unreachable!("First term should always be Add")
                            }
                        })
                    }).collect();

                MinMaxGroup::Max(distributed)
            }
        }
    }

    // Rules for subtraction:
    // a - b = a + (-b)
    pub fn sub<I>(forms: I) -> Self
    where
        I: IntoIterator<Item = MinMaxGroup<Id>>,
    {
        let mut forms = forms.into_iter();
        if let Some(first) = forms.next() {
            MinMaxGroup::add([first].into_iter().chain(forms.map(MinMaxGroup::neg)))
        } else {
            MinMaxGroup::Term(Lin::zero())
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
                MinMaxGroup::Max(inner) => inner,
                other => Set::from([other]),
            }).collect();

        let cloned = terms.clone();

        // Absorption: max(a, min(a,b)) = a
        terms.retain(|form| {
            if let MinMaxGroup::Min(min_terms) = form {
                !cloned.iter().any(|other| min_terms.contains(other))
            } else {
                true
            }
        });

        match terms.len() {
            0 => MinMaxGroup::Term(Lin::lit(i32::MIN)),  // Identity for max
            1 => terms.into_iter().next().unwrap(),
            _ => MinMaxGroup::Max(terms),  // BTreeSet handles idempotence and commutativity
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
                MinMaxGroup::Min(inner) => inner,
                other => Set::from([other]),
            }).collect();

        let cloned = terms.clone();
        // Absorption: min(a, max(a,b)) = a
        terms.retain(|form| {
            if let MinMaxGroup::Max(max_terms) = form {
                !cloned.iter().any(|other| max_terms.contains(other))
            } else {
                true
            }
        });

        match terms.len() {
            0 => MinMaxGroup::Term(Lin::lit(i32::MAX)),  // Identity for max
            1 => terms.into_iter().next().unwrap(),
            _ => MinMaxGroup::Min(terms),  // BTreeSet handles idempotence and commutativity
        }
    }

    pub fn zero() -> Self {
        MinMaxGroup::Term(Lin::zero())
    }
}

//////////////////////////////////////////////////////////////////////////////////////////////
/// Pretty printing, display and Arbitrary for BinGroup
//////////////////////////////////////////////////////////////////////////////////////////////
impl<'a, D, A, T> Pretty<'a, D, A> for MinMaxGroup<T>
where
    D: DocAllocator<'a, A>,
    T: Pretty<'a, D, A> + Clone + Ord,
    D::Doc: Clone,
    A: 'a + Clone,
{
    fn pretty(self, allocator: &'a D) -> DocBuilder<'a, D, A> {
        match self {
            MinMaxGroup::Max(terms) =>
                allocator.concat([
                    allocator.text("max {"),
                    allocator.intersperse(terms.into_iter().map(|term| term.pretty(allocator)),
                        allocator.text(", ")),
                    allocator.text("}")
                ]),
            MinMaxGroup::Min(terms) =>
                allocator.concat([
                    allocator.text("min {"),
                    allocator.intersperse(terms.into_iter().map(|term| term.pretty(allocator)),
                        allocator.text(", ")),
                    allocator.text("}")
                ]),
            MinMaxGroup::Term(term) => term.pretty(allocator),
        }
    }
}

/// Display instance calls the pretty printer
impl<'a, T> fmt::Display for MinMaxGroup<T>
where
    T: Pretty<'a, BoxAllocator, ()> + Clone + Ord,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        <MinMaxGroup<T> as Pretty<'_, BoxAllocator, ()>>::pretty(self.clone(), &BoxAllocator)
            .1
            .render_fmt(100, f)
    }
}

#[test]
fn test_minmax_idempotence() {
    let a = MinMaxGroup::var("a");
    // max(a,a) = a
    let max_aa = MinMaxGroup::max([a.clone(), a.clone()]);
    assert_eq!(max_aa, a);
}

#[test]
fn test_minmax_absorption() {
    let a = MinMaxGroup::var("a");
    let b = MinMaxGroup::var("b");
    // min(a, max(a,b)) = a
    let min_a_maxab =
        MinMaxGroup::min([a.clone(), MinMaxGroup::max([a.clone(), b.clone()])]);
    assert_eq!(min_a_maxab, a);

    // max(a,min(a,b)) = a
    let max_a_minab =
        MinMaxGroup::max([a.clone(), MinMaxGroup::min([a.clone(), b.clone()])]);
    assert_eq!(max_a_minab, a);
}

#[test]
fn test_minmax_distributivity() {
    let a = MinMaxGroup::var("a");
    let b = MinMaxGroup::var("b");
    let c = MinMaxGroup::var("c");
    // max(a,b) + c = max(a+c, b+c)
    let max_ab = MinMaxGroup::max([a.clone(), b.clone()]);
    let max_ab_plus_c = MinMaxGroup::add([max_ab, c.clone()]);
    let a_plus_c = MinMaxGroup::add([a, c.clone()]);
    let b_plus_c = MinMaxGroup::add([b, c]);
    let expected = MinMaxGroup::max([a_plus_c, b_plus_c]);
    assert_eq!(max_ab_plus_c, expected);
}

#[test]
fn test_minmax_add_distribution() {
    let a = MinMaxGroup::var("a");
    let b = MinMaxGroup::var("b");
    let c = MinMaxGroup::var("c");
    let d = MinMaxGroup::var("d");

    // (max(a,b) + max(c,d)) = max(a+c, a+d, b+c, b+d)
    let max_ab = MinMaxGroup::max([a.clone(), b.clone()]);
    let max_cd = MinMaxGroup::max([c.clone(), d.clone()]);
    let sum = MinMaxGroup::add([max_ab, max_cd]);

    let ac = MinMaxGroup::add([a.clone(), c.clone()]);
    let ad = MinMaxGroup::add([a, d.clone()]);
    let bc = MinMaxGroup::add([b.clone(), c]);
    let bd = MinMaxGroup::add([b, d]);

    let expected = MinMaxGroup::max([ac, ad, bc, bd]);
    assert_eq!(sum, expected);
}

#[test]
fn test_minmax_negation() {
    let a = MinMaxGroup::var("a");
    let b = MinMaxGroup::var("b");
    // -max(a,b) = min(-a,-b)
    let max_ab = MinMaxGroup::max([a.clone(), b.clone()]);
    let neg_max = MinMaxGroup::neg(max_ab);
    let neg_a = MinMaxGroup::neg(a);
    let neg_b = MinMaxGroup::neg(b);
    let expected = MinMaxGroup::min([neg_a, neg_b]);
    assert_eq!(neg_max, expected);
}

#[test]
fn test_minmax_n_ary_add() {
    let a = MinMaxGroup::var("a");
    let b = MinMaxGroup::var("b");
    let c = MinMaxGroup::var("c");
    let sum = MinMaxGroup::add([a.clone(), b.clone(), c.clone()]);

    // Associativity: (a + b) + c = a + (b + c)
    let sum2 = MinMaxGroup::add([MinMaxGroup::add([a.clone(), b.clone()]), c.clone()]);
    assert_eq!(sum, sum2);

    // Commutativity: a + b + c = c + b + a
    let sum3 = MinMaxGroup::add([c, b, a]);
    assert_eq!(sum, sum3);
}

#[test]
fn test_minmax_id() {
    let a = MinMaxGroup::var("x");
    let b = MinMaxGroup::var("y");
    let max_ab = MinMaxGroup::max([a.clone(), b.clone()]);
    let neg_max = MinMaxGroup::neg(max_ab);
    let neg_a = MinMaxGroup::neg(a);
    let neg_b = MinMaxGroup::neg(b);
    let expected = MinMaxGroup::min([neg_a, neg_b]);
    assert_eq!(neg_max, expected);
}
