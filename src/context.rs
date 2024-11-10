use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use pretty::{Pretty, DocAllocator, DocBuilder, BoxAllocator};

/// General BTreeMap context
#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub struct Ctx<K, V>(BTreeMap<K, V>);

/// Pretty printer instance for Ctx
impl<'a, D, A, K, V> Pretty<'a, D, A> for Ctx<K, V>
where
    K: Clone + Pretty<'a, D, A>,
    V: Clone + Pretty<'a, D, A>,
    D: DocAllocator<'a, A>,
    D::Doc: Clone,
    A: 'a + Clone,
{
    fn pretty(self, allocator: &'a D) -> DocBuilder<'a, D, A> {
        allocator.concat([
            allocator.text("{"),
            allocator.intersperse(
                self.0.iter().map(|(k, v)| {
                    k.clone().pretty(allocator)
                        .append(allocator.text(" -> "))
                        .append(v.clone().pretty(allocator))
                }),
                ", ",
            ),
            allocator.text("}"),
        ])
    }
}

/// IntoIterator instance for Ctx
impl<K, V> IntoIterator for Ctx<K, V> {
    type Item = (K, V);
    type IntoIter = std::collections::btree_map::IntoIter<K, V>;

    fn into_iter(self) -> Self::IntoIter {
        self.0.into_iter()
    }
}

/// Iterator for borrowing key-value pairs
pub struct CtxIterator<'a, K, V> {
    iter: std::collections::btree_map::Iter<'a, K, V>,
}

/// Iterator instance for Ctx
impl<'a, K, V> Iterator for CtxIterator<'a, K, V> {
    type Item = (&'a K, &'a V);

    fn next(&mut self) -> Option<Self::Item> {
        self.iter.next()
    }
}

impl<K, V> FromIterator<(K, V)> for Ctx<K, V>
where
    K: Ord
{
    fn from_iter<I: IntoIterator<Item = (K, V)>>(iter: I) -> Self {
        Ctx(BTreeMap::from_iter(iter))
    }
}

/// Display instance for Ctx calls the pretty printer
impl<'a, K, V> fmt::Display for Ctx<K, V>
where
    K: Pretty<'a, BoxAllocator, ()> + Clone,
    V: Pretty<'a, BoxAllocator, ()> + Clone
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        <Ctx<_, _> as Pretty<'_, BoxAllocator, ()>>::pretty(self.clone(), &BoxAllocator)
            .1
            .render_fmt(80, f)
    }
}

/// Special and wrapper methods for Ctx
impl<K: Ord, V> Ctx<K, V> {
    pub fn new() -> Self {
        Ctx(BTreeMap::new())
    }

    pub fn singleton(k: K, v: V) -> Self {
        Ctx(BTreeMap::from([(k, v)]))
    }

    pub fn find<FF>(&self, f: FF) -> Option<(&K, &V)> where FF: Fn(&K, &V) -> bool {
        self.0.iter().find(|(k, v)| f(k, v))
    }
    pub fn insert_with<FF>(&mut self, k: K, v1: V, f: &FF)
    where
        V:Clone,
        FF: Fn(V, V) -> V
    {
        self.0.entry(k).and_modify(|v2| *v2 = f(v1.clone(), v2.clone())).or_insert(v1);
    }

    pub fn insert(&mut self, k: K, v: V)
    where V: Clone
    {
        self.insert_with(k, v, &|_, v| v);
    }

    pub fn append_with<It, FF>(&mut self, it: It, f: &FF) -> &mut Self
    where
        V: Clone,
        It: Iterator<Item = (K, V)>,
        FF: Fn(V, V) -> V,
    {
        for (k, v) in it {
            self.insert_with(k, v, f);
        }
        self
    }
    pub fn append<It>(&mut self, it: It) -> &mut Self
    where
        V: Clone,
        It: Iterator<Item = (K, V)>,
    {
        self.append_with(it, &|_, v| v)
    }

    pub fn union_with<FF>(&self, other: Self, f: &FF) -> Self
    where
        K: Clone,
        V: Clone,
        FF: Fn(V, V) -> V
    {
        let mut c = self.clone();
        c.append_with(other.into_iter(), f);
        c
    }

    pub fn union(&self, other: Self) -> Self
    where K: Clone, V: Clone
    {
        self.union_with(other, &|_, v| v)
    }

    pub fn intersection_with<VV, FF>(&self, other: Self, f: &FF) -> Ctx<K, VV>
    where
        K: Clone,
        V: Clone,
        VV: Clone,
        FF: Fn(V, V) -> VV
    {
        let mut diff = Ctx::new();
        for (k, v1) in self.0.clone().into_iter() {
            if let Some(v2) = other.0.get(&k) {
                diff.insert(k, f(v1.clone(), v2.clone()));
            }
        }
        diff
    }
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
    pub fn get(&self, k: &K) -> Option<&V> {
        self.0.get(k)
    }

    pub fn get_mut(&mut self, k: &K) -> Option<&mut V> {
        self.0.get_mut(k)
    }

    pub fn remove(&mut self, k: &K) -> Option<V>
    where
        V: Clone,
    {
        self.0.remove(k)
    }
    pub fn keys(&self) -> Set<&K> {
        Set(self.0.keys().collect::<BTreeSet<_>>())
    }
    pub fn values(&self) -> impl Iterator<Item = &V> {
        self.0.values()
    }
    pub fn contains(&self, k: &K) -> bool {
        self.0.contains_key(k)
    }
    pub fn iter(&self) -> CtxIterator<K, V> {
        CtxIterator {
            iter: self.0.iter(),
        }
    }
    pub fn iter_mut(&mut self) -> impl Iterator<Item = (&K, &mut V)> {
        self.0.iter_mut()
    }
    pub fn modify<F>(&mut self, mut f: F)
    where
        F: FnMut(&K, &mut V)
    {
        for (k, v) in self.iter_mut() {
            f(k, v);
        }
    }
    pub fn retain(&mut self, f: impl Fn(&K, &mut V) -> bool) {
        self.0.retain(f);
    }

    pub fn entry(&mut self, k: K) -> std::collections::btree_map::Entry<K, V> {
        self.0.entry(k)
    }
    pub fn extract_if(&mut self, f: impl Fn(&K, &V) -> bool) -> Ctx<K, V>
    where
        K: Clone,
        V: Clone,
    {
        let mut c = Ctx::new();
        for (k, v) in self.0.clone().into_iter() {
            if f(&k, &v) {
                c.insert(k, v);
            }
        }
        self.retain(|k, _| !c.contains(k));
        c
    }
}

impl<K: Ord, V> Default for Ctx<K, V> {
    fn default() -> Self {
        Ctx(BTreeMap::new())
    }
}

/// From instance
impl<X, Y, K: From<X> + Ord, V: From<Y>, const N: usize> From<[(X, Y); N]> for Ctx<K, V> {
    fn from(v: [(X, Y); N]) -> Self {
        Ctx(BTreeMap::from_iter(v.into_iter().map(|(k, v)| (K::from(k), V::from(v)))))
    }
}

///////////////////////////////////////////////////////////////////////////////////
// A set of values with Pretty and Display traits and other useful methods
///////////////////////////////////////////////////////////////////////////////////
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Set<V>(BTreeSet<V>);

/// Pretty printer instance
impl<'a, D, A, V> Pretty<'a, D, A> for Set<V>
where
    D: DocAllocator<'a, A>,
    D::Doc: Clone,
    A: 'a + Clone,
    V: Pretty<'a, D, A> + Clone
{
    fn pretty(self, allocator: &'a D) -> DocBuilder<'a, D, A> {
        allocator.concat([
            allocator.text("{"),
            allocator.intersperse(
                self.0.into_iter()
                    .map(|k| k.pretty(allocator)), ", "),
            allocator.text("}"),
        ])
    }
}

impl<V> IntoIterator for Set<V> {
    type Item = V;
    type IntoIter = std::collections::btree_set::IntoIter<V>;

    fn into_iter(self) -> Self::IntoIter {
        self.0.into_iter()
    }
}

// Iterator for borrowing key-value pairs
pub struct SetIterator<'a, V> {
    iter: std::collections::btree_set::Iter<'a, V>,
}

impl<'a, V> Iterator for SetIterator<'a, V> {
    type Item = &'a V;

    fn next(&mut self) -> Option<Self::Item> {
        self.iter.next()
    }
}

impl<V> FromIterator<V> for Set<V>
where
    V: Ord,
{
    fn from_iter<I: IntoIterator<Item = V>>(iter: I) -> Self {
        Set(BTreeSet::from_iter(iter))
    }
}

impl<V> Default for Set<V> {
    fn default() -> Self {
        Set(BTreeSet::new())
    }
}

impl<'a, V> fmt::Display for Set<V>
where
    V: Pretty<'a, BoxAllocator, ()> + Clone
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        <Set<_> as Pretty<'_, BoxAllocator, ()>>::pretty(self.clone(), &BoxAllocator)
                .1
                .render_fmt(80, f)
    }
}

impl<V: Ord, const N: usize> From<[V; N]> for Set<V> {
    fn from(v: [V; N]) -> Self {
        Set(BTreeSet::from_iter(v.into_iter()))
    }
}

impl<V: Ord> From<Vec<V>> for Set<V> {
    fn from(v: Vec<V>) -> Self {
        Set(BTreeSet::from_iter(v.into_iter()))
    }
}

impl<V: Ord> Set<V> {
    pub fn new() -> Self where V: Ord {
        Set(BTreeSet::new())
    }
    pub fn singleton(v: V) -> Self {
        Set(BTreeSet::from([v]))
    }
    pub fn insert_with(&mut self, k: V, f: impl Fn(V) -> V) -> bool {
        if self.0.remove(&k) {
            self.0.insert(f(k))
        } else {
            self.0.insert(k)
        }
    }

    pub fn len(&self) -> usize {
        self.0.len()
    }

    pub fn insert(&mut self, k: V) -> bool {
        self.0.insert(k)
    }

    pub fn append_with<FF, It>(&mut self, it: It, f: &FF) -> bool
    where
        FF: Fn(V) -> V,
        It: Iterator<Item = V>,
    {
        let mut ins = false;
        for k in it {
            ins |= self.insert_with(k, f)
        }
        ins
    }

    pub fn pop_first(&mut self) -> Option<V> {
        self.0.pop_first()
    }

    pub fn append<It>(&mut self, it: It) -> bool
    where
        It: Iterator<Item = V>,
    {
        let mut ins = false;
        for k in it {
            ins |= self.0.insert(k)
        }
        ins
    }
    pub fn first(&self) -> Option<&V> {
        self.0.iter().next()
    }
    pub fn contains(&self, k: &V) -> bool {
        self.0.contains(k)
    }
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
    /// Union of two sets with conflict hanlder [f]
    pub fn union_with<FF>(&self, other: Set<V>, f: &FF) -> Self
    where
        V: Clone,
        FF: Fn(V) -> V
    {
        let mut c = self.clone();
        c.append_with(other.into_iter(), f);
        c
    }

    /// Union of two sets
    pub fn union(&self, other: Set<V>) -> Self
    where
        V: Clone
    {
        let mut c = self.clone();
        c.append(other.into_iter());
        c
    }
    /// Intersection of two sets
    pub fn intersection(&self, other: Set<V>) -> Self
    where
        V: Clone
    {
        Set(self.0.intersection(&other.0).cloned().collect())
    }

    pub fn iter(&self) -> SetIterator<V> {
        SetIterator {
            iter: self.0.iter(),
        }
    }

    // Method to iterate over a mutable Vec, modify elements, and return a new Set<T>
    pub fn modify<F>(&mut self, mut f: F)
    where
        V: Clone,
        F: FnMut(&mut V),
    {
        // Convert BTreeSet<V> to Vec<V>
        let mut vec: Vec<V> = self.0.iter().cloned().collect();

        // Apply modification function to each element
        for elem in vec.iter_mut() {
            f(elem);
        }
        // Collect back into Set<V>
        self.0 = vec.into_iter().collect();
    }

    // Extract elements from the map that satisfy a predicate
    pub fn extract_if(&mut self, f: impl Fn(&V) -> bool) -> Set<V>
    where
        V: Clone
    {
        self.0.extract_if(f).collect()
    }

    pub fn retain(&mut self, f: impl Fn(&V) -> bool) {
        self.0.retain(f);
    }
}

impl<V: Ord + Clone> Set<&V> {
    pub fn cloned(self) -> Set<V> {
        Set(self.0.into_iter().cloned().collect())
    }
}
