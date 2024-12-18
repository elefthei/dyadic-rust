use crate::context::{Set, Ctx};

pub trait Specializable<T, V> {
    fn specialize(&mut self, id: &T, val: V)
    where
        Self: Sized;

    fn free_vars(&self) -> Set<&T>;

    fn is_closed(&self) -> bool where T: Ord {
        self.free_vars().is_empty()
    }

    fn specialize_all(&mut self, ctx: Ctx<T, V>)
    where
        T: Ord + Clone,
        V: Clone,
        Self: Sized {
        for (id, val) in ctx.iter() {
            self.specialize(id, val.clone());
        }
    }
}

/// A term that can be normalized
pub trait Normalizable {
    fn normalize(&mut self);

    /// Check if it is in normal form by syntactic equality
    fn is_normal(&self) -> bool where Self: Clone + Eq {
        let mut clone = self.clone();
        clone.normalize();
        self == &clone
    }

    /// Equality modulo normalization
    fn eqn(&self, other: &Self) -> bool where Self: Eq + Clone {
        let mut self_clone = self.clone();
        let mut other_clone = other.clone();
        self_clone.normalize();
        other_clone.normalize();
        self_clone == other_clone
    }
}

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
                stringify!($left), stringify!($right), $left, $right, l, r,
            );
        }
    });
}
