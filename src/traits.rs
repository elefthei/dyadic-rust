use crate::context::{Set, Ctx};
use thiserror::Error;
use std::fmt;

#[derive(Error, PartialEq, Debug)]
pub enum SpecializeError<T: fmt::Display> {
    #[error("Free variable not found {0}")]
    VarNotFound(T),
    #[error("Free variables found after specializing {0}")]
    FreeVars(Set<T>),
}

pub trait Specializable<T: fmt::Display> {
    fn specialize(&mut self, id: T, val: u8) -> Result<(), SpecializeError<T>>
    where
        Self: Sized;

    fn free_vars(&self) -> Set<&T>;

    fn is_closed(&self) -> bool where T: Ord {
        self.free_vars().is_empty()
    }

    fn specialize_all(&mut self, ctx: Ctx<T, u8>) -> Result<(), SpecializeError<T>>
    where
        T: Ord + Clone,
        Self: Sized {
        for (id, val) in ctx.into_iter() {
            self.specialize(id, val)?;
        }
        let free_vars = self.free_vars();
        if free_vars.is_empty() {
            Ok(())
        } else {
            Err(SpecializeError::FreeVars(free_vars.into_iter().map(|v| v.clone()).collect()))
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
