use crate::context::{Set, Ctx};
use thiserror::Error;
use std::fmt;

#[derive(PartialEq, Eq, Error, Debug)]
pub enum EvalError<Id: Ord + Clone + fmt::Display, E> {
    #[error("Unbounded variables found: {0}.")]
    FreeVars(Set<Id>),
    #[error(transparent)]
    ReflectError(#[from] E),
}

pub trait Eval<Id: Ord + Clone + fmt::Display, V: Clone> {
    type ReflectError;
    fn specialize(&mut self, id: &Id, val: V);
    fn reflect(&self) -> Result<V, Self::ReflectError>;
    fn free_vars(&self) -> Set<&Id>;

    fn is_closed(&self) -> bool where Id: Ord {
        self.free_vars().is_empty()
    }

    fn specialize_all(&mut self, ctx: Ctx<Id, V>) {
        for (id, val) in ctx.iter() {
            self.specialize(id, val.clone());
        }
    }

    fn eval(&mut self, ctx: Ctx<Id, V>) -> Result<V, EvalError<Id, Self::ReflectError>> {
        self.specialize_all(ctx);
        if self.is_closed() {
            Ok(self.reflect()?)
        } else {
            Err(EvalError::FreeVars(self.free_vars().cloned()))
        }
    }
}
