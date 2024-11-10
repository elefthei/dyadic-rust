#![feature(btree_extract_if)]
pub mod bin;
pub mod lin;
pub mod group;
pub mod max;
pub mod context;
pub mod traits;
pub mod id;

pub use context::Ctx;
pub use context::Set;
pub use group::BinGroup;
pub use bin::Bin;
pub use lin::Lin;
pub use max::MinMaxGroup;
pub use traits::Eval;
