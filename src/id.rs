use pretty::{Pretty, DocAllocator, BoxAllocator, DocBuilder};
use arbitrary::{Unstructured, Arbitrary};
use std::fmt;

#[derive(Clone, Hash, Ord, PartialOrd, Eq, PartialEq)]
pub struct Id(char);

impl<'a, D, A> Pretty<'a, D, A> for Id
where
    D: DocAllocator<'a, A>,
    D::Doc: Clone,
    A: 'a + Clone,
{
    fn pretty(self, allocator: &'a D) -> DocBuilder<'a, D, A> {
        allocator.text(self.0.to_string())
    }
}

impl<'a> Arbitrary<'a> for Id {
    fn arbitrary(u: &mut Unstructured<'a>) -> arbitrary::Result<Id, arbitrary::Error> {
        let c = u.int_in_range(0..=25)?;
        Ok(Id((b'a' + c) as char))
    }
}

/// Display instance calls the pretty printer
impl<'a> fmt::Display for Id {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        <Id as Pretty<'_, BoxAllocator, ()>>::pretty(self.clone(), &BoxAllocator)
            .1
            .render_fmt(4, f)
    }
}

impl<'a> fmt::Debug for Id {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self)
    }
}

