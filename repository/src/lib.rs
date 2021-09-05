#![forbid(unsafe_code)]
#![feature(trait_alias)]
#![feature(generic_associated_types)]
#![feature(type_alias_impl_trait)]

use std::collections::HashMap;
use std::ops::{Deref, DerefMut};
use std::rc::Rc;

pub trait LendingIterator {
    type Item<'a>
    where
        Self: 'a;

    fn next<'a>(&'a mut self) -> Option<Self::Item<'a>>;
}

pub trait IterTrait {
    type ID;
    type T;

    fn next<'a>(&'a mut self) -> Option<(&'a Self::ID, &'a Self::T)>;
}

// TODO implement LendingIterator for IterTrait

pub trait IterMutTrait {
    type ID;
    type T;

    fn next<'a>(&'a mut self) -> Option<(&'a Self::ID, &'a mut Self::T)>;
}

// TODO implement LendingIterator for IterMutTrait

pub trait IDTrait = Clone + Eq;
pub trait DataTrait = Clone;

pub trait Repository: Clone {
    type ID: IDTrait;
    type T: DataTrait;
    type Get<'a>: Deref<Target = Self::T>
    where
        Self::T: 'a;
    type GetMut<'a>: DerefMut<Target = Self::T>
    where
        Self::T: 'a;
    type Iter<'a>: IterTrait<ID = Self::ID, T = Self::T>
    where
        Self: 'a;
    type IterMut<'a>: IterMutTrait<ID = Self::ID, T = Self::T>
    where
        Self: 'a;
    type GetIds<'a>: Iterator<Item = Self::ID>;

    fn create(&mut self, data: Self::T) -> Self::ID;
    fn read(&self, id: &Self::ID) -> Option<Self::T>;
    fn update(&mut self, id: &Self::ID, new_data: Self::T) -> Result<(), UpdatingError>;
    fn delete(&mut self, id: &Self::ID) -> Result<(), DeletionError>;
    fn exists(&self, id: &Self::ID) -> bool;

    fn get<'a>(&'a self, id: &Self::ID) -> Option<Self::Get<'a>>;
    fn get_mut<'a>(&'a mut self, id: &Self::ID) -> Option<Self::GetMut<'a>>;

    fn iter<'a>(&'a self) -> Self::Iter<'a>;
    fn iter_mut<'a>(&'a mut self) -> Self::IterMut<'a>;

    fn get_ids<'a>(&'a self) -> Self::GetIds<'a>;
}

#[derive(Copy, Clone, Debug)]
pub enum UpdatingError {
    NoSuchID,
}

#[derive(Copy, Clone, Debug)]
pub enum DeletionError {
    NoSuchID,
}

#[derive(Clone)]
pub struct RepositoryImpl<T: DataTrait> {
    // TODO try persistent data structures (im crate)
    objs: HashMap<<Self as Repository>::ID, Rc<T>>,
    next_id: <Self as Repository>::ID,
}

impl<T: DataTrait> RepositoryImpl<T> {
    #[inline]
    pub fn new() -> Self {
        RepositoryImpl {
            objs: HashMap::new(),
            next_id: 0,
        }
    }
}

impl<T: DataTrait> Default for RepositoryImpl<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: DataTrait> Repository for RepositoryImpl<T> {
    type ID = u64;
    type T = T;
    type Get<'a>
    where
        T: 'a,
    = GetImpl<'a, T>;
    type GetMut<'a>
    where
        T: 'a,
    = GetMutImpl<'a, T>;
    type Iter<'a>
    where
        Self: 'a,
    = IterImpl<'a, Self::ID, Self::T>;
    type IterMut<'a>
    where
        Self: 'a,
    = IterMutImpl<'a, Self::ID, Self::T>;
    type GetIds<'a> = impl Iterator<Item = Self::ID>;

    #[inline]
    fn create(&mut self, data: Self::T) -> Self::ID {
        let id = self.next_id;
        self.next_id = self.next_id.checked_add(1).expect("ID overflowed!");
        self.objs.insert(id, Rc::new(data));
        id
    }

    #[inline]
    fn read(&self, id: &Self::ID) -> Option<Self::T> {
        self.objs.get(&id).map(|obj_rc| (**obj_rc).clone())
    }

    #[inline]
    fn update(&mut self, id: &Self::ID, new_data: Self::T) -> Result<(), UpdatingError> {
        if let Some(obj) = self.objs.get_mut(id) {
            *Rc::make_mut(obj) = new_data;
            Ok(())
        } else {
            Err(UpdatingError::NoSuchID)
        }
    }

    #[inline]
    fn delete(&mut self, id: &Self::ID) -> Result<(), DeletionError> {
        if let Some(_) = self.objs.remove(id) {
            Ok(())
        } else {
            Err(DeletionError::NoSuchID)
        }
    }

    #[inline]
    fn exists(&self, id: &Self::ID) -> bool {
        self.objs.contains_key(id)
    }

    #[inline]
    fn get<'a>(&'a self, id: &<Self as Repository>::ID) -> Option<GetImpl<'a, T>> {
        self.objs.get(id).map(|obj| GetImpl { obj: &**obj })
    }

    #[inline]
    fn get_mut<'a>(&'a mut self, id: &<Self as Repository>::ID) -> Option<GetMutImpl<'a, T>> {
        self.objs.get_mut(id).map(|obj| GetMutImpl {
            obj: Rc::make_mut(obj),
        })
    }

    #[inline]
    fn iter<'a>(&'a self) -> Self::Iter<'a> {
        IterImpl {
            iter: self.objs.iter(),
        }
    }

    #[inline]
    fn iter_mut<'a>(&'a mut self) -> Self::IterMut<'a> {
        IterMutImpl {
            iter: self.objs.iter_mut(),
        }
    }

    #[inline]
    fn get_ids<'a>(&'a self) -> Self::GetIds<'a> {
        self.objs.keys().cloned()
    }
}

pub struct GetImpl<'a, T: DataTrait> {
    obj: &'a T,
}

impl<'a, T: DataTrait> Deref for GetImpl<'a, T> {
    type Target = T;

    #[inline]
    fn deref(&self) -> &Self::Target {
        self.obj
    }
}

pub struct GetMutImpl<'a, T: DataTrait> {
    obj: &'a mut T,
}

impl<'a, T: DataTrait> Deref for GetMutImpl<'a, T> {
    type Target = T;

    #[inline]
    fn deref(&self) -> &Self::Target {
        self.obj
    }
}

impl<'a, T: DataTrait> DerefMut for GetMutImpl<'a, T> {
    #[inline]
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.obj
    }
}

pub struct IterImpl<'a, ID: 'a, T: 'a> {
    iter: std::collections::hash_map::Iter<'a, ID, Rc<T>>,
}

impl<'a, ID: 'a, T: 'a + DataTrait> IterTrait for IterImpl<'a, ID, T> {
    type ID = ID;
    type T = T;

    #[inline]
    fn next<'b>(&'b mut self) -> Option<(&ID, &T)> {
        self.iter.next().map(|(id, obj)| (id, &**obj))
    }
}

pub struct IterMutImpl<'a, ID: 'a, T: 'a> {
    iter: std::collections::hash_map::IterMut<'a, ID, Rc<T>>,
}

impl<'a, ID: 'a, T: 'a + DataTrait> IterMutTrait for IterMutImpl<'a, ID, T> {
    type ID = ID;
    type T = T;

    #[inline]
    fn next<'b>(&'b mut self) -> Option<(&ID, &mut T)> {
        self.iter.next().map(|(id, obj)| (id, Rc::make_mut(obj)))
    }
}
