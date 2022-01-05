#![forbid(unsafe_code)]
#![feature(trait_alias)]

use repository::{DeletionError, Repository, UpdatingError};
use repository::{IterMutTrait, IterTrait};
use std::ops::{Deref, DerefMut};
use std::rc::Rc;

#[derive(Clone, Default, Eq, PartialEq)]
pub struct TDBPData<
    FramesRepo: Repository<T = Frame>,
    ObjectsRepo: Repository<T = Object>,
    ExactMarksRepo: Repository<T = ExactMark<FramesRepo::ID, ObjectsRepo::ID>>,
    InexactMarksRepo: Repository<T = InexactMark<FramesRepo::ID, ObjectsRepo::ID>>,
    ObjectsRelationsRepo: Repository<T = OrObjectsRelation<ObjectsRepo::ID>>,
> {
    frames: Rc<FramesRepo>,
    objects: Rc<ObjectsRepo>,
    exact_marks: Rc<ExactMarksRepo>,
    inexact_marks: Rc<InexactMarksRepo>,
    objects_relations: Rc<ObjectsRelationsRepo>,
}

impl<
        FramesRepo: Repository<T = Frame>,
        ObjectsRepo: Repository<T = Object>,
        ExactMarksRepo: Repository<T = ExactMark<FramesRepo::ID, ObjectsRepo::ID>>,
        InexactMarksRepo: Repository<T = InexactMark<FramesRepo::ID, ObjectsRepo::ID>>,
        ObjectsRelationsRepo: Repository<T = OrObjectsRelation<ObjectsRepo::ID>>,
    > TDBPData<FramesRepo, ObjectsRepo, ExactMarksRepo, InexactMarksRepo, ObjectsRelationsRepo>
{
    #[inline]
    pub fn new(
        frames: FramesRepo,
        objects: ObjectsRepo,
        exact_marks: ExactMarksRepo,
        inexact_marks: InexactMarksRepo,
        objects_relations: ObjectsRelationsRepo,
    ) -> Self {
        TDBPData {
            frames: Rc::new(frames),
            objects: Rc::new(objects),
            exact_marks: Rc::new(exact_marks),
            inexact_marks: Rc::new(inexact_marks),
            objects_relations: Rc::new(objects_relations),
        }
    }

    #[inline]
    pub fn create_frame(&mut self, frame: Frame) -> FramesRepo::ID {
        Rc::make_mut(&mut self.frames).create(frame)
    }

    #[inline]
    pub fn read_frame(&self, id: &FramesRepo::ID) -> Option<Frame> {
        self.frames.read(id)
    }

    #[inline]
    pub fn update_frame(
        &mut self,
        id: &FramesRepo::ID,
        new_frame: Frame,
    ) -> Result<(), FrameUpdatingError> {
        Rc::make_mut(&mut self.frames).update(id, new_frame)
    }

    pub fn delete_frame(
        &mut self,
        id: &FramesRepo::ID,
    ) -> Result<(), FrameDeletionError<ExactMarksRepo::ID, InexactMarksRepo::ID>> {
        {
            let mut exact_marks_iter = self.exact_marks.iter();
            while let Some((exact_mark_id, exact_mark)) = exact_marks_iter.next() {
                if &exact_mark.frame_id == id {
                    return Err(FrameDeletionError::ReferencedByExactMark {
                        id: exact_mark_id.clone(),
                    });
                }
            }
        }
        {
            let mut inexact_marks_iter = self.inexact_marks.iter();
            while let Some((inexact_mark_id, inexact_mark)) = inexact_marks_iter.next() {
                if &inexact_mark.frame_id == id {
                    return Err(FrameDeletionError::ReferencedByInexactMark {
                        id: inexact_mark_id.clone(),
                    });
                }
            }
        }
        Ok(Rc::make_mut(&mut self.frames).delete(id)?)
    }

    #[inline]
    pub fn frame_exists(&self, id: &FramesRepo::ID) -> bool {
        self.frames.exists(id)
    }

    #[inline]
    pub fn get_frame(&self, id: &FramesRepo::ID) -> Option<FramesRepo::Get<'_>> {
        self.frames.get(id)
    }

    #[inline]
    pub fn get_frame_mut(&mut self, id: &FramesRepo::ID) -> Option<FramesRepo::GetMut<'_>> {
        Rc::make_mut(&mut self.frames).get_mut(id)
    }

    #[inline]
    pub fn get_frames(&self) -> FramesRepo::Iter<'_> {
        self.frames.iter()
    }

    #[inline]
    pub fn get_frames_mut(&mut self) -> FramesRepo::IterMut<'_> {
        Rc::make_mut(&mut self.frames).iter_mut()
    }

    #[inline]
    pub fn get_frames_ids(&self) -> FramesRepo::GetIds<'_> {
        self.frames.get_ids()
    }

    #[inline]
    pub fn create_object(&mut self, object: Object) -> ObjectsRepo::ID {
        Rc::make_mut(&mut self.objects).create(object)
    }

    #[inline]
    pub fn read_object(&self, id: &ObjectsRepo::ID) -> Option<Object> {
        self.objects.read(id)
    }

    #[inline]
    pub fn update_object(
        &mut self,
        id: &ObjectsRepo::ID,
        new_object: Object,
    ) -> Result<(), ObjectUpdatingError> {
        Rc::make_mut(&mut self.objects).update(id, new_object)
    }

    pub fn delete_object(
        &mut self,
        id: &ObjectsRepo::ID,
    ) -> Result<(), ObjectDeletionError<ExactMarksRepo::ID, InexactMarksRepo::ID>> {
        {
            let mut exact_marks_iter = self.exact_marks.iter();
            while let Some((exact_mark_id, exact_mark)) = exact_marks_iter.next() {
                if &exact_mark.object_id == id {
                    return Err(ObjectDeletionError::ReferencedByExactMark {
                        id: exact_mark_id.clone(),
                    });
                }
            }
        }
        {
            let mut inexact_marks_iter = self.inexact_marks.iter();
            while let Some((inexact_mark_id, inexact_mark)) = inexact_marks_iter.next() {
                if &inexact_mark.object_id == id {
                    return Err(ObjectDeletionError::ReferencedByInexactMark {
                        id: inexact_mark_id.clone(),
                    });
                }
            }
        }
        Ok(Rc::make_mut(&mut self.objects).delete(id)?)
    }

    #[inline]
    pub fn object_exists(&self, id: &ObjectsRepo::ID) -> bool {
        self.objects.exists(id)
    }

    #[inline]
    pub fn get_object(&self, id: &ObjectsRepo::ID) -> Option<ObjectsRepo::Get<'_>> {
        self.objects.get(id)
    }

    #[inline]
    pub fn get_object_mut(&mut self, id: &ObjectsRepo::ID) -> Option<ObjectsRepo::GetMut<'_>> {
        Rc::make_mut(&mut self.objects).get_mut(id)
    }

    #[inline]
    pub fn get_objects(&self) -> ObjectsRepo::Iter<'_> {
        self.objects.iter()
    }

    #[inline]
    pub fn get_objects_mut(&mut self) -> ObjectsRepo::IterMut<'_> {
        Rc::make_mut(&mut self.objects).iter_mut()
    }

    #[inline]
    pub fn get_objects_ids(&self) -> ObjectsRepo::GetIds<'_> {
        self.objects.get_ids()
    }

    #[inline]
    pub fn create_exact_mark(
        &mut self,
        exact_mark: ExactMark<FramesRepo::ID, ObjectsRepo::ID>,
    ) -> Result<ExactMarksRepo::ID, ExactMarkCreationError> {
        self.check_frame_id_exists(&exact_mark.frame_id)?;
        self.check_object_id_exists(&exact_mark.object_id)?;
        Ok(Rc::make_mut(&mut self.exact_marks).create(exact_mark))
    }

    #[inline]
    pub fn update_exact_mark_non_ref_data(
        &mut self,
        id: &ExactMarksRepo::ID,
        exact_mark_non_ref_data: ExactMarkNonRefData,
    ) -> Result<(), ExactMarkNonRefUpdatingError> {
        if let Some(mut exact_mark) = Rc::make_mut(&mut self.exact_marks).get_mut(id) {
            (*exact_mark).non_ref_data = exact_mark_non_ref_data;
            Ok(())
        } else {
            Err(ExactMarkNonRefUpdatingError::NoSuchID)
        }
    }

    #[inline]
    pub fn delete_exact_mark(
        &mut self,
        id: &ExactMarksRepo::ID,
    ) -> Result<(), ExactMarkDeletionError> {
        Rc::make_mut(&mut self.exact_marks).delete(id)
    }

    #[inline]
    pub fn exact_mark_exists(&self, id: &ExactMarksRepo::ID) -> bool {
        self.exact_marks.exists(id)
    }

    #[inline]
    pub fn get_exact_mark(&self, id: &ExactMarksRepo::ID) -> Option<ExactMarksRepo::Get<'_>> {
        self.exact_marks.get(id)
    }

    #[inline]
    pub fn get_exact_mark_non_ref_data_mut(
        &mut self,
        id: &ExactMarksRepo::ID,
    ) -> Option<ExactMarkNonRefDataGetMut<'_, FramesRepo::ID, ObjectsRepo::ID, ExactMarksRepo>>
    {
        Rc::make_mut(&mut self.exact_marks)
            .get_mut(id)
            .map(|exact_mark_get_mut| ExactMarkNonRefDataGetMut { exact_mark_get_mut })
    }

    #[inline]
    pub fn get_exact_marks(&self) -> ExactMarksRepo::Iter<'_> {
        self.exact_marks.iter()
    }

    #[inline]
    pub fn get_exact_marks_non_ref_datas_mut(
        &mut self,
    ) -> ExactMarksNonRefDatasIterMut<'_, FramesRepo::ID, ObjectsRepo::ID, ExactMarksRepo> {
        ExactMarksNonRefDatasIterMut {
            exact_marks_iter_mut: Rc::make_mut(&mut self.exact_marks).iter_mut(),
        }
    }

    #[inline]
    pub fn get_exact_marks_ids(&self) -> ExactMarksRepo::GetIds<'_> {
        self.exact_marks.get_ids()
    }

    #[inline]
    pub fn create_inexact_mark(
        &mut self,
        inexact_mark: InexactMark<FramesRepo::ID, ObjectsRepo::ID>,
    ) -> Result<InexactMarksRepo::ID, InexactMarkCreationError> {
        self.check_frame_id_exists(&inexact_mark.frame_id)?;
        self.check_object_id_exists(&inexact_mark.object_id)?;
        Ok(Rc::make_mut(&mut self.inexact_marks).create(inexact_mark))
    }

    #[inline]
    pub fn update_inexact_mark_non_ref_data(
        &mut self,
        id: &InexactMarksRepo::ID,
        inexact_mark_non_ref_data: InexactMarkNonRefData,
    ) -> Result<(), InexactMarkNonRefUpdatingError> {
        if let Some(mut inexact_mark) = Rc::make_mut(&mut self.inexact_marks).get_mut(id) {
            (*inexact_mark).non_ref_data = inexact_mark_non_ref_data;
            Ok(())
        } else {
            Err(InexactMarkNonRefUpdatingError::NoSuchID)
        }
    }

    #[inline]
    pub fn delete_inexact_mark(
        &mut self,
        id: &InexactMarksRepo::ID,
    ) -> Result<(), InexactMarkDeletionError> {
        Rc::make_mut(&mut self.inexact_marks).delete(id)
    }

    #[inline]
    pub fn inexact_mark_exists(&self, id: &InexactMarksRepo::ID) -> bool {
        self.inexact_marks.exists(id)
    }

    #[inline]
    pub fn get_inexact_mark(&self, id: &InexactMarksRepo::ID) -> Option<InexactMarksRepo::Get<'_>> {
        self.inexact_marks.get(id)
    }

    #[inline]
    pub fn get_inexact_mark_non_ref_data_mut(
        &mut self,
        id: &InexactMarksRepo::ID,
    ) -> Option<InexactMarkNonRefDataGetMut<'_, FramesRepo::ID, ObjectsRepo::ID, InexactMarksRepo>>
    {
        Rc::make_mut(&mut self.inexact_marks)
            .get_mut(id)
            .map(|inexact_mark_get_mut| InexactMarkNonRefDataGetMut {
                inexact_mark_get_mut,
            })
    }

    #[inline]
    pub fn get_inexact_marks(&self) -> InexactMarksRepo::Iter<'_> {
        self.inexact_marks.iter()
    }

    #[inline]
    pub fn get_inexact_marks_non_ref_datas_mut(
        &mut self,
    ) -> InexactMarksNonRefDatasIterMut<'_, FramesRepo::ID, ObjectsRepo::ID, InexactMarksRepo> {
        InexactMarksNonRefDatasIterMut {
            inexact_marks_iter_mut: Rc::make_mut(&mut self.inexact_marks).iter_mut(),
        }
    }

    #[inline]
    pub fn get_inexact_marks_ids(&self) -> InexactMarksRepo::GetIds<'_> {
        self.inexact_marks.get_ids()
    }

    #[inline]
    pub fn create_objects_relation(
        &mut self,
        objects_relation: OrObjectsRelation<ObjectsRepo::ID>,
    ) -> Result<ObjectsRelationsRepo::ID, ObjectRelationCreationError<ObjectsRepo::ID>> {
        self.check_all_or_relation_objects_exist(&objects_relation)?;
        Ok(Rc::make_mut(&mut self.objects_relations).create(objects_relation))
    }

    #[inline]
    fn check_all_or_relation_objects_exist(
        &self,
        objects_relation: &OrObjectsRelation<ObjectsRepo::ID>,
    ) -> Result<(), ObjectIDDoesNotExist<ObjectsRepo::ID>> {
        for simple_relation in &objects_relation.simple_relations {
            self.check_all_simple_relation_objects_exist(simple_relation)?;
        }
        for and_relation in &objects_relation.and_relations {
            self.check_all_and_relation_objects_exist(and_relation)?;
        }
        Ok(())
    }

    #[inline]
    fn check_all_and_relation_objects_exist(
        &self,
        objects_relation: &AndObjectsRelation<ObjectsRepo::ID>,
    ) -> Result<(), ObjectIDDoesNotExist<ObjectsRepo::ID>> {
        for simple_relation in &objects_relation.simple_relations {
            self.check_all_simple_relation_objects_exist(simple_relation)?;
        }
        for or_relation in &objects_relation.or_relations {
            self.check_all_or_relation_objects_exist(or_relation)?;
        }
        Ok(())
    }

    #[inline]
    fn check_all_simple_relation_objects_exist(
        &self,
        objects_relation: &SimpleObjectsRelation<ObjectsRepo::ID>,
    ) -> Result<(), ObjectIDDoesNotExist<ObjectsRepo::ID>> {
        match objects_relation {
            SimpleObjectsRelation::EqClass(eq_class) => {
                for object_id in &eq_class.members {
                    self.check_object_id_exists(object_id)?;
                }
            }
            SimpleObjectsRelation::Eq(eq) => {
                self.check_object_id_exists(&eq.object1)?;
                self.check_object_id_exists(&eq.object2)?;
            }
            SimpleObjectsRelation::SubObject(sub_object) => {
                self.check_object_id_exists(&sub_object.sub_object)?;
                self.check_object_id_exists(&sub_object.super_object)?;
            }
        }
        Ok(())
    }

    #[inline]
    pub fn delete_objects_relation(
        &mut self,
        id: &ObjectsRelationsRepo::ID,
    ) -> Result<(), ObjectRelationDeletionError> {
        Rc::make_mut(&mut self.objects_relations).delete(id)
    }

    #[inline]
    pub fn objects_relation_exists(&self, id: &ObjectsRelationsRepo::ID) -> bool {
        self.objects_relations.exists(id)
    }

    #[inline]
    pub fn get_objects_relation(
        &self,
        id: &ObjectsRelationsRepo::ID,
    ) -> Option<ObjectsRelationsRepo::Get<'_>> {
        self.objects_relations.get(id)
    }

    #[inline]
    pub fn get_objects_relations(&self) -> ObjectsRelationsRepo::Iter<'_> {
        self.objects_relations.iter()
    }

    #[inline]
    pub fn get_objects_relations_ids(&self) -> ObjectsRelationsRepo::GetIds<'_> {
        self.objects_relations.get_ids()
    }

    #[inline]
    fn check_frame_id_exists(
        &self,
        frame_id: &FramesRepo::ID,
    ) -> Result<(), FrameIDDoesNotExist<FramesRepo::ID>> {
        if !self.frames.exists(frame_id) {
            return Err(FrameIDDoesNotExist {
                frame_id: frame_id.clone(),
            });
        }
        Ok(())
    }

    #[inline]
    fn check_object_id_exists(
        &self,
        object_id: &ObjectsRepo::ID,
    ) -> Result<(), ObjectIDDoesNotExist<ObjectsRepo::ID>> {
        if !self.objects.exists(object_id) {
            return Err(ObjectIDDoesNotExist {
                object_id: object_id.clone(),
            });
        }
        Ok(())
    }
}

struct FrameIDDoesNotExist<FrameID> {
    #[allow(dead_code)]
    frame_id: FrameID,
}

struct ObjectIDDoesNotExist<ObjectID> {
    object_id: ObjectID,
}

pub type FrameUpdatingError = UpdatingError;

#[derive(Copy, Clone, Debug)]
pub enum FrameDeletionError<ExactMarkID, InexactMarkID> {
    NoSuchID,
    ReferencedByExactMark { id: ExactMarkID },
    ReferencedByInexactMark { id: InexactMarkID },
}

impl<ExactMarkID, InexactMarkID> From<DeletionError>
    for FrameDeletionError<ExactMarkID, InexactMarkID>
{
    fn from(error: DeletionError) -> Self {
        match error {
            DeletionError::NoSuchID => FrameDeletionError::NoSuchID,
        }
    }
}

pub type ObjectUpdatingError = UpdatingError;

#[derive(Copy, Clone, Debug)]
pub enum ObjectDeletionError<ExactMarkID, InexactMarkID> {
    NoSuchID,
    ReferencedByExactMark { id: ExactMarkID },
    ReferencedByInexactMark { id: InexactMarkID },
}

impl<ExactMarkID, InexactMarkID> From<DeletionError>
    for ObjectDeletionError<ExactMarkID, InexactMarkID>
{
    fn from(error: DeletionError) -> Self {
        match error {
            DeletionError::NoSuchID => ObjectDeletionError::NoSuchID,
        }
    }
}

#[derive(Copy, Clone, Debug)]
pub enum ExactMarkCreationError {
    NoSuchFrameID,
    NoSuchObjectID,
}

impl<FrameID> From<FrameIDDoesNotExist<FrameID>> for ExactMarkCreationError {
    fn from(_: FrameIDDoesNotExist<FrameID>) -> Self {
        ExactMarkCreationError::NoSuchFrameID
    }
}

impl<ObjectID> From<ObjectIDDoesNotExist<ObjectID>> for ExactMarkCreationError {
    fn from(_: ObjectIDDoesNotExist<ObjectID>) -> Self {
        ExactMarkCreationError::NoSuchObjectID
    }
}

pub type ExactMarkNonRefUpdatingError = UpdatingError;

pub type ExactMarkDeletionError = DeletionError;

pub struct ExactMarkNonRefDataGetMut<
    'a,
    FrameID: 'a,
    ObjectID: 'a,
    ExactMarksRepo: 'a + Repository<T = ExactMark<FrameID, ObjectID>>,
> {
    exact_mark_get_mut: ExactMarksRepo::GetMut<'a>,
}

impl<
        'a,
        FrameID: 'a,
        ObjectID: 'a,
        ExactMarksRepo: Repository<T = ExactMark<FrameID, ObjectID>>,
    > Deref for ExactMarkNonRefDataGetMut<'a, FrameID, ObjectID, ExactMarksRepo>
{
    type Target = ExactMarkNonRefData;

    #[inline]
    fn deref(&self) -> &Self::Target {
        &(*self.exact_mark_get_mut).non_ref_data
    }
}

impl<
        'a,
        FrameID: 'a,
        ObjectID: 'a,
        ExactMarksRepo: Repository<T = ExactMark<FrameID, ObjectID>>,
    > DerefMut for ExactMarkNonRefDataGetMut<'a, FrameID, ObjectID, ExactMarksRepo>
{
    #[inline]
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut (*self.exact_mark_get_mut).non_ref_data
    }
}

pub struct ExactMarksNonRefDatasIterMut<
    'a,
    FrameID: 'a,
    ObjectID: 'a,
    ExactMarksRepo: 'a + Repository<T = ExactMark<FrameID, ObjectID>>,
> {
    exact_marks_iter_mut: ExactMarksRepo::IterMut<'a>,
}

impl<
        'a,
        FrameID: 'a,
        ObjectID: 'a,
        ExactMarksRepo: 'a + Repository<T = ExactMark<FrameID, ObjectID>>,
    > IterMutTrait for ExactMarksNonRefDatasIterMut<'a, FrameID, ObjectID, ExactMarksRepo>
{
    type ID = ExactMarksRepo::ID;
    type T = ExactMarkNonRefData;

    fn next(&mut self) -> Option<(&Self::ID, &mut Self::T)> {
        self.exact_marks_iter_mut
            .next()
            .map(|(id, exact_mark)| (id, &mut exact_mark.non_ref_data))
    }
}

#[derive(Copy, Clone, Debug)]
pub enum InexactMarkCreationError {
    NoSuchFrameID,
    NoSuchObjectID,
}

impl<FrameID> From<FrameIDDoesNotExist<FrameID>> for InexactMarkCreationError {
    fn from(_: FrameIDDoesNotExist<FrameID>) -> Self {
        InexactMarkCreationError::NoSuchFrameID
    }
}

impl<ObjectID> From<ObjectIDDoesNotExist<ObjectID>> for InexactMarkCreationError {
    fn from(_: ObjectIDDoesNotExist<ObjectID>) -> Self {
        InexactMarkCreationError::NoSuchObjectID
    }
}

pub type InexactMarkNonRefUpdatingError = UpdatingError;

pub type InexactMarkDeletionError = DeletionError;

pub struct InexactMarkNonRefDataGetMut<
    'a,
    FrameID: 'a,
    ObjectID: 'a,
    InexactMarksRepo: 'a + Repository<T = InexactMark<FrameID, ObjectID>>,
> {
    inexact_mark_get_mut: InexactMarksRepo::GetMut<'a>,
}

impl<
        'a,
        FrameID: 'a,
        ObjectID: 'a,
        InexactMarksRepo: Repository<T = InexactMark<FrameID, ObjectID>>,
    > Deref for InexactMarkNonRefDataGetMut<'a, FrameID, ObjectID, InexactMarksRepo>
{
    type Target = InexactMarkNonRefData;

    #[inline]
    fn deref(&self) -> &Self::Target {
        &(*self.inexact_mark_get_mut).non_ref_data
    }
}

impl<
        'a,
        FrameID: 'a,
        ObjectID: 'a,
        InexactMarksRepo: Repository<T = InexactMark<FrameID, ObjectID>>,
    > DerefMut for InexactMarkNonRefDataGetMut<'a, FrameID, ObjectID, InexactMarksRepo>
{
    #[inline]
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut (*self.inexact_mark_get_mut).non_ref_data
    }
}

pub struct InexactMarksNonRefDatasIterMut<
    'a,
    FrameID: 'a,
    ObjectID: 'a,
    InexactMarksRepo: 'a + Repository<T = InexactMark<FrameID, ObjectID>>,
> {
    inexact_marks_iter_mut: InexactMarksRepo::IterMut<'a>,
}

impl<
        'a,
        FrameID: 'a,
        ObjectID: 'a,
        InexactMarksRepo: 'a + Repository<T = InexactMark<FrameID, ObjectID>>,
    > IterMutTrait for InexactMarksNonRefDatasIterMut<'a, FrameID, ObjectID, InexactMarksRepo>
{
    type ID = InexactMarksRepo::ID;
    type T = InexactMarkNonRefData;

    fn next(&mut self) -> Option<(&Self::ID, &mut Self::T)> {
        self.inexact_marks_iter_mut
            .next()
            .map(|(id, inexact_mark)| (id, &mut inexact_mark.non_ref_data))
    }
}

#[derive(Copy, Clone, Debug)]
pub enum ObjectRelationCreationError<ObjectID> {
    NoSuchObjectID { object_id: ObjectID },
}

impl<ObjectID> From<ObjectIDDoesNotExist<ObjectID>> for ObjectRelationCreationError<ObjectID> {
    fn from(error: ObjectIDDoesNotExist<ObjectID>) -> Self {
        ObjectRelationCreationError::NoSuchObjectID {
            object_id: error.object_id,
        }
    }
}

pub type ObjectRelationDeletionError = DeletionError;

// TODO File (identified by basename/checksum) entity & its relations with Frame
// (multi-frame file, multiple files for frame, coords transformations..?)
pub struct Frame {}

pub struct Object {}

pub struct ExactMark<FrameID, ObjectID> {
    pub non_ref_data: ExactMarkNonRefData,
    pub frame_id: FrameID,
    pub object_id: ObjectID,
}

pub struct ExactMarkNonRefData {
    pub pos: MarkPos,
}

pub struct InexactMark<FrameID, ObjectID> {
    pub non_ref_data: InexactMarkNonRefData,
    pub frame_id: FrameID,
    pub object_id: ObjectID,
}

pub struct InexactMarkNonRefData {
    pub pos: MarkPos,
}

/// Coords are device normalized. (0, 0) is the center of the image.
/// E.g. right bottom corner of WxH photo has coords (1, H/W) if W>=H.
pub struct MarkPos {
    pub x: f64,
    pub y: f64,
}

pub struct AndObjectsRelation<ObjectID> {
    pub simple_relations: Vec<SimpleObjectsRelation<ObjectID>>,
    pub or_relations: Vec<OrObjectsRelation<ObjectID>>,
}

pub struct OrObjectsRelation<ObjectID> {
    pub simple_relations: Vec<SimpleObjectsRelation<ObjectID>>,
    pub and_relations: Vec<AndObjectsRelation<ObjectID>>,
}

pub enum SimpleObjectsRelation<ObjectID> {
    EqClass(MarkableObjectsEqClass<ObjectID>),
    Eq(MarkableObjectsEq<ObjectID>),
    SubObject(MarkableObjectsSubObject<ObjectID>),
}

/// Mainly for migration of data created by (old) C++ TDBP application
pub struct MarkableObjectsEqClass<ObjectID> {
    pub members: Vec<ObjectID>,
}

pub struct MarkableObjectsEq<ObjectID> {
    pub object1: ObjectID,
    pub object2: ObjectID,
}

pub struct MarkableObjectsSubObject<ObjectID> {
    pub sub_object: ObjectID,
    pub super_object: ObjectID,
}
