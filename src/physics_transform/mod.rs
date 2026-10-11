//! Manages physics transforms and synchronizes them with [`Transform`].
//!
//! See [`PhysicsTransformPlugin`].

mod transform;
#[allow(unused_imports)]
pub(crate) use transform::init_physics_transform;
pub use transform::{Position, PreSolveDeltaPosition, PreSolveDeltaRotation, Rotation};

mod helper;
pub use helper::PhysicsTransformHelper;

#[cfg(test)]
mod tests;

use crate::{
    prelude::*,
    schedule::{LastPhysicsTick, is_changed_after_tick},
};
use approx::AbsDiffEq;
use bevy::{
    ecs::{
        change_detection::Tick, intern::Interned, schedule::ScheduleLabel, system::SystemChangeTick,
    },
    prelude::*,
    transform::systems::{mark_dirty_trees, propagate_parent_transforms, sync_simple_transforms},
};

/// Manages physics transforms and synchronizes them with [`Transform`].
///
/// # Syncing Between [`Position`]/[`Rotation`] and [`Transform`]
///
/// By default, each body's `Transform` will be updated when [`Position`] or [`Rotation`]
/// change, and vice versa. This means that you can use any of these components to move
/// or position bodies, and the changes be reflected in the other components.
///
/// You can configure what data is synchronized and how it is synchronized
/// using the [`PhysicsTransformConfig`] resource.
///
/// # `Transform` Hierarchies
///
/// When synchronizing changes in [`Position`] or [`Rotation`] to `Transform`,
/// the engine treats nested [rigid bodies](RigidBody) as a flat structure. This means that
/// the bodies move independently of the parents, and moving the parent will not affect the child.
///
/// If you would like a child entity to be rigidly attached to its parent, you could use a [`FixedJoint`]
/// or write your own system to handle hierarchies differently.
pub struct PhysicsTransformPlugin {
    schedule: Interned<dyn ScheduleLabel>,
}

impl PhysicsTransformPlugin {
    /// Creates a [`PhysicsTransformPlugin`] with the schedule that is used for running the [`PhysicsSchedule`].
    ///
    /// The default schedule is `FixedPostUpdate`.
    pub fn new(schedule: impl ScheduleLabel) -> Self {
        Self {
            schedule: schedule.intern(),
        }
    }
}

impl Default for PhysicsTransformPlugin {
    fn default() -> Self {
        Self::new(FixedPostUpdate)
    }
}

impl Plugin for PhysicsTransformPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<PhysicsTransformConfig>();

        // In case `TransformPlugin` is not added
        app.init_resource::<StaticTransformOptimizations>();

        if app
            .world()
            .resource::<PhysicsTransformConfig>()
            .position_to_transform
        {
            app.register_required_components::<Position, Transform>();
            app.register_required_components::<Rotation, Transform>();
        }

        // Run transform propagation and transform-to-position synchronization before physics.
        app.configure_sets(
            self.schedule,
            (
                PhysicsTransformSystems::Propagate,
                PhysicsTransformSystems::TransformToPosition,
            )
                .chain()
                .in_set(PhysicsSystems::Prepare),
        );
        app.add_systems(
            self.schedule,
            (
                mark_dirty_trees,
                propagate_parent_transforms,
                sync_simple_transforms,
            )
                .chain()
                .in_set(PhysicsTransformSystems::Propagate)
                .run_if(|config: Res<PhysicsTransformConfig>| config.propagate_before_physics),
        );
        app.add_systems(
            self.schedule,
            transform_to_position
                .in_set(PhysicsTransformSystems::TransformToPosition)
                .run_if(|config: Res<PhysicsTransformConfig>| config.transform_to_position),
        );

        // Run position-to-transform synchronization after physics.
        app.configure_sets(
            self.schedule,
            PhysicsTransformSystems::PositionToTransform.in_set(PhysicsSystems::Writeback),
        );
        app.add_systems(
            self.schedule,
            position_to_transform
                .in_set(PhysicsTransformSystems::PositionToTransform)
                .run_if(|config: Res<PhysicsTransformConfig>| config.position_to_transform),
        );
    }
}

/// Configures how physics transforms are managed and synchronized with [`Transform`].
#[derive(Resource, Reflect, Clone, Debug, PartialEq, Eq)]
#[reflect(Resource)]
pub struct PhysicsTransformConfig {
    /// If true, [`Transform`] is propagated before stepping physics to ensure that
    /// [`GlobalTransform`] is up-to-date.
    ///
    /// Default: `true`
    pub propagate_before_physics: bool,
    /// Updates [`Position`] and [`Rotation`] based on [`Transform`] changes
    /// in [`PhysicsTransformSystems::TransformToPosition`],
    ///
    /// This allows using transforms for moving and positioning bodies,
    ///
    /// Default: `true`
    pub transform_to_position: bool,
    /// Updates [`Transform`] based on [`Position`] and [`Rotation`] changes
    /// in [`PhysicsTransformSystems::PositionToTransform`],
    ///
    /// Default: `true`
    pub position_to_transform: bool,
    /// Updates [`Collider::scale()`] based on transform changes.
    ///
    /// This allows using transforms for scaling colliders.
    ///
    /// Default: `true`
    pub transform_to_collider_scale: bool,
}

impl Default for PhysicsTransformConfig {
    fn default() -> Self {
        PhysicsTransformConfig {
            propagate_before_physics: true,
            position_to_transform: true,
            transform_to_position: true,
            transform_to_collider_scale: true,
        }
    }
}

/// System sets for managing physics transforms.
#[derive(SystemSet, Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum PhysicsTransformSystems {
    /// Propagates [`Transform`] before physics simulation.
    Propagate,
    /// Updates [`Position`] and [`Rotation`] based on [`Transform`] changes before physics simulation.
    TransformToPosition,
    /// Updates [`Transform`] based on [`Position`] and [`Rotation`] changes after physics simulation.
    PositionToTransform,
}

/// A deprecated alias for [`PhysicsTransformSystems`].
#[deprecated(since = "0.4.0", note = "Renamed to `PhysicsTransformSystems`")]
pub type PhysicsTransformSet = PhysicsTransformSystems;

/// Copies transform changes to [`Position`] and [`Rotation`].
/// This allows users to use transforms for moving and positioning bodies and colliders.
///
/// Computes position relative to the [`PhysicsEnvironment`] ancestor by walking up the
/// hierarchy and composing [`Transform`]s, avoiding f32 precision loss from [`GlobalTransform`]
/// at large distances (e.g. when the world is anchored far from the origin under a
/// floating-origin system). The composition uses [`Scalar`], so it is f32 by default and
/// f64 when avian's `f64` feature is enabled.
#[allow(clippy::type_complexity)]
pub fn transform_to_position(
    mut query: Query<(
        Entity,
        &Transform,
        Option<&ChildOf>,
        &mut Position,
        &mut Rotation,
    )>,
    ancestors: Query<(&Transform, Option<&ChildOf>, Has<PhysicsEnvironment>), Without<Position>>,
    world_lookup: PhysicsEnvironmentLookup,
    worlds: Query<&PhysicsLengthUnit, With<PhysicsEnvironment>>,
    last_physics_tick: Res<LastPhysicsTick>,
    system_tick: SystemChangeTick,
) {
    // On the first tick, the last physics tick and system tick are both defaulted to 0,
    // but to handle change detection correctly, the system tick should always be larger.
    // So we use a minimum system tick of 1 here.
    let this_run = if last_physics_tick.0.get() == 0 {
        Tick::new(1)
    } else {
        system_tick.this_run()
    };

    // If the rotation and `Rotation` differ by less than 0.1 degrees, we ignore the change.
    let rotation_tolerance = 0.1f32.to_radians();

    for (entity, transform, child_of, mut position, mut rotation) in &mut query {
        // In no environment: not simulated, and no frame to be in.
        let Some(length_unit) = world_lookup
            .environment_of(entity)
            .and_then(|world| worlds.get(world).ok())
        else {
            continue;
        };
        // If the translation and `Position` differ by less than 0.01 mm, we ignore the change.
        let distance_tolerance = length_unit.0 * 1e-5;

        // Compose transforms up the hierarchy to the PhysicsEnvironment ancestor,
        // staying in f64 to preserve precision at large distances.
        let (composed_translation, composed_rotation) =
            compose_to_physics_world(transform, child_of, &ancestors);

        let transform_translation = composed_translation;
        let transform_rotation = Rotation::from(composed_rotation);

        let position_changed = !position.is_added()
            && is_changed_after_tick(
                Ref::from(position.reborrow()),
                last_physics_tick.0,
                this_run,
            );
        if !position_changed && position.abs_diff_ne(&transform_translation, distance_tolerance) {
            position.0 = transform_translation;
        }

        let rotation_changed = !rotation.is_added()
            && is_changed_after_tick(
                Ref::from(rotation.reborrow()),
                last_physics_tick.0,
                this_run,
            );
        if !rotation_changed
            && rotation.angle_between(transform_rotation).abs() > rotation_tolerance
        {
            *rotation = transform_rotation;
        }
    }
}

/// Composes [`Transform`]s from the entity up to (but not including) the nearest
/// [`PhysicsEnvironment`] ancestor, returning the accumulated translation and rotation.
///
/// This avoids reading [`GlobalTransform`] (which suffers f32 precision loss at
/// large distances) and instead walks the hierarchy composing local transforms.
fn compose_to_physics_world(
    entity_transform: &Transform,
    entity_child_of: Option<&ChildOf>,
    ancestors: &Query<(&Transform, Option<&ChildOf>, Has<PhysicsEnvironment>), Without<Position>>,
) -> (Vector, Quat) {
    // Start with the entity's own transform.
    #[cfg(feature = "2d")]
    let mut translation = entity_transform.translation.truncate();
    #[cfg(feature = "3d")]
    let mut translation = entity_transform.translation;
    let mut rotation: Quat = entity_transform.rotation;

    // Walk up to the PhysicsEnvironment, composing ancestor transforms.
    let Some(&ChildOf(mut current_parent)) = entity_child_of else {
        return (translation, rotation);
    };

    loop {
        let Ok((parent_transform, parent_child_of, is_physics_world)) =
            ancestors.get(current_parent)
        else {
            break;
        };
        // Stop at the PhysicsEnvironment — it defines the coordinate origin.
        if is_physics_world {
            break;
        }
        // Compose: position_in_parent_space = parent_transform * child_position
        let parent_rot: Quat = parent_transform.rotation;
        #[cfg(feature = "2d")]
        {
            let parent_translation = parent_transform.translation.truncate();
            let parent_scale = parent_transform.scale.truncate();
            translation = (parent_rot * (parent_scale * translation).extend(0.0)).truncate()
                + parent_translation;
        }
        #[cfg(feature = "3d")]
        {
            let parent_translation = parent_transform.translation;
            let parent_scale = parent_transform.scale;
            translation = parent_rot * (parent_scale * translation) + parent_translation;
        }
        rotation = parent_rot * rotation;

        // Move to the next ancestor.
        match parent_child_of {
            Some(&ChildOf(grandparent)) => current_parent = grandparent,
            None => break,
        }
    }

    (translation, rotation)
}

/// Marker component indicating that the `position_to_transform` system should be applied
/// to this entity.
///
/// By default, the `position_to_transform` system only runs for entities that have a
/// [`RigidBody`] component
#[derive(Component, Default)]
pub struct ApplyPosToTransform;

type PosToTransformComponents = (
    &'static mut Transform,
    &'static Position,
    &'static Rotation,
    Option<&'static ChildOf>,
);

type PosToTransformFilter = (
    Or<(With<RigidBody>, With<ApplyPosToTransform>)>,
    Or<(Changed<Position>, Changed<Rotation>)>,
);

/// Ancestors whose frame a body's [`Position`] is relative to: everything up to the nearest
/// [`PhysicsEnvironment`], stopping at one that has a [`Position`] of its own. The same walk
/// as [`compose_to_physics_world`], so the two directions agree exactly.
type FrameAncestors<'w, 's> = Query<
    'w,
    's,
    (
        &'static Transform,
        Option<&'static ChildOf>,
        Has<PhysicsEnvironment>,
    ),
    Without<Position>,
>;

/// The ancestors' transforms a child's [`Position`] composes through, nearest first (the
/// walk [`compose_to_physics_world`] takes past the child itself).
fn frame_chain(child_of: Option<&ChildOf>, ancestors: &FrameAncestors, chain: &mut Vec<Transform>) {
    chain.clear();
    let Some(&ChildOf(mut current)) = child_of else {
        return;
    };
    while let Ok((transform, child_of, is_physics_world)) = ancestors.get(current) {
        if is_physics_world {
            break;
        }
        chain.push(*transform);
        match child_of {
            Some(&ChildOf(next)) => current = next,
            None => break,
        }
    }
}

/// Copies [`Position`] and [`Rotation`] changes to [`Transform`].
/// This allows users and the engine to use these components for moving and positioning bodies.
///
/// The exact inverse of [`transform_to_position`]: the ancestors' transforms are undone from
/// the outermost in, so a body under any chain of plain entities (a scene root under a placed
/// tile, a prop under a room) lands where its [`Position`] says. Reading a parent's local
/// `Transform` as its frame, as this once did, misplaced such bodies every tick and
/// `transform_to_position` pulled them back the next.
#[cfg(feature = "2d")]
pub fn position_to_transform(
    mut query: Query<PosToTransformComponents, PosToTransformFilter>,
    ancestors: FrameAncestors,
    mut chain: Local<Vec<Transform>>,
) {
    for (mut transform, pos, rot, child_of) in &mut query {
        frame_chain(child_of, &ancestors, &mut chain);
        let mut translation = pos.f32();
        let mut rotation = Quat::from(*rot);
        for parent in chain.iter().rev() {
            let inv = parent.rotation.inverse();
            translation = (inv * (translation - parent.translation.truncate()).extend(0.0))
                .truncate()
                / parent.scale.truncate();
            rotation = inv * rotation;
        }
        transform.translation = translation.extend(transform.translation.z);
        transform.rotation = rotation;
    }
}

/// Copies [`Position`] and [`Rotation`] changes to [`Transform`].
/// This allows users and the engine to use these components for moving and positioning bodies.
///
/// The exact inverse of [`transform_to_position`]: the ancestors' transforms are undone from
/// the outermost in, so a body under any chain of plain entities (a scene root under a placed
/// tile, a prop under a room) lands where its [`Position`] says. Reading a parent's local
/// `Transform` as its frame, as this once did, misplaced such bodies every tick and
/// `transform_to_position` pulled them back the next.
#[cfg(feature = "3d")]
pub fn position_to_transform(
    mut query: Query<PosToTransformComponents, PosToTransformFilter>,
    ancestors: FrameAncestors,
    mut chain: Local<Vec<Transform>>,
) {
    for (mut transform, pos, rot, child_of) in &mut query {
        frame_chain(child_of, &ancestors, &mut chain);
        let mut translation = pos.f32();
        let mut rotation = rot.0;
        for parent in chain.iter().rev() {
            let inv = parent.rotation.inverse();
            translation = (inv * (translation - parent.translation)) / parent.scale;
            rotation = inv * rotation;
        }
        transform.translation = translation;
        transform.rotation = rotation;
    }
}
