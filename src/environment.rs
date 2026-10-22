//! Physics environment components and management.
//!
//! A [`PhysicsEnvironment`] entity holds all per-environment physics state as components
//! (e.g. [`Gravity`], [`SolverConfig`], [`ColliderTrees`]).
//!
//! Physics entities are assigned to an environment by being descendants of a
//! [`PhysicsEnvironment`] entity in the hierarchy, and every one must have such an ancestor:
//! there is no default environment. One without is an error (logged once) and is not
//! simulated. The environment is also the frame: [`Position`] is relative to it.

use bevy::{
    ecs::{
        entity_disabling::Disabled, relationship::Relationship, system::SystemParam,
        world::DeferredWorld,
    },
    prelude::*,
};

#[cfg(feature = "bevy_diagnostic")]
use crate::diagnostics::{PhysicsEntityDiagnostics, PhysicsTotalDiagnostics};
use crate::{
    collider_tree::{
        ColliderTreeDiagnostics, ColliderTreeProxy, ColliderTreeProxyFlags, ColliderTreeProxyKey,
        ColliderTreeType, ColliderTrees, EnlargedProxies, MovedProxies,
        optimization::OptimizationTasks, update::LastDynamicKinematicAabbUpdate,
    },
    collision::{
        CollisionDiagnostics,
        narrow_phase::{
            NarrowPhaseConfig, remove_collider,
            system_param::{ContactStatusBits, ContactStatusChangeQueue},
        },
    },
    dynamics::joints::joint_graph::JointGraph,
    dynamics::{
        rigid_body::{DefaultFriction, DefaultRestitution},
        solver::{
            ContactConstraints, ContactSoftnessCoefficients, SolverConfig, SolverDiagnostics,
            constraint_graph::ConstraintGraph,
            islands::{
                BodyIslandNode, IslandId, PhysicsIslands,
                sleeping::{AwakeIslandBitVec, WakeIslands},
            },
        },
    },
    prelude::*,
    spatial_query::SpatialQueryDiagnostics,
};

/// A physics environment entity that holds per-environment physics state as components.
///
/// Each per-environment component is added via `#[require]` so that
/// spawning a `PhysicsEnvironment` automatically initializes all state.
///
/// Physics entities are assigned to an environment by being descendants
/// of a `PhysicsEnvironment` entity in the bevy hierarchy. Use
/// [`PhysicsEnvironmentLookup`] to resolve which environment an entity belongs to.
#[derive(Component, Default, Reflect)]
#[reflect(Component, Default)]
#[require(
    Transform,
    Visibility,
    Gravity,
    PhysicsLengthUnit,
    SolverConfig,
    ContactSoftnessCoefficients,
    NarrowPhaseConfig,
    DefaultFriction,
    DefaultRestitution,
    TimeToSleep,
    ColliderTrees,
    MovedProxies,
    EnlargedProxies,
    ContactGraph,
    JointGraph,
    // TODO: ContactConstraints was previously conditionally initialized via
    // NarrowPhasePlugin::generate_constraints. Now always required.
    // May need a flag to disable constraint generation for sensor-only environments.
    ContactConstraints,
    ConstraintGraph,
    PhysicsIslands,
    AwakeIslandBitVec,
    ContactStatusBits,
    ContactStatusChangeQueue,
    ColliderTreeOptimization,
    OptimizationTasks,
    LastDynamicKinematicAabbUpdate,
    CollisionDiagnostics,
    SolverDiagnostics,
    ColliderTreeDiagnostics,
    SpatialQueryDiagnostics,
)]
#[cfg_attr(
    feature = "bevy_diagnostic",
    require(PhysicsTotalDiagnostics, PhysicsEntityDiagnostics)
)]
pub struct PhysicsEnvironment;

// --- Cached physics environment assignment ---

/// A cached reference to the [`PhysicsEnvironment`] entity that a physics entity belongs to.
///
/// This is populated automatically by walking the hierarchy on spawn (via the
/// [`BodyIslandNode`] `on_add` hook) and updated directly by [`TransferToEnvironment`].
///
/// [`PhysicsEnvironmentLookup`] reads this cache for O(1) lookup instead of
/// walking the [`ChildOf`] hierarchy every frame.
#[derive(Component, Clone, Copy, Debug, PartialEq, Eq, Reflect)]
#[reflect(Component)]
pub struct PhysicsEnvironmentEntity(pub Entity);

// --- Hierarchy-based environment lookup ---

/// Finds the [`PhysicsEnvironment`] entity for the given entity.
///
/// Checks the [`PhysicsEnvironmentEntity`] cache first (O(1)), then falls back
/// to walking the [`ChildOf`] hierarchy. Works in [`DeferredWorld`] (hooks).
///
/// Returns `None` if no `PhysicsEnvironment` is found.
pub fn find_physics_environment_in_hierarchy(
    world: &DeferredWorld,
    entity: Entity,
) -> Option<Entity> {
    // Fast path: cached environment assignment (set by BodyIslandNode::on_add and TransferToEnvironment).
    if let Some(cached) = world.get::<PhysicsEnvironmentEntity>(entity) {
        return Some(cached.0);
    }
    // Slow path: walk hierarchy.
    let mut current = entity;
    loop {
        if world.get::<PhysicsEnvironment>(current).is_some() {
            return Some(current);
        }
        current = world.get::<ChildOf>(current)?.get();
    }
}

/// Logs a physics entity that has no [`PhysicsEnvironment`] ancestor: it is not simulated.
/// Called once, where its physics state would have been created.
pub(crate) fn report_missing_environment(world: &DeferredWorld, entity: Entity, what: &str) {
    let name = world
        .get::<Name>(entity)
        .map_or_else(|| "unnamed".to_string(), |name| name.as_str().to_string());
    error!(
        "{what} {entity} ({name}) is in no PhysicsEnvironment: spawn it under one; it is not simulated"
    );
}

/// A [`SystemParam`] that resolves which [`PhysicsEnvironment`] entity a given entity belongs to:
/// its nearest [`PhysicsEnvironment`] ancestor, or `None` (it is then not simulated).
#[derive(SystemParam)]
pub struct PhysicsEnvironmentLookup<'w, 's> {
    cache: Query<'w, 's, &'static PhysicsEnvironmentEntity>,
    parents: Query<'w, 's, &'static ChildOf>,
    worlds: Query<'w, 's, (), With<PhysicsEnvironment>>,
}

impl PhysicsEnvironmentLookup<'_, '_> {
    /// Returns the [`PhysicsEnvironment`] entity for the given entity, `None` when it has no
    /// environment ancestor.
    ///
    /// Uses the [`PhysicsEnvironmentEntity`] cache if present (O(1)),
    /// otherwise walks up the [`ChildOf`] hierarchy.
    pub fn environment_of(&self, entity: Entity) -> Option<Entity> {
        // Fast path: cached environment assignment.
        if let Ok(cached) = self.cache.get(entity) {
            return Some(cached.0);
        }
        // Slow path: walk hierarchy.
        let mut current = entity;
        loop {
            if self.worlds.contains(current) {
                return Some(current);
            }
            current = self.parents.get(current).ok()?.get();
        }
    }
}

// --- Transfer between physics environments ---

/// An [`EntityEvent`] that transfers a physics entity to a different [`PhysicsEnvironment`].
///
/// This re-parents the entity and handles all physics cleanup/re-initialization:
/// solver bodies, islands, collider tree proxies, and contacts are properly
/// migrated from the old environment to the new one.
///
/// # Reframing
///
/// The body's **velocity** is reframed into the destination frame (subtract the
/// destination frame's velocity, add the source frame's) so a body keeping pace with the
/// source frame arrives keeping pace with the destination — see [`on_transfer_to_environment`].
/// Its **pose** keeps the same frame-local `Transform` (correct for co-located frames,
/// e.g. a docked airlock). Full environment-position-preserving pose decomposition for
/// spatially-offset frames is a refinement. Completion is announced via [`EnvironmentTransferred`].
///
/// # Example
///
/// ```ignore
/// // Transfer player from ship interior to planet surface
/// commands.trigger(TransferToEnvironment {
///     entity: player_entity,
///     environment: planet_world,
/// });
/// ```
#[derive(EntityEvent, Clone, Copy, Debug)]
pub struct TransferToEnvironment {
    /// The entity to transfer.
    pub entity: Entity,
    /// The [`PhysicsEnvironment`] entity to transfer to.
    pub world: Entity,
}

/// Fired by [`on_transfer_to_environment`] **after** a [`TransferToEnvironment`] completes — the
/// notification that velocity reframing and any external rebase bind to (e.g. solari's
/// rotational rebase of the previous view-projection when the camera crosses frames).
///
/// Carries **both** frame anchors explicitly, so observers never race the environment-cache
/// update inside the transfer handler. Read each anchor's `LinearVelocity`/`AngularVelocity`
/// (and, for a renderer, its world rotation) off the `from`/`to` entities to compute the
/// reframe — the kinematic state lives on the frames, keeping this event pure routing.
#[derive(EntityEvent, Clone, Copy, Debug)]
pub struct EnvironmentTransferred {
    /// The entity that was transferred.
    pub entity: Entity,
    /// The [`PhysicsEnvironment`] it came from.
    pub from: Entity,
    /// The [`PhysicsEnvironment`] it now belongs to.
    pub to: Entity,
}

/// Handles [`TransferToEnvironment`] events.
///
/// Directly manipulates physics state via queries — no deferred SolverBody
/// remove/re-add, so no hook ordering issues and no exclusive environment lock.
///
/// 1. Removes collider proxy from old environment's [`ColliderTrees`]
/// 2. Unlinks body from old environment's [`PhysicsIslands`]
/// 3. Re-parents entity via deferred [`ChildOf`] insert
/// 4. Creates a new island in the target environment's [`PhysicsIslands`]
/// 5. Re-registers collider proxy via deferred [`ColliderOf`] re-insert
/// 6. Wakes sleeping islands in the target environment
#[derive(SystemParam)]
#[allow(clippy::type_complexity)]
struct TransferParams<'w, 's> {
    collider_keys: Query<
        'w,
        's,
        (
            &'static mut ColliderTreeProxyKey,
            Option<&'static RigidBodyColliders>,
            Option<&'static ColliderOf>,
            &'static ColliderAabb,
            Option<&'static CollisionLayers>,
            Has<Sensor>,
            Has<CollisionEventsEnabled>,
            Option<&'static ActiveCollisionHooks>,
        ),
    >,
    bodies: Query<'w, 's, (&'static RigidBody, Has<RigidBodyDisabled>)>,
    body_islands: Query<'w, 's, &'static mut BodyIslandNode>,
    contact_graphs: Query<
        'w,
        's,
        (&'static mut ContactGraph, &'static mut ContactStatusChangeQueue),
        With<PhysicsEnvironment>,
    >,
    colliding_entities: Query<'w, 's, &'static mut CollidingEntities, Allow<Disabled>>,
    collision_ends: MessageWriter<'w, CollisionEnd>,
    sleep_timers: Query<'w, 's, &'static mut crate::dynamics::rigid_body::sleeping::SleepTimer>,
    world_cache: Query<'w, 's, &'static mut PhysicsEnvironmentEntity>,
    parents: Query<'w, 's, &'static ChildOf>,
    physics_worlds: Query<'w, 's, (), With<PhysicsEnvironment>>,
    world_state: Query<
        'w,
        's,
        (
            &'static mut ColliderTrees,
            &'static mut MovedProxies,
            &'static mut PhysicsIslands,
            &'static mut AwakeIslandBitVec,
        ),
        With<PhysicsEnvironment>,
    >,
    // Frame anchors' own motion (kinematic movers like a flying ship); static frames
    // simply lack these. `With`/`Without<PhysicsEnvironment>` keeps the two velocity queries
    // archetype-disjoint, so the mutable body query doesn't conflict with the frame read.
    frame_vels: Query<
        'w,
        's,
        (
            Option<&'static LinearVelocity>,
            Option<&'static AngularVelocity>,
        ),
        With<PhysicsEnvironment>,
    >,
    body_vels: Query<
        'w,
        's,
        (&'static mut LinearVelocity, &'static mut AngularVelocity),
        Without<PhysicsEnvironment>,
    >,
    commands: Commands<'w, 's>,
}

impl TransferParams<'_, '_> {
    // Cached environment, falling back to a hierarchy walk.
    fn current_world(&self, entity: Entity) -> Option<Entity> {
        if let Ok(cached) = self.world_cache.get(entity) {
            return Some(cached.0);
        }
        let mut current = entity;
        loop {
            if self.physics_worlds.contains(current) {
                return Some(current);
            }
            current = self.parents.get(current).ok()?.get();
        }
    }
}

fn on_transfer_to_environment(trigger: On<TransferToEnvironment>, mut p: TransferParams) {
    let entity = trigger.event_target();
    let Some(from) = p.current_world(entity) else {
        error!("TransferToEnvironment: {entity} is in no PhysicsEnvironment to transfer from");
        return;
    };
    transfer(&mut p, entity, from, trigger.event().world, true);
}

/// Moves an entity's physics state between environments after its hierarchy already changed.
#[derive(EntityEvent, Clone, Copy, Debug)]
pub(crate) struct RehomeToEnvironment {
    entity: Entity,
    from: Entity,
    to: Entity,
}

fn on_rehome_to_environment(trigger: On<RehomeToEnvironment>, mut p: TransferParams) {
    let event = trigger.event();
    transfer(&mut p, event.entity, event.from, event.to, false);
}

fn transfer(
    p: &mut TransferParams,
    entity: Entity,
    old_world: Entity,
    target_world: Entity,
    reparent: bool,
) {
    let TransferParams {
        collider_keys,
        bodies,
        body_islands,
        contact_graphs,
        colliding_entities,
        collision_ends,
        sleep_timers,
        world_cache,
        world_state,
        frame_vels,
        body_vels,
        commands,
        ..
    } = p;

    if old_world == target_world {
        return; // Already in the target world.
    }

    // Update the cached environment assignment immediately.
    if let Ok(mut cached) = world_cache.get_mut(entity) {
        cached.0 = target_world;
    }

    // --- 1. Remove collider proxy from old environment's tree ---
    // Collect child collider entities to avoid borrow conflicts.
    let child_colliders_for_remove: Vec<Entity> = collider_keys
        .get(entity)
        .ok()
        .and_then(|(_, body_colliders, ..)| body_colliders.map(|c| c.iter().collect()))
        .unwrap_or_default();

    remove_proxy_from_world(entity, old_world, collider_keys, world_state);
    for &collider in &child_colliders_for_remove {
        remove_proxy_from_world(collider, old_world, collider_keys, world_state);
    }

    // Its contact pairs leave the old environment too: left there, the old narrow phase keeps
    // updating them, and one that starts touching links the body's island id (by then the new
    // environment's) into the old environment's islands. The queued status changes unlink the
    // touching ones from the old islands and constraint graph at the next flush.
    if let Ok((mut contact_graph, mut status_changes)) = contact_graphs.get_mut(old_world) {
        for collider in core::iter::once(entity).chain(child_colliders_for_remove) {
            remove_collider(
                collider,
                &mut contact_graph,
                &mut status_changes,
                colliding_entities,
                collision_ends,
            );
        }
    }

    // --- 2. Unlink body from old environment's island ---
    let mut old_island_id = IslandId::PLACEHOLDER;
    let mut old_island_survived = false;

    if let Ok(body_island) = body_islands.get(entity) {
        old_island_id = body_island.island_id;
        let prev = body_island.prev;
        let next = body_island.next;

        // Fix the linked list in the old island.
        if let Some(prev_entity) = prev {
            if let Ok(mut prev_node) = body_islands.get_mut(prev_entity) {
                prev_node.next = next;
            }
        }
        if let Some(next_entity) = next {
            if let Ok(mut next_node) = body_islands.get_mut(next_entity) {
                next_node.prev = prev;
            }
        }

        // Update the old environment's island.
        if let Ok((_, _, mut old_islands, _)) = world_state.get_mut(old_world) {
            if let Some(island) = old_islands.get_mut(old_island_id) {
                island.body_count = island.body_count.saturating_sub(1);
                if island.head_body == Some(entity) {
                    island.head_body = next;
                }
                if island.tail_body == Some(entity) {
                    island.tail_body = prev;
                }
                if island.body_count == 0 && island.contact_count == 0 {
                    old_islands.remove_island(old_island_id);
                } else {
                    old_island_survived = true;
                }
            }
        }

        // --- 4. Create a new island in the target environment and update the body's node ---
        let mut new_island_id = IslandId::PLACEHOLDER;
        if let Ok((_, _, mut new_islands, mut awake_bits)) = world_state.get_mut(target_world) {
            new_island_id = new_islands.create_island_with(|island| {
                island.head_body = Some(entity);
                island.tail_body = Some(entity);
                island.body_count = 1;
            });
            // Mark the new island as awake so sleep_islands doesn't immediately re-sleep it.
            awake_bits.set_and_grow(new_island_id.0 as usize);
        }
        if let Ok(mut body_island) = body_islands.get_mut(entity) {
            body_island.island_id = new_island_id;
            body_island.prev = None;
            body_island.next = None;
        }
    }

    // --- 3. Re-parent to new environment (deferred — only affects hierarchy lookup) ---
    if reparent {
        commands.entity(entity).insert(ChildOf(target_world));
    }

    // --- 5. Add collider proxy to new environment's tree (direct) ---
    // Collect child collider entities first to avoid borrow conflicts.
    let child_colliders: Vec<Entity> = collider_keys
        .get(entity)
        .ok()
        .and_then(|(_, body_colliders, ..)| body_colliders.map(|c| c.iter().collect()))
        .unwrap_or_default();

    add_proxy_to_world(entity, target_world, collider_keys, bodies, world_state);
    for collider in child_colliders {
        add_proxy_to_world(collider, target_world, collider_keys, bodies, world_state);
    }

    // --- 6. Wake the transferred entity and affected islands ---
    // Remove Sleeping and reset sleep timer so physics processes the entity.
    commands.entity(entity).try_remove::<Sleeping>();
    if let Ok(mut timer) = sleep_timers.get_mut(entity) {
        timer.0 = 0.0;
    }

    // Wake the old island so remaining bodies react to the gap.
    // WakeIslands handles setting is_sleeping=false AND removing Sleeping from bodies.
    if old_island_survived {
        commands.queue(WakeIslands {
            world_entity: old_world,
            islands: vec![old_island_id],
        });
    }

    // Wake all sleeping islands in the destination environment.
    let new_to_wake: Vec<_> = world_state
        .get(target_world)
        .ok()
        .map(|(_, _, islands, _)| {
            islands
                .iter()
                .filter(|i| i.is_sleeping)
                .map(|i| i.id)
                .collect()
        })
        .unwrap_or_default();
    if !new_to_wake.is_empty() {
        commands.queue(WakeIslands {
            world_entity: target_world,
            islands: new_to_wake,
        });
    }

    // --- 7. Reframe velocity into the destination frame ---
    // A body keeping pace with the source frame must arrive keeping pace with the
    // destination frame, not hurtling: subtract the destination frame's velocity, add the
    // source frame's. Frame anchors carry these only when they are kinematic movers (the
    // "treadmill" ship); static frames default to zero.
    //
    // NOTE: this is the translational (+ spin-about-anchor) reframe. The ω×r lever-arm
    // between rotating *offset* frames is a refinement for when such frames are used.
    let frame_lin = |w: Entity| {
        frame_vels
            .get(w)
            .ok()
            .and_then(|(l, _)| l)
            .map_or(LinearVelocity::default().0, |l| l.0)
    };
    let frame_ang = |w: Entity| {
        frame_vels
            .get(w)
            .ok()
            .and_then(|(_, a)| a)
            .map_or(AngularVelocity::default().0, |a| a.0)
    };
    if reparent && let Ok((mut lin, mut ang)) = body_vels.get_mut(entity) {
        lin.0 += frame_lin(old_world) - frame_lin(target_world);
        ang.0 += frame_ang(old_world) - frame_ang(target_world);
    }

    // --- 8. Notify observers (avian-side + renderer) of the completed handoff ---
    // Both frames explicit so consumers (e.g. solari's rotational rebase of the previous
    // view-projection) never race the environment-cache update above.
    commands.trigger(EnvironmentTransferred {
        entity,
        from: old_world,
        to: target_world,
    });
}

/// Removes an entity's collider proxy from an environment's [`ColliderTrees`].
#[allow(clippy::type_complexity)]
fn remove_proxy_from_world(
    entity: Entity,
    world_entity: Entity,
    collider_keys: &mut Query<(
        &mut ColliderTreeProxyKey,
        Option<&RigidBodyColliders>,
        Option<&ColliderOf>,
        &ColliderAabb,
        Option<&CollisionLayers>,
        Has<Sensor>,
        Has<CollisionEventsEnabled>,
        Option<&ActiveCollisionHooks>,
    )>,
    world_state: &mut Query<
        (
            &mut ColliderTrees,
            &mut MovedProxies,
            &mut PhysicsIslands,
            &mut AwakeIslandBitVec,
        ),
        With<PhysicsEnvironment>,
    >,
) {
    let Ok((proxy_key, ..)) = collider_keys.get(entity) else {
        return;
    };
    if *proxy_key == ColliderTreeProxyKey::PLACEHOLDER {
        return;
    }
    let Ok((mut trees, mut moved_proxies, _, _)) = world_state.get_mut(world_entity) else {
        return;
    };
    let tree = trees.tree_for_type_mut(proxy_key.tree_type());
    tree.remove_proxy(proxy_key.id());
    moved_proxies.remove(proxy_key);
}

/// Adds an entity's collider proxy to an environment's [`ColliderTrees`].
#[allow(clippy::type_complexity)]
fn add_proxy_to_world(
    entity: Entity,
    world_entity: Entity,
    collider_keys: &mut Query<(
        &mut ColliderTreeProxyKey,
        Option<&RigidBodyColliders>,
        Option<&ColliderOf>,
        &ColliderAabb,
        Option<&CollisionLayers>,
        Has<Sensor>,
        Has<CollisionEventsEnabled>,
        Option<&ActiveCollisionHooks>,
    )>,
    bodies: &Query<(&RigidBody, Has<RigidBodyDisabled>)>,
    world_state: &mut Query<
        (
            &mut ColliderTrees,
            &mut MovedProxies,
            &mut PhysicsIslands,
            &mut AwakeIslandBitVec,
        ),
        With<PhysicsEnvironment>,
    >,
) {
    let Ok((
        mut proxy_key,
        _,
        collider_of,
        aabb,
        layers,
        is_sensor,
        has_contact_events,
        active_hooks,
    )) = collider_keys.get_mut(entity)
    else {
        return;
    };

    let (tree_type, is_body_disabled) =
        if let Some(Ok((rb, disabled))) = collider_of.map(|c| bodies.get(c.body)) {
            (ColliderTreeType::from_body(Some(*rb)), disabled)
        } else {
            (ColliderTreeType::Standalone, false)
        };

    let proxy = ColliderTreeProxy {
        collider: entity,
        body: collider_of.map(|c| c.body),
        layers: layers.copied().unwrap_or_default(),
        flags: ColliderTreeProxyFlags::new(
            is_sensor,
            is_body_disabled,
            has_contact_events,
            active_hooks.copied().unwrap_or_default(),
        ),
    };

    let Ok((mut trees, mut moved_proxies, _, _)) = world_state.get_mut(world_entity) else {
        return;
    };

    let tree = trees.tree_for_type_mut(tree_type);
    let proxy_id = tree.add_proxy((*aabb).into(), proxy);
    let new_key = ColliderTreeProxyKey::new(proxy_id, tree_type);
    *proxy_key = new_key;
    moved_proxies.insert(new_key);
}

// --- Re-homing after hierarchy changes ---

/// Entities whose parent changed since the last drain.
#[derive(Resource, Default)]
struct PendingRehome(Vec<Entity>);

fn queue_rehome(entity: Entity, mut pending: ResMut<PendingRehome>, mut commands: Commands) {
    if pending.0.is_empty() {
        commands.queue(drain_rehome);
    }
    pending.0.push(entity);
}

// Bodies and standalone colliders under a reparented entity follow it to its new environment.
fn drain_rehome(world: &mut World) {
    let roots = core::mem::take(&mut world.resource_mut::<PendingRehome>().0);
    let mut worlds = world.query_filtered::<(Entity, &ColliderTrees), With<PhysicsEnvironment>>();
    worlds.update_archetypes(world);
    let mut seen = bevy::ecs::entity::EntityHashSet::default();
    let mut moves = Vec::new();
    // Physics entities that were in no environment and now are: created again, so their
    // hooks give them an island / a collider proxy in it.
    let mut adopt = Vec::new();
    for root in roots {
        if world.get_entity(root).is_err() || world.get::<PhysicsEnvironment>(root).is_some() {
            continue;
        }
        // Moved out of every environment: its physics state stays where it was (and is
        // reported where it is created next); only moves between environments rehome.
        let Some(target) = hierarchy_world(world, root) else {
            continue;
        };
        let mut stack = vec![(root, target)];
        while let Some((entity, target)) = stack.pop() {
            if !seen.insert(entity) || world.get::<PhysicsEnvironment>(entity).is_some() {
                continue;
            }
            match physics_home(world, &mut worlds, entity) {
                Some(from) if from != target => moves.push((entity, from, target)),
                Some(_) => {}
                None if is_orphan(world, entity) => adopt.push(entity),
                None => {}
            }
            if let Some(children) = world.get::<Children>(entity) {
                stack.extend(children.iter().map(|child| (child, target)));
            }
        }
    }
    for (entity, from, to) in moves {
        world.trigger(RehomeToEnvironment { entity, from, to });
    }
    for entity in adopt {
        let mut entity = world.entity_mut(entity);
        if entity.contains::<BodyIslandNode>() {
            entity.remove::<BodyIslandNode>();
            entity.insert(BodyIslandNode::default());
        }
        if let Some(of) = entity.get::<ColliderOf>().copied() {
            entity.insert(of);
        }
    }
}

// Physics state that was never created because the entity had no environment: a body with
// no environment cached, or a collider with no proxy.
fn is_orphan(world: &World, entity: Entity) -> bool {
    let body = world.get::<BodyIslandNode>(entity).is_some()
        && world.get::<PhysicsEnvironmentEntity>(entity).is_none();
    let collider = world
        .get::<ColliderTreeProxyKey>(entity)
        .is_some_and(|key| *key == ColliderTreeProxyKey::PLACEHOLDER)
        && world.get::<ColliderOf>(entity).is_some();
    body || collider
}

fn hierarchy_world(world: &World, entity: Entity) -> Option<Entity> {
    let mut current = entity;
    loop {
        if world.get::<PhysicsEnvironment>(current).is_some() {
            return Some(current);
        }
        current = world.get::<ChildOf>(current)?.get();
    }
}

// The environment an entity's physics state lives in: a body's cache, or the tree holding a
// standalone collider's proxy. Colliders attached to another body move with it.
fn physics_home(
    world: &World,
    worlds: &mut QueryState<(Entity, &ColliderTrees), With<PhysicsEnvironment>>,
    entity: Entity,
) -> Option<Entity> {
    if world.get::<BodyIslandNode>(entity).is_some() {
        return world
            .get::<PhysicsEnvironmentEntity>(entity)
            .map(|cached| cached.0);
    }
    let key = *world.get::<ColliderTreeProxyKey>(entity)?;
    if key == ColliderTreeProxyKey::PLACEHOLDER
        || world
            .get::<ColliderOf>(entity)
            .is_some_and(|of| of.body != entity)
    {
        return None;
    }
    worlds
        .iter_manual(world)
        .find(|(_, trees)| {
            trees
                .get_proxy(key)
                .is_some_and(|proxy| proxy.collider == entity)
        })
        .map(|(world_entity, _)| world_entity)
}

/// Plugin that registers environment types and transfer handling. It spawns no environment:
/// the app spawns its own, and every physics entity goes under one.
pub struct PhysicsEnvironmentPlugin;

/// An environment and the settings it carries, for reflection: scenes save and editors show them.
pub fn register_environment_types(app: &mut App) {
    app.register_type::<PhysicsEnvironment>()
        .register_type::<Gravity>()
        .register_type::<PhysicsLengthUnit>()
        .register_type::<SolverConfig>()
        .register_type::<ContactSoftnessCoefficients>()
        .register_type::<NarrowPhaseConfig>()
        .register_type::<DefaultFriction>()
        .register_type::<DefaultRestitution>()
        .register_type::<TimeToSleep>();
}

impl Plugin for PhysicsEnvironmentPlugin {
    fn build(&self, app: &mut App) {
        register_environment_types(app);
        // Register the transfer observer.
        app.add_observer(on_transfer_to_environment)
            .add_observer(on_rehome_to_environment)
            .init_resource::<PendingRehome>()
            .add_observer(
                |trigger: On<Insert<ChildOf>>,
                 pending: ResMut<PendingRehome>,
                 commands: Commands| {
                    queue_rehome(trigger.entity, pending, commands);
                },
            )
            .add_observer(
                |trigger: On<Remove<ChildOf>>,
                 pending: ResMut<PendingRehome>,
                 commands: Commands| {
                    queue_rehome(trigger.entity, pending, commands);
                },
            );
    }
}
