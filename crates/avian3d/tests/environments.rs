use avian3d::collider_tree::ColliderTrees;
use avian3d::environment::{PhysicsEnvironment, TransferToEnvironment};
use avian3d::prelude::*;
use bevy::{prelude::*, time::TimeUpdateStrategy};
use core::time::Duration;

fn app() -> App {
    let mut app = App::new();
    app.add_plugins((
        MinimalPlugins,
        TransformPlugin,
        bevy::asset::AssetPlugin::default(),
        bevy::mesh::MeshPlugin,
        PhysicsPlugins::default(),
    ))
    .insert_resource(TimeUpdateStrategy::ManualDuration(Duration::from_secs_f32(
        1.0 / 60.0,
    )));
    app.finish();
    app
}

fn environment(app: &mut App) -> Entity {
    let env = app.world_mut().spawn(PhysicsEnvironment).id();
    app.world_mut().spawn((
        RigidBody::Static,
        Collider::cuboid(20.0, 1.0, 20.0),
        Transform::from_xyz(0.0, -0.5, 0.0),
        ChildOf(env),
    ));
    env
}

fn proxies_of(app: &App, env: Entity, collider: Entity) -> usize {
    app.world()
        .get::<ColliderTrees>(env)
        .unwrap()
        .iter_trees()
        .flat_map(|tree| tree.proxies.iter())
        .filter(|(_, proxy)| proxy.collider == collider)
        .count()
}

fn step(app: &mut App, frames: usize) {
    for _ in 0..frames {
        app.update();
    }
}

fn round_trip(own_collider: bool) {
    let mut app = app();
    let a = environment(&mut app);
    let b = environment(&mut app);
    let body = app
        .world_mut()
        .spawn((
            RigidBody::Dynamic,
            Transform::from_xyz(0.0, 0.5, 0.0),
            ChildOf(a),
        ))
        .id();
    let shape = (Collider::cuboid(1.0, 1.0, 1.0), Transform::default());
    let collider = if own_collider {
        app.world_mut().entity_mut(body).insert(shape.0);
        body
    } else {
        app.world_mut().spawn((shape, ChildOf(body))).id()
    };
    step(&mut app, 30);

    for (from, to) in [(a, b), (b, a), (a, b), (b, a)] {
        app.world_mut().trigger(TransferToEnvironment {
            entity: body,
            world: to,
        });
        step(&mut app, 30);
        assert_eq!(proxies_of(&app, to, collider), 1, "one proxy where it is");
        assert_eq!(proxies_of(&app, from, collider), 0, "none where it was");
        let left = app.world().get::<ContactGraph>(from).unwrap();
        assert_eq!(
            left.contact_pairs_with(collider).count(),
            0,
            "no pairs where it was"
        );
    }
}

// A body crosses to another environment and back, resting on each one's floor. A proxy left in
// the environment it left still names the body, and paired with that floor it links the body's
// island id (another environment's) into those islands: "Island IslandId(n) does not exist".
// The body's own collider is also in its `RigidBodyColliders`, and was added twice.
#[test]
fn round_trip_with_own_collider() {
    round_trip(true);
}

// Its colliders were read through the collider query, so a body without one of its own moved none.
#[test]
fn round_trip_with_child_collider() {
    round_trip(false);
}

// A prop's static collider sits under its root; despawning the root removes the collider after
// its ancestors are gone. Its proxy must still leave the tree, and nothing may rest on it after.
#[test]
fn despawned_prop_leaves_no_collider() {
    let mut app = app();
    let env = app.world_mut().spawn(PhysicsEnvironment).id();
    let prop = app
        .world_mut()
        .spawn((Transform::default(), ChildOf(env)))
        .id();
    let part = app
        .world_mut()
        .spawn((Transform::default(), ChildOf(prop)))
        .id();
    let collider = app
        .world_mut()
        .spawn((
            RigidBody::Static,
            Collider::cuboid(4.0, 1.0, 4.0),
            Transform::from_xyz(0.0, -0.5, 0.0),
            ChildOf(part),
        ))
        .id();
    let crate_ = app
        .world_mut()
        .spawn((
            RigidBody::Dynamic,
            Collider::cuboid(0.5, 0.5, 0.5),
            Transform::from_xyz(0.0, 0.3, 0.0),
            ChildOf(env),
        ))
        .id();
    step(&mut app, 60);
    assert_eq!(proxies_of(&app, env, collider), 1);
    let rest = app.world().get::<Position>(crate_).unwrap().y;

    app.world_mut().entity_mut(prop).despawn();
    step(&mut app, 60);
    assert!(app.world().get_entity(collider).is_err());
    assert_eq!(proxies_of(&app, env, collider), 0, "the prop's proxy is gone");
    let now = app.world().get::<Position>(crate_).unwrap().y;
    assert!(now < rest - 1.0, "the crate falls where the prop was: {rest} -> {now}");
}

/// A parent rotation that is not exactly unit (a scene file written to four decimals has norm
/// 1.00005) must not make a static child drift: the Transform -> Position -> Transform round trip
/// inverts the parent's rotation, and `Quat::inverse` is only the conjugate, so each pass gained
/// the quaternion's squared norm and a placed building's pieces climbed every physics step.
#[test]
fn non_unit_parent_rotation_does_not_drift_a_static_child() {
    let mut app = app();
    let env = environment(&mut app);
    let prop = app
        .world_mut()
        .spawn((
            Transform {
                translation: Vec3::new(-25.0, 0.0, 24.0),
                rotation: Quat::from_xyzw(0.0, 0.9239, 0.0, 0.3827),
                scale: Vec3::splat(0.6),
            },
            ChildOf(env),
        ))
        .id();
    assert!((app.world().get::<Transform>(prop).unwrap().rotation.length() - 1.0).abs() > 1e-5);
    let local = Vec3::new(0.02, 15.7, -0.23);
    let child = app
        .world_mut()
        .spawn((
            RigidBody::Static,
            Collider::cuboid(1.0, 1.0, 1.0),
            Transform::from_translation(local),
            ChildOf(prop),
        ))
        .id();
    step(&mut app, 240);
    let now = app.world().get::<Transform>(child).unwrap();
    assert!(
        now.translation.distance(local) < 1e-3,
        "the child drifted from {local} to {} in 240 steps",
        now.translation
    );
    assert!((now.rotation.length() - 1.0).abs() < 1e-4, "rotation norm {}", now.rotation.length());
}
