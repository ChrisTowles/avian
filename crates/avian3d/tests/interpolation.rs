//! The interpolation plugin in an app as tests build one.

use avian3d::prelude::*;
use bevy::prelude::*;

/// An app updated without `App::finish` (as a test's is, with no runner) has everything the
/// plugin's own systems read: the reset it runs before the fixed ticks reads the easing
/// plugin's `LastEasingTick`, which the interpolation plugins add only in `finish`.
#[test]
fn an_app_that_was_never_finished_updates() {
    let mut app = App::new();
    app.add_plugins((
        MinimalPlugins,
        TransformPlugin,
        bevy::asset::AssetPlugin::default(),
        bevy::mesh::MeshPlugin,
        PhysicsPlugins::default(),
    ));
    let environment = app
        .world_mut()
        .spawn((PhysicsEnvironment, Transform::default()))
        .id();
    app.world_mut().spawn((
        RigidBody::Kinematic,
        Collider::sphere(0.5),
        TransformInterpolation,
        ChildOf(environment),
    ));
    for _ in 0..3 {
        app.update();
    }
    // Finishing it afterwards adds the easing plugin no second time.
    app.finish();
    app.cleanup();
    app.update();
}
