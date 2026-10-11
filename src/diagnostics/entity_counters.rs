use bevy::{diagnostic::DiagnosticPath, prelude::*};

use crate::{ColliderMarker, PhysicsSchedule, PhysicsStepSystems, RigidBody, dynamics::joints::*};

use super::{AppDiagnosticsExt, PhysicsDiagnostics, impl_diagnostic_paths};

/// A plugin that adds diagnostics for physics entity counts.
pub struct PhysicsEntityDiagnosticsPlugin;

impl Plugin for PhysicsEntityDiagnosticsPlugin {
    fn build(&self, app: &mut App) {
        // Register diagnostics for physics entity counts.
        // NOTE: This should be done after the `PhysicsDiagnosticsPlugin` is added.
        app.register_physics_diagnostics::<PhysicsEntityDiagnostics>();

        app.add_systems(
            PhysicsSchedule,
            diagnostic_entity_counts.in_set(PhysicsStepSystems::First),
        );
    }
}

/// Diagnostics for physics entity counts.
#[derive(Component, Debug, Default, Reflect)]
#[reflect(Component, Debug)]
pub struct PhysicsEntityDiagnostics {
    /// The number of dynamic bodies.
    pub dynamic_body_count: u32,
    /// The number of kinematic bodies.
    pub kinematic_body_count: u32,
    /// The number of static bodies.
    pub static_body_count: u32,
    /// The number of colliders.
    pub collider_count: u32,
    /// The number of joint.
    pub joint_count: u32,
}

impl PhysicsDiagnostics for PhysicsEntityDiagnostics {
    fn counter_paths(&self) -> Vec<(&'static DiagnosticPath, u32)> {
        vec![
            (Self::DYNAMIC_BODY_COUNT, self.dynamic_body_count),
            (Self::KINEMATIC_BODY_COUNT, self.kinematic_body_count),
            (Self::STATIC_BODY_COUNT, self.static_body_count),
            (Self::COLLIDER_COUNT, self.collider_count),
            (Self::JOINT_COUNT, self.joint_count),
        ]
    }
}

impl_diagnostic_paths! {
    impl PhysicsEntityDiagnostics {
        DYNAMIC_BODY_COUNT: "avian/entity_count/dynamic_bodies",
        KINEMATIC_BODY_COUNT: "avian/entity_count/kinematic_bodies",
        STATIC_BODY_COUNT: "avian/entity_count/static_bodies",
        COLLIDER_COUNT: "avian/entity_count/colliders",
        JOINT_COUNT: "avian/entity_count/joints",
    }
}

fn diagnostic_entity_counts(
    rigid_bodies_query: Query<&RigidBody>,
    colliders_query: Query<&ColliderMarker>,
    fixed_joint_query: Query<&FixedJoint>,
    prismatic_joint_query: Query<&PrismaticJoint>,
    distance_joint_query: Query<&DistanceJoint>,
    revolute_joint_query: Query<&RevoluteJoint>,
    #[cfg(feature = "3d")] spherical_joint_query: Query<&SphericalJoint>,
    mut worlds: Query<&mut PhysicsEntityDiagnostics, With<crate::environment::PhysicsEnvironment>>,
) {
    // Counted over the whole app and shown on every environment.
    // Count the body types in a single pass.
    let (mut dynamic, mut kinematic, mut static_) = (0, 0, 0);
    for rb in &rigid_bodies_query {
        match rb {
            RigidBody::Dynamic => dynamic += 1,
            RigidBody::Kinematic => kinematic += 1,
            RigidBody::Static => static_ += 1,
        }
    }
    // These are archetypal queries, so counting them does not iterate the entities.
    let colliders = colliders_query.count() as u32;
    #[allow(unused_mut)]
    let mut joints = fixed_joint_query.count() as u32
        + prismatic_joint_query.count() as u32
        + distance_joint_query.count() as u32
        + revolute_joint_query.count() as u32;
    #[cfg(feature = "3d")]
    {
        joints += spherical_joint_query.count() as u32;
    }
    for mut diagnostics in &mut worlds {
        diagnostics.dynamic_body_count = dynamic;
        diagnostics.kinematic_body_count = kinematic;
        diagnostics.static_body_count = static_;
        diagnostics.collider_count = colliders;
        diagnostics.joint_count = joints;
    }
}
