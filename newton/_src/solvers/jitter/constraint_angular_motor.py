 public struct AngularMotorData
 {
     internal int _internal;
     internal uint DispatchId;
     internal ulong ConstraintId;

     public JHandle<RigidBodyData> Body1;
     public JHandle<RigidBodyData> Body2;

     public JVector LocalAxis1;
     public JVector LocalAxis2;

     public Real Velocity;
     public Real MaxForce;
     public Real MaxLambda;

     public Real EffectiveMass;

     public Real AccumulatedImpulse;
 }

  public void Initialize(JVector axis1, JVector axis2)
 {
     VerifyNotZero();
     ref AngularMotorData data = ref Data;
     ref RigidBodyData body1 = ref data.Body1.Data;
     ref RigidBodyData body2 = ref data.Body2.Data;

     JVector.NormalizeInPlace(ref axis1);
     JVector.NormalizeInPlace(ref axis2);

     JVector.ConjugatedTransform(axis1, body1.Orientation, out data.LocalAxis1);
     JVector.ConjugatedTransform(axis2, body2.Orientation, out data.LocalAxis2);

     data.MaxForce = 0;
     data.Velocity = 0;
 }

 public static void PrepareForIterationAngularMotor(ref ConstraintData constraint, Real idt)
{
    ref var data = ref Unsafe.As<ConstraintData, AngularMotorData>(ref constraint);

    ref RigidBodyData body1 = ref data.Body1.Data;
    ref RigidBodyData body2 = ref data.Body2.Data;

    JVector.Transform(data.LocalAxis1, body1.Orientation, out JVector j1);
    JVector.Transform(data.LocalAxis2, body2.Orientation, out JVector j2);

    data.EffectiveMass = JVector.Transform(j1, body1.InverseInertiaWorld) * j1 +
                         JVector.Transform(j2, body2.InverseInertiaWorld) * j2;
    data.EffectiveMass = (Real)1.0 / data.EffectiveMass;

    data.MaxLambda = (Real)1.0 / idt * data.MaxForce;

    body1.AngularVelocity -= JVector.Transform(j1 * data.AccumulatedImpulse, body1.InverseInertiaWorld);
    body2.AngularVelocity += JVector.Transform(j2 * data.AccumulatedImpulse, body2.InverseInertiaWorld);
}

public static void IterateAngularMotor(ref ConstraintData constraint, Real idt)
{
    ref var data = ref Unsafe.As<ConstraintData, AngularMotorData>(ref constraint);

    ref RigidBodyData body1 = ref constraint.Body1.Data;
    ref RigidBodyData body2 = ref constraint.Body2.Data;

    JVector.Transform(data.LocalAxis1, body1.Orientation, out JVector j1);
    JVector.Transform(data.LocalAxis2, body2.Orientation, out JVector j2);

    Real jv = -j1 * body1.AngularVelocity + j2 * body2.AngularVelocity;

    Real lambda = -(jv - data.Velocity) * data.EffectiveMass;

    Real oldAccumulated = data.AccumulatedImpulse;

    data.AccumulatedImpulse += lambda;

    data.AccumulatedImpulse = Math.Clamp(data.AccumulatedImpulse, -data.MaxLambda, data.MaxLambda);

    lambda = data.AccumulatedImpulse - oldAccumulated;

    body1.AngularVelocity -= JVector.Transform(j1 * lambda, body1.InverseInertiaWorld);
    body2.AngularVelocity += JVector.Transform(j2 * lambda, body2.InverseInertiaWorld);
}