import axisAngle from "../axisAngle";
import mat3 from "../dense/mat3";
import mat4 from "../dense/mat4";
import quat from "../quat";
import { assert, Tolerance, near, radians, SmallTolerance } from "../utils";
import transform3D from "../transform3D";
import transform2D from "../transform2D";
import vec2 from "../dense/vec2";
import vec3 from "../dense/vec3";
import mat2 from "../dense/mat2";
import vec4 from "../dense/vec4";
import Matrix from "../dense/denseMatrix";

test("Rotation conversions", () => {
    let axisAngleRotation = new axisAngle(new vec3(1., 2., -3.), radians(70));

    let quatRotation = quat.fromAxisAngle(axisAngleRotation);
    let matRotation = mat3.fromAxisAngle(axisAngleRotation);
    expect(matRotation.determinant()).toBeCloseTo(1);
    expect(Matrix.near(quatRotation.toMat3(), matRotation, SmallTolerance)).toBeTruthy();
    expect(quat.near(quatRotation, matRotation.toQuat(), SmallTolerance)).toBeTruthy();
    expect(axisAngle.near(quatRotation.toAxisAngle(), axisAngleRotation, SmallTolerance)).toBeTruthy();
    expect(axisAngle.near(matRotation.toAxisAngle(), axisAngleRotation, SmallTolerance)).toBeTruthy();

    let point = new vec3(0.3, -0.5, -0.2);
    let eulerRotation = { yaw: radians(10), pitch: radians(25), roll: radians(62) };
    matRotation = mat3.fromEulerAngles(eulerRotation.yaw, eulerRotation.pitch, eulerRotation.roll);
    quatRotation = quat.fromEulerAngles(eulerRotation.yaw, eulerRotation.pitch, eulerRotation.roll);
    let matRotationEuler = mat3.yaw(eulerRotation.yaw).mulSelf(mat3.pitch(eulerRotation.pitch)).mulSelf(mat3.roll(eulerRotation.roll));
    expect(vec3.near(quatRotation.toMat3().transformPoint3D(point), matRotation.transformPoint3D(point), SmallTolerance)).toBeTruthy();
    expect(vec3.near(quatRotation.rotate(point), matRotation.toQuat().rotate(point), SmallTolerance)).toBeTruthy();
    expect(vec3.near(matRotation.transformPoint3D(point), matRotationEuler.transformPoint3D(point), SmallTolerance)).toBeTruthy();
    expect(Matrix.near(matRotation, matRotationEuler, SmallTolerance)).toBeTruthy();

    matRotation = mat3.fromEulerAngles(eulerRotation.yaw, eulerRotation.pitch, eulerRotation.roll);
    let ea = matRotation.toEulerAngles();
    let eaMat = mat3.fromEulerAngles(ea.x, ea.y, ea.z);

    expect(vec3.near(eaMat.transformPoint3D(point), matRotation.transformPoint3D(point), SmallTolerance)).toBeTruthy();

    matRotation = mat3.fromEulerAngles(radians(180), 0, 0);
    expect(Math.abs(vec3.dot(matRotation.axis(), new vec3(0, 1, 0)))).toBeCloseTo(1, 4);
});

test("Rotations", () => {
    let axisAngleRotation = new axisAngle(new vec3(1., 2., -3.), radians(70));
    let trans = new transform3D(new vec3(-2, 3, 4.), quat.fromAxisAngle(axisAngleRotation), new vec3(1.2, -0.2, 0.4));
    let point = new vec3(0.3, -0.5, -0.2);
    let vector = new vec3(0.3, -0.5, -0.2);
    let affineTransform = trans.toAffine();
    expect(vec3.near(affineTransform.transformPoint3D(point), trans.transformPoint3D(point), Tolerance)).toBeTruthy();
    expect(vec3.near(affineTransform.transformVector3D(vector), trans.transformVector3D(vector), Tolerance)).toBeTruthy();

    let quatRotation = quat.fromAxisAngle(axisAngleRotation);
    let matRotation = mat3.fromAxisAngle(axisAngleRotation);
    expect(vec3.near(quatRotation.rotate(point), matRotation.transformPoint3D(point), Tolerance)).toBeTruthy();
    expect(vec3.near(quatRotation.rotate(point), axisAngleRotation.rotate(point), Tolerance)).toBeTruthy();

    let eulerRotation = new vec3(radians(30), radians(55), radians(72));
    quatRotation = quat.fromEulerAngles(eulerRotation.y, eulerRotation.x, eulerRotation.z);
    matRotation = mat3.fromEulerAngles(eulerRotation.y, eulerRotation.x, eulerRotation.z);
    axisAngleRotation = quatRotation.toAxisAngle();
    expect(vec3.near(quatRotation.rotate(point), matRotation.transformPoint3D(point), Tolerance)).toBeTruthy();
    expect(vec3.near(quatRotation.rotate(point), eulerRotation.rotateEuler(point), Tolerance)).toBeTruthy();
    expect(vec3.near(axisAngleRotation.rotate(point), eulerRotation.rotateEuler(point), Tolerance)).toBeTruthy();

    axisAngleRotation = new axisAngle(new vec3(1., 2., -3.), radians(70));
    matRotation = mat3.fromAxisAngle(axisAngleRotation);
    expect(Matrix.near(matRotation.inverse(), matRotation.transpose(), Tolerance)).toBeTruthy();
});

test("Transformations", () => {
    let axisAngleRotation = new axisAngle(new vec3(1., 2., -3.), radians(70));
    let translation3D = new vec3(-2, 3, 4.);
    let rotation3D = quat.fromAxisAngle(axisAngleRotation);
    let scale3D = new vec3(1.2, -0.2, 0.4);
    let t3D = mat4.fromTranslation3D(translation3D);
    let r3D = mat4.fromRotation3D(rotation3D);
    let s3D = mat4.fromScale3D(scale3D);
    let affine3D = mat4.fromTRS(translation3D, rotation3D, scale3D);
    let inverseAffine3D = affine3D.inverse();
    let transformation3D = new transform3D(translation3D, rotation3D, scale3D);
    let point3D = new vec3(0.6, -1.2, 1.6);
    expect(Matrix.near(affine3D, t3D.mul(r3D).mul(s3D), Tolerance)).toBeTruthy();
    expect(vec3.near(transformation3D.transformPoint3D(point3D), affine3D.transformPoint3D(point3D), Tolerance)).toBeTruthy();
    expect(vec3.near(transformation3D.transformVector3D(point3D), affine3D.transformVector3D(point3D), Tolerance)).toBeTruthy();
    expect(vec3.near(transformation3D.transformInversePoint3D(point3D), inverseAffine3D.transformPoint3D(point3D), Tolerance)).toBeTruthy();
    expect(vec3.near(transformation3D.transformInverseVector3D(point3D), inverseAffine3D.transformVector3D(point3D), Tolerance)).toBeTruthy();
    expect(transformation3D.hasUniformScale()).toBeFalsy();


    let rigidTransform3D = new transform3D(translation3D, rotation3D, new vec3(1., 1., 1.));
    let inverseRigidTransform3D = rigidTransform3D.inverse();
    expect(vec3.near(point3D, affine3D.transformPoint3D(inverseAffine3D.transformPoint3D(point3D)), Tolerance)).toBeTruthy();
    expect(vec3.near(rigidTransform3D.transformInversePoint3D(point3D),
        inverseRigidTransform3D.transformPoint3D(point3D), SmallTolerance)).toBeTruthy();

    let translation2D = new vec2(1.0, -2.0);
    let rotation2D = radians(30);
    let scale2D = new vec2(-1.1, 0.1);
    let t2D = mat3.fromTranslation2D(translation2D);
    let r2D = mat3.fromRotation2D(rotation2D);
    let s2D = mat3.fromScale2D(scale2D);
    let affine2D = mat3.fromTRS(translation2D, rotation2D, scale2D);
    let inverseAffine2D = affine2D.inverse();
    let transformation2D = new transform2D(translation2D, rotation2D, scale2D);
    let point2D = new vec2(3.6, 1.6);
    expect(Matrix.near(affine2D, t2D.mul(r2D).mul(s2D), Tolerance)).toBeTruthy();
    expect(vec2.near(transformation2D.transformPoint2D(point2D), affine2D.transformPoint2D(point2D), Tolerance)).toBeTruthy();
    expect(vec2.near(transformation2D.transformVector2D(point2D), affine2D.transformVector2D(point2D), Tolerance)).toBeTruthy();

    expect(vec2.near(point2D, affine2D.transformPoint2D(inverseAffine2D.transformPoint2D(point2D)), Tolerance)).toBeTruthy();

    expect(vec2.near(transformation2D.transformInversePoint2D(point2D), inverseAffine2D.transformPoint2D(point2D), Tolerance)).toBeTruthy();
    expect(vec2.near(transformation2D.transformInverseVector2D(point2D), inverseAffine2D.transformVector2D(point2D), Tolerance)).toBeTruthy();
    expect(transformation2D.hasUniformScale()).toBeFalsy();

    let rigidTransform2D = new transform2D(translation2D, rotation2D, new vec2(-1., -1.));
    let inverseRigidTransform2D = rigidTransform2D.inverse();
    expect(vec2.near(rigidTransform2D.transformInversePoint2D(point2D),
        inverseRigidTransform2D.transformPoint2D(point2D), Tolerance)).toBeTruthy();
});

test("Transform components", () => {
    let point3D = new vec3(0.3, -0.5, -0.2);
    let axisAngleRotation = new axisAngle(new vec3(1., 2., -3.), radians(70));
    let translation3D = new vec3(-2, 3., 4.);
    let rotation3D = quat.fromAxisAngle(axisAngleRotation);
    let scale3D = new vec3(1.2, -0.2, 0.4);
    let t = mat4.fromTranslation3D(translation3D);
    let r = mat4.fromRotation3D(rotation3D);
    let s = mat4.fromScale3D(scale3D);
    let trs = mat4.fromTRS(translation3D, rotation3D, scale3D);
    let pS1 = vec3.mul(point3D, scale3D);
    let pS2 = s.transformPoint3D(point3D);
    let pRS1 = rotation3D.rotate(pS1);
    let pRS2 = r.transformPoint3D(pS2);
    let pTRS1 = vec3.add(translation3D, pRS1);
    let pTRS2 = t.transformPoint3D(pRS2);
    expect(vec3.near(pS1, pS2, Tolerance)).toBeTruthy();
    expect(vec3.near(pRS1, pRS2, Tolerance)).toBeTruthy();
    expect(vec3.near(pTRS1, pTRS2, Tolerance)).toBeTruthy();
    expect(vec3.near(pTRS1, trs.transformPoint3D(point3D), Tolerance)).toBeTruthy();

    let point2D = new vec2(3.0, 13.0);
    let translation2D = new vec2(-3.0, 0.1);
    let rotation2D = radians(30.0);
    let scale2D = new vec2(0.1, -2.3);
    let t2D = mat3.fromTranslation2D(translation2D);
    let r2D = mat3.fromRotation2D(rotation2D);
    let s2D = mat3.fromScale2D(scale2D);
    let trs_2D = mat3.fromTRS(translation2D, rotation2D, scale2D);
    let pS1_2D = vec2.mul(point2D, scale2D);
    let pS2_2D = s2D.transformPoint2D(point2D);
    let pRS1_2D = vec2.rotate2D(pS1_2D, rotation2D);
    let pRS2_2D = r2D.transformPoint2D(pS2_2D);
    let pTRS1_2D = vec2.add(translation2D, pRS1_2D);
    let pTRS2_2D = t2D.transformPoint2D(pRS2_2D);
    expect(vec2.near(pS1_2D, pS1_2D, SmallTolerance)).toBeTruthy();
    expect(vec2.near(pRS1_2D, pRS2_2D, SmallTolerance)).toBeTruthy();
    expect(vec2.near(pTRS1_2D, pTRS2_2D, SmallTolerance)).toBeTruthy();
    expect(vec2.near(pTRS1_2D, trs_2D.transformPoint2D(point2D), SmallTolerance)).toBeTruthy();
});