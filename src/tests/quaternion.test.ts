import axisAngle from "../axisAngle";
import quat from "../quat";
import { radians } from "../utils";
import vec3 from "../dense/vec3";

test('Quaternion basic operations', () => {
    let axisAngleRotation = new axisAngle(new vec3(1., 2., -3.), radians(70));
    let q1 = quat.fromAxisAngle(axisAngleRotation);
    expect(quat.near(q1.mul(quat.inverse(q1)), quat.identity())).toBeTruthy();
    let q2 = quat.fromComponents(1.0, 2.0, 0.3, 0.157);
    let qDiv = q1.div(q2);
    let qAdd = q1.add(q2);
    let qSub = q1.sub(q2);
    let qMul = q1.mul(q2);
    expect(quat.near(q1, qDiv.mul(q2))).toBeTruthy();
    expect(quat.near(q1, qAdd.sub(q2))).toBeTruthy();
    expect(quat.near(q1, qSub.add(q2))).toBeTruthy();
    expect(quat.near(q1, qMul.div(q2))).toBeTruthy();
});