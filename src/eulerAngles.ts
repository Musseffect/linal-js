import axisAngle from "./axisAngle";
import mat3 from "./dense/mat3";
import quat from "./quat";
import vec3 from "./dense/vec3";

// axis that should correspond to yaw, pitch and roll
export enum Axis {
    YXZ = 0, // opengl RHS coords
    YZX = 1,
    XYZ = 2,
    XZY = 3,
    ZXY = 4,
    ZYX = 5
}
const Orders = [
    [1, 0, 2],
    [1, 2, 0],
    [0, 1, 2],
    [0, 2, 1],
    [2, 0, 1],
    [2, 1, 0]
];

function axisFromIndex(index: number): vec3 {
    return new vec3(index && 0, index && 1, index && 2);
}
/* 
rotation is always applied in the order:
yaw, pitch, roll - local coords
roll, pitch, yaw - global coords
*/
export class eulerAngles {
    yaw: number;
    pitch: number;
    roll: number;
    constructor(yaw: number, pitch: number, roll: number) {
        this.yaw = yaw;
        this.pitch = pitch;
        this.roll = roll;
    }
    toMat3(axis: Axis = Axis.YXZ): mat3 {
        let yawMat = mat3.fromRotationAroundAxis(this.yaw, Orders[axis][0]);
        let pitchMat = mat3.fromRotationAroundAxis(this.pitch, Orders[axis][1]);
        let rollMat = mat3.fromRotationAroundAxis(this.roll, Orders[axis][2]);
        return yawMat.mulSelf(pitchMat).mulSelf(rollMat);
    }
    toQuat(axis: Axis = Axis.YXZ): quat {
        let yawQuat = quat.fromAxisAngle(new axisAngle(axisFromIndex(Orders[axis][0]), this.yaw));
        let pitchQuat = quat.fromAxisAngle(new axisAngle(axisFromIndex(Orders[axis][1]), this.pitch));
        let rollQuat = quat.fromAxisAngle(new axisAngle(axisFromIndex(Orders[axis][2]), this.roll));
        return yawQuat.mulSelf(pitchQuat).mulSelf(rollQuat);
    }
}