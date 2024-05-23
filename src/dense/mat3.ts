import axisAngle from "../axisAngle";
import mat2 from "./mat2";
import mat4 from "./mat4";
import Matrix from "./denseMatrix";
import quat from "../quat";
import { assert, determinant2x2, determinant3x3, Tolerance, near, SmallTolerance } from "../utils";
import vec2 from "./vec2";
import vec3 from "./vec3";
import mat from "./abstractDenseMatrix";


export default class mat3 extends mat {
    constructor(m11: number, m12: number, m13: number,
        m21: number, m22: number, m23: number,
        m31: number, m32: number, m33: number) {
        super([
            m11, m12, m13,
            m21, m22, m23,
            m31, m32, m33
        ], 3, 3);
    }
    isIdentity(tolerance: number = SmallTolerance): boolean {
        let diff = 0;
        for (let i = 0; i < 3; ++i) {
            for (let j = 0; j < 3; ++j) {
                diff = Math.max(Math.abs(this.get(i, j) - (i == j ? 0 : 1)), diff);
            }
        }
        return diff < tolerance;
    }
    clone(): mat3 {
        return new mat3(
            this.get(0, 0), this.get(0, 1), this.get(0, 2),
            this.get(1, 0), this.get(1, 1), this.get(1, 2),
            this.get(2, 0), this.get(2, 1), this.get(2, 2),
        );
    }
    static empty(): mat3 {
        return new mat3(
            0, 0, 0,
            0, 0, 0,
            0, 0, 0);
    }
    static identity(): mat3 {
        return new mat3(
            1, 0, 0,
            0, 1, 0,
            0, 0, 1);
    }
    static fromAxisAngle(aa: axisAngle): mat3 {
        return aa.toMat3();
    }
    trace(): number {
        return this.get(0, 0) + this.get(1, 1) + this.get(2, 2);
    }
    // return normalized axis of rotation matrix
    axis(): vec3 {
        let trace = this.trace();
        let axis = new vec3(
            this.get(2, 1) - this.get(1, 2),
            this.get(0, 2) - this.get(2, 0),
            this.get(1, 0) - this.get(0, 1)
        );
        if (near(trace, 3, SmallTolerance)) {
            axis = new vec3(1., 0., 0.);
        } else if (near(axis.l1norm(), 0, SmallTolerance)) {
            let values = [this.get(0, 0),
            this.get(1, 1), this.get(2, 2)];
            if (values[0] > values[1]) {
                if (values[0] > values[2]) {
                    axis.x = 0.5 * Math.sqrt(values[0] - values[1] - values[2] + 1);
                    axis.y = this.get(0, 1) / (2 * axis.x);
                    axis.z = this.get(0, 2) / (2 * axis.x);
                    return axis;
                }
            } else {
                if (values[1] > values[2]) {
                    axis.y = 0.5 * Math.sqrt(values[1] - values[2] - values[0] + 1);
                    axis.x = this.get(0, 1) / (2 * axis.y);
                    axis.z = this.get(1, 2) / (2 * axis.y);
                    return axis;
                }
            }
            axis.z = 0.5 * Math.sqrt(values[2] - values[0] - values[1] + 1)
            axis.x = this.get(0, 2) / (2 * axis.z);
            axis.y = this.get(1, 2) / (2 * axis.z);
        }
        return axis.normalize();
    }
    toAxisAngle(): axisAngle {
        let trace = this.trace();
        let axis = this.axis();
        let angle = Math.acos((trace - 1) / 2);
        return new axisAngle(axis, angle);
    }
    toEulerAngles(): vec3 {
        let sp = -this.get(1, 2);
        let cp = Math.sqrt(1 - sp * sp);
        let sr = 0.0;
        let cr = 1.0;
        let cy = 0.5 * (this.get(0, 0) + this.get(0, 1) + this.get(2, 0) + this.get(2, 1));
        let sy = 0.5 * (this.get(0, 0) + this.get(0, 1) - this.get(2, 0) - this.get(2, 1));
        if (cp > SmallTolerance) {
            sr = this.get(1, 0) / cp;
            cr = this.get(1, 1) / cp;
            sy = this.get(0, 2) / cp;
            cy = this.get(2, 2) / cp;
        }
        return new vec3(Math.atan2(sy, cy), Math.atan2(sp, cp), Math.atan2(sr, cr));
    }
    // rotation around y
    static yaw(yaw: number): mat3 {
        let cy = Math.cos(yaw);
        let sy = Math.sin(yaw);
        let m11 = cy;
        let m12 = 0;
        let m13 = sy;
        let m21 = 0;
        let m22 = 1;
        let m23 = 0;
        let m31 = -sy;
        let m32 = 0;
        let m33 = cy;
        return new mat3(
            m11, m12, m13,
            m21, m22, m23,
            m31, m32, m33);
    }
    // rotation around x
    static pitch(pitch: number): mat3 {
        let cp = Math.cos(pitch);
        let sp = Math.sin(pitch);
        let m11 = 1;
        let m12 = 0;
        let m13 = 0;
        let m21 = 0;
        let m22 = cp;
        let m23 = -sp;
        let m31 = 0;
        let m32 = sp;
        let m33 = cp;
        return new mat3(
            m11, m12, m13,
            m21, m22, m23,
            m31, m32, m33);
    }
    // rotation around z
    static roll(roll: number): mat3 {
        let cr = Math.cos(roll);
        let sr = Math.sin(roll);
        let m11 = cr;
        let m12 = -sr;
        let m13 = 0;
        let m21 = sr;
        let m22 = cr;
        let m23 = 0;
        let m31 = 0;
        let m32 = 0;
        let m33 = 1;
        return new mat3(
            m11, m12, m13,
            m21, m22, m23,
            m31, m32, m33);
    }
    // ZXY rotations - opengl coordinate system with z - forward, y - up, x - left
    // rotate around z(roll), rotate around x(pitch), rotate around y(yaw)
    static fromEulerAngles(yaw: number, pitch: number, roll: number): mat3 {
        let cy = Math.cos(yaw);
        let sy = Math.sin(yaw);
        let cp = Math.cos(pitch);
        let sp = Math.sin(pitch);
        let cr = Math.cos(roll);
        let sr = Math.sin(roll);

        let m11 = cy * cr + sy * sp * sr;
        let m12 = sy * sp * cr - cy * sr;
        let m13 = sy * cp;
        let m21 = cp * sr;
        let m22 = cp * cr;
        let m23 = -sp;
        let m31 = cy * sp * sr - sy * cr;
        let m32 = sy * sr + cy * sp * cr;
        let m33 = cy * cp;

        return new mat3(
            m11, m12, m13,
            m21, m22, m23,
            m31, m32, m33);
    }
    set(row: number, col: number, value: number): void {
        this.data[row * 3 + col] = value;
    }
    get(row: number, col: number): number {
        assert(row >= 0 && row < 3 && col >= 0 && col < 3, "Invalid index");
        return this.data[row * 3 + col];
    }
    toQuat(): quat {
        let trace = this.trace();
        let axis = this.axis();
        let angle = Math.acos((trace - 1) / 2);
        return new quat(axis.scaleSelf(Math.sin(angle / 2.0)), Math.cos(angle / 2.0));
    }
    toMat4(): mat4 {
        return new mat4(
            this.get(0, 0), this.get(0, 1), this.get(0, 2), 0.,
            this.get(1, 0), this.get(1, 1), this.get(1, 2), 0.,
            this.get(2, 0), this.get(2, 1), this.get(2, 2), 0.,
            0., 0., 0., 1.
        );
    }
    static scale(a: mat3, l: number): mat3 {
        return a.scale(l);
    }
    scale(l: number, out?: mat3): mat3 {
        if (!out) {
            out = this.clone();
        }

        for (let i = 0; i < 3; i++) {
            for (let j = 0; j < 3; j++) {
                out.set(i, j, this.get(i, j) * l);
            }
        }
        return out;
    }
    scaleSelf(l: number): mat3 {
        return this.scale(l, this);
    }
    static mul(a: mat3, b: mat3): mat3 {
        return a.mul(b);
    }
    mul(m: mat3, out?: mat3): mat3 {
        if (!out) {
            out = this.clone();
        }

        for (let i = 0; i < 3; i++) {
            for (let j = 0; j < 3; j++) {
                let result = 0.0;
                result += this.get(i, 0) * m.get(0, j);
                result += this.get(i, 1) * m.get(1, j);
                result += this.get(i, 2) * m.get(2, j);
                out.set(i, j, result);
            }
        }
        return out;
    }
    mulSelf(m: mat3): mat3 {
        return this.clone().mul(m, this);
    }
    determinant(): number {
        return this.get(0, 0) * (this.get(1, 1) * this.get(2, 2) - this.get(1, 2) * this.get(2, 1))
            - this.get(0, 1) * (this.get(1, 0) * this.get(2, 2) - this.get(2, 0) * this.get(1, 2))
            + this.get(0, 2) * (this.get(1, 0) * this.get(2, 1) - this.get(2, 0) * this.get(1, 1));
    }
    static solve(m: mat3, v: vec3): vec3 {
        let d = m.determinant();
        let dx = determinant3x3(
            v.x, m.get(0, 1), m.get(0, 2),
            v.y, m.get(1, 1), m.get(1, 2),
            v.z, m.get(2, 1), m.get(2, 2)
        );
        let dy = determinant3x3(
            m.get(0, 0), v.x, m.get(0, 2),
            m.get(1, 0), v.y, m.get(1, 2),
            m.get(2, 0), v.z, m.get(2, 2)
        );
        let dz = determinant3x3(
            m.get(0, 0), m.get(0, 1), v.x,
            m.get(1, 0), m.get(1, 1), v.y,
            m.get(2, 0), m.get(2, 1), v.z
        );
        return new vec3(dx / d, dy / d, dz / d);
    }
    inverse(): mat3 {
        let d = this.determinant();
        let m11 = determinant2x2(
            this.get(1, 1), this.get(1, 2),
            this.get(2, 1), this.get(2, 2)
        );
        let m12 = determinant2x2(
            this.get(0, 2), this.get(0, 1),
            this.get(2, 2), this.get(2, 1)
        );
        let m13 = determinant2x2(
            this.get(0, 1), this.get(0, 2),
            this.get(1, 1), this.get(1, 2)
        );
        let m21 = determinant2x2(
            this.get(1, 2), this.get(1, 0),
            this.get(2, 2), this.get(2, 0)
        );
        let m22 = determinant2x2(
            this.get(0, 0), this.get(0, 2),
            this.get(2, 0), this.get(2, 2)
        );
        let m23 = determinant2x2(
            this.get(0, 2), this.get(0, 0),
            this.get(1, 2), this.get(1, 0)
        );
        let m31 = determinant2x2(
            this.get(1, 0), this.get(1, 1),
            this.get(2, 0), this.get(2, 1)
        );
        let m32 = determinant2x2(
            this.get(0, 1), this.get(0, 0),
            this.get(2, 1), this.get(2, 0)
        );
        let m33 = determinant2x2(
            this.get(0, 0), this.get(0, 1),
            this.get(1, 0), this.get(1, 1)
        );
        return new mat3(
            m11, m12, m13,
            m21, m22, m23,
            m31, m32, m33
        ).scaleSelf(1.0 / d);
    }
    transpose(): mat3 {
        return new mat3(
            this.get(0, 0), this.get(1, 0), this.get(2, 0),
            this.get(0, 1), this.get(1, 1), this.get(2, 1),
            this.get(0, 2), this.get(1, 2), this.get(2, 2)
        );
    }
    toMatrix(): Matrix {
        let data = [];
        for (let j = 0; j < 3; ++j) {
            for (let i = 0; i < 3; ++i) {
                data.push(this.get(i, j));
            }
        }
        return new Matrix(data, 4, 4);
    }
    transformPoint3D(p: vec3): vec3 {
        return this.postMulVec(p);
    }
    transformPoint2D(p: vec2): vec2 {
        let result = [0, 0, 0];
        for (let i = 0; i < 3; i++) {
            result[i] += this.get(i, 0) * p.x;
            result[i] += this.get(i, 1) * p.y;
            result[i] += this.get(i, 2);
        }
        return new vec2(result[0] / result[2], result[1] / result[2]);
    }
    transformVector2D(v: vec2): vec2 {
        let result = [0, 0];
        for (let i = 0; i < 2; i++) {
            result[i] += this.get(i, 0) * v.x;
            result[i] += this.get(i, 1) * v.y;
        }
        return new vec2(result[0], result[1]);
    }
    toString(): string {
        return `[
    [${this.get(0, 0).toFixed(4)}, ${this.get(0, 1).toFixed(4)}, ${this.get(0, 2).toFixed(4)}]
    [${this.get(1, 0).toFixed(4)}, ${this.get(1, 1).toFixed(4)}, ${this.get(1, 2).toFixed(4)}]
    [${this.get(2, 0).toFixed(4)}, ${this.get(2, 1).toFixed(4)}, ${this.get(2, 2).toFixed(4)}]
]`;
    }
    static fromRotationAroundAxis(angle: number, axis: number = 2): mat3 {
        let ca = Math.cos(angle);
        let sa = Math.sin(angle);
        assert(axis >= 0 && axis < 3, "Wrong axis");
        let result = mat3.empty();
        let f = (axis + 1) % 3;
        let s = (axis + 2) % 3;
        result.set(f, f, ca);
        result.set(f, s, -sa);
        result.set(s, f, sa);
        result.set(s, s, ca);
        return result;
    }
    static fromRotation2D(angle: number): mat3 {
        let ca = Math.cos(angle);
        let sa = Math.sin(angle);
        return new mat3(
            ca, -sa, 0,
            sa, ca, 0,
            0, 0, 1
        );
    }
    static fromScale2D(scale: vec2): mat3 {
        return new mat3(
            scale.x, 0, 0,
            0, scale.y, 0,
            0, 0, 1);
    }
    static fromTranslation2D(translation: vec2): mat3 {
        return new mat3(
            1, 0, translation.x,
            0, 1, translation.y,
            0, 0, 1
        );
    }
    static fromTRS(t: vec2, r: number, s: vec2): mat3 {
        let ca = Math.cos(r);
        let sa = Math.sin(r);
        return new mat3(
            ca * s.x, -sa * s.y, t.x,
            sa * s.x, ca * s.y, t.y,
            0, 0, 1
        );
    }
    extractMat2(): mat2 {
        return new mat2(
            this.get(0, 0), this.get(0, 1),
            this.get(1, 0), this.get(1, 1)
        );
    }
    preMulVec(p: vec3): vec3 {
        let result = [0, 0, 0,];
        for (let i = 0; i < 3; i++) {
            result[i] += this.get(0, i) * p.x;
            result[i] += this.get(1, i) * p.y;
            result[i] += this.get(2, i) * p.z;
        }
        return new vec3(result[0], result[1], result[2]);
    }
    postMulVec(p: vec3): vec3 {
        let result = [0, 0, 0];
        for (let i = 0; i < 3; i++) {
            result[i] += this.get(i, 0) * p.x;
            result[i] += this.get(i, 1) * p.y;
            result[i] += this.get(i, 2) * p.z;
        }
        return new vec3(result[0], result[1], result[2]);
    }
    static add(a: mat3, b: mat3): mat3 {
        return a.add(b);
    }
    add(m: mat3, out?: mat3): mat3 {
        if (!out)
            out = this.clone();
        return out.addSelf(m);
    }
    addSelf(m: mat3): mat3 {
        super.addSelf(m);
        return this;
    }
    static sub(a: mat3, b: mat3): mat3 {
        return a.sub(b);
    }
    sub(m: mat3, out?: mat3): mat3 {
        if (!out)
            out = this.clone();
        return out.subSelf(m);
    }
    subSelf(m: mat3): mat3 {
        super.subSelf(m);
        return this;
    }
}