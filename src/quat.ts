import axisAngle from "./axisAngle";
import mat3 from "./dense/mat3";
import mat4 from "./dense/mat4";
import { Tolerance, SmallTolerance, SmallestTolerance } from "./utils";
import vec3 from "./dense/vec3";

export default class quat {
    v: vec3;
    s: number;
    constructor(v: vec3, s: number) {
        this.v = v;
        this.s = s;
    }
    static fromVectors(a: vec3, b: vec3): quat {
        let axis = vec3.cross(a, b);
        let cos = vec3.dot(a, b) / a.l2norm() / b.l2norm();
        if (1 - Math.abs(cos) <= 0.0001) {
            axis = new vec3(1, 0, 0);
        }
        let angle = Math.acos(cos);
        return this.fromAxisAngle(new axisAngle(axis, angle));
    }
    clone(): quat {
        return new quat(this.v.clone(), this.s);
    }
    static near(a: quat, b: quat, threshold?: number): boolean {
        if (!threshold) {
            threshold = Tolerance;
        }
        return Math.abs(a.s - b.s) <= threshold && vec3.near(a.v, b.v, threshold);
    }
    static identity(): quat {
        return new quat(new vec3(0., 0., 0.), 1.0);
    }
    static empty(): quat {
        return new quat(vec3.empty(), 0.0);
    }
    static fromAxisAngle(o: axisAngle): quat {
        return new quat(vec3.normalize(o.axis).scaleSelf(Math.sin(o.angle / 2)), Math.cos(o.angle / 2));
    }
    static fromComponents(x: number, y: number, z: number, w: number): quat {
        return new quat(new vec3(x, y, z), w);
    }
    set(q: quat): quat {
        this.v.set(q.v);
        this.s = q.s;
        return this;
    }
    // ZXY rotations - opengl coordinate system with z - forward, y - up, x - left
    // rotate around z(roll), rotate around x(pitch), rotate around y(yaw)
    static fromEulerAngles(yaw: number, pitch: number, roll: number): quat {
        let cy = Math.cos(yaw / 2);
        let sy = Math.sin(yaw / 2);
        let cp = Math.cos(pitch / 2);
        let sp = Math.sin(pitch / 2);
        let cr = Math.cos(roll / 2);
        let sr = Math.sin(roll / 2);
        return new quat(new vec3(
            cy * sp * cr + sy * cp * sr,
            sy * cp * cr - cy * sp * sr,
            cy * cp * sr - sy * sp * cr
        ),
            cy * cp * cr + sy * sp * sr
        );
    }
    static fromMat3(m: mat3): quat {
        return m.toQuat();
    }
    //add
    static add(a: quat, b: quat): quat {
        return a.add(b);
    }
    add(q: quat, out?: quat): quat {
        if (!out)
            out = this.clone();
        this.v.add(q.v, out.v);
        out.s = this.s + q.s;
        return out;
    }
    addSelf(q: quat): quat {
        return this.add(q, this);
    }
    //sub
    static sub(a: quat, b: quat): quat {
        return a.sub(b);
    }
    sub(q: quat, out?: quat): quat {
        if (!out)
            out = this.clone();
        this.v.sub(q.v, out.v);
        out.s = this.s - q.s;
        return out;
    }
    subSelf(q: quat): quat {
        return this.sub(q, this);
    }
    conj(out?: quat): quat {
        if (!out)
            out = this.clone();
        else
            out.set(this);
        out.v.negateSelf();
        return out;
    }
    conjSelf(): quat {
        return this.conj(this);
    }
    static conj(q: quat): quat {
        return q.conj();
    }
    static inverse(q: quat): quat {
        return q.inverse();
    }
    inverseSelf(): quat {
        return this.inverse(this);
    }
    inverse(out?: quat): quat {
        if (!out)
            out = this.clone();
        else
            out.set(this);
        return out.conj().scale(1.0 / out.squaredLength());
    }
    //mul
    static mul(a: quat, b: quat): quat {
        return a.mul(b);
    }
    mul(q: quat, out?: quat): quat {
        if (!out) {
            out = this.clone();
        }
        const { x, y, z } = this.v;
        const s = this.s;
        out.s = s * q.s - x * q.v.x - y * q.v.y - z * q.v.z;
        out.v.x = s * q.v.x + x * q.s + y * q.v.z - z * q.v.y;
        out.v.y = s * q.v.y - x * q.v.z + y * q.s + z * q.v.x;
        out.v.z = s * q.v.z + x * q.v.y - y * q.v.x + z * q.s;
        return out;
    }
    mulSelf(q: quat): quat {
        return this.mul(q, this);
    }
    //div
    static div(a: quat, b: quat): quat {
        return a.div(b);
    }
    div(q: quat, out?: quat): quat {
        if (!out) {
            out = this.clone();
        }
        out.s = this.s * q.s + vec3.dot(this.v, q.v);
        out.v = this.v.scale(q.s).subSelf(q.v.scale(this.s)).subSelf(vec3.cross(this.v, q.v));
        return out.scaleSelf(1.0 / q.squaredLength());
    }
    divSelf(q: quat): quat {
        return this.div(q, this);
    }
    //scale
    static scale(q: quat, l: number): quat {
        return q.scale(l);
    }
    scale(l: number, out?: quat): quat {
        if (!out) {
            out = this.clone();
        }
        out.s = this.s * l;
        this.v.scale(l, out.v);
        return out;
    }
    scaleSelf(l: number): quat {
        return this.scale(l, this);
    }
    static normalize(v: quat): quat {
        return v.clone().normalize();
    }
    normalize(): quat {
        let l = this.l2norm();
        if (l < SmallestTolerance) {
            this.v.x = 0;
            this.v.y = 0;
            this.v.z = 0;
            this.s = 1;
            return this;
        }
        return this.scaleSelf(1. / l);
    }
    toMat3(): mat3 {
        const { x, y, z } = this.v;
        const x2 = this.v.x * this.v.x;
        const y2 = this.v.y * this.v.y;
        const z2 = this.v.z * this.v.z;
        return new mat3(
            1. - 2. * y2 - 2. * z2,
            2. * x * y - 2. * this.s * z,
            2. * x * z + 2. * this.s * y,

            2. * x * y + 2. * this.s * z,
            1.0 - 2. * x2 - 2. * z2,
            2. * y * z - 2. * this.s * x,

            2. * x * z - 2.0 * this.s * y,
            2. * y * z + 2.0 * this.s * x,
            1.0 - 2. * x2 - 2. * y2
        );
    }
    toMat4(): mat4 {
        const { x, y, z } = this.v;
        const x2 = this.v.x * this.v.x;
        const y2 = this.v.y * this.v.y;
        const z2 = this.v.z * this.v.z;
        return new mat4(
            1. - 2. * y2 - 2. * z2,
            2. * x * y - 2. * this.s * z,
            2. * x * z + 2. * this.s * y,
            0.,

            2. * x * y + 2. * this.s * z,
            1.0 - 2. * x2 - 2. * z2,
            2. * y * z - 2. * this.s * x,
            0.,

            2. * x * z - 2.0 * this.s * y,
            2. * y * z + 2.0 * this.s * x,
            1.0 - 2. * x2 - 2. * y2,
            0.,

            0.,
            0.,
            0.,
            1.
        );
    }
    toAxisAngle(): axisAngle {
        let angle = 2. * Math.atan2(this.v.l2norm(), this.s);
        angle = (Math.abs(angle) > Math.PI ? 2 * Math.PI * Math.sign(angle) - angle : angle);
        let axis = this.v.clone().normalize();
        let ca = Math.sqrt(1.0 - this.s * this.s);
        if (ca < SmallTolerance) {
            axis = new vec3(1., 0., 0.);
        }
        return new axisAngle(axis, angle);
    }
    rotate(v: vec3): vec3 {
        return this.mul(new quat(v.clone(), 0.0)).mulSelf(quat.conj(this)).v;
    }
    l2norm(): number {
        return Math.sqrt(this.squaredLength());
    }
    squaredLength(): number {
        return this.v.squaredLength() + this.s * this.s;
    }
    toString(): string {
        return `[${this.v.x.toFixed(4)}i, ${this.v.y.toFixed(4)}j, ${this.v.z.toFixed(4)}k, ${this.s.toFixed(4)}]`;
    }
}