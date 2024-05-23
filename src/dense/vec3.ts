import axisAngle from "../axisAngle";
import { lerp, Tolerance, SmallTolerance, SmallestTolerance } from "../utils";
import vector from "./vector";

export default class vec3 {
    x: number;
    y: number;
    z: number;
    constructor(x: number, y: number, z: number) {
        this.x = x;
        this.y = y;
        this.z = z;
    }
    set(v: vec3): void {
        this.x = v.x;
        this.y = v.y;
        this.z = v.z;
    }
    toArray(): number[] {
        return [this.x, this.y, this.z];
    }
    toVector(): vector {
        return new vector(this.toArray());
    }
    static normalize(v: vec3): vec3 {
        return v.clone().normalize();
    }
    normalize(out?: vec3): vec3 {
        if (!out)
            out = this.clone();
        else
            out.set(this);

        let l = this.l2norm();
        if (l < SmallestTolerance) {
            out.x = 0;
            out.y = 0;
            out.z = 0;
            return out;
        }
        return out.scaleSelf(1 / l);
    }
    normalizeSelf(): vec3 {
        return this.normalize(this);
    }
    negate(out?: vec3): vec3 {
        if (!out)
            out = this.clone();
        out.x = -this.x;
        out.y = -this.y;
        out.z = -this.z;
        return out;
    }
    negateSelf(): vec3 {
        return this.negate(this);
    }
    static negate(v: vec3) {
        return new vec3(-v.x, -v.y, -v.z);
    }
    static lerp(a: vec3, b: vec3, t: number): vec3 {
        return new vec3(
            lerp(a.x, b.x, t),
            lerp(a.y, b.y, t),
            lerp(a.z, b.z, t)
        );
    }
    static near(a: vec3, b: vec3, absTolerance: number = SmallTolerance, relTolerance: number = 0): boolean {
        let norm = vec3.sub(a, b).lInfnorm();
        if (relTolerance == 0)
            return norm <= absTolerance;
        return norm <= Math.max(absTolerance, relTolerance * Math.max(a.lInfnorm(), b.lInfnorm()));
    }
    static empty(): vec3 {
        return new vec3(0., 0., 0.);
    }
    clone(): vec3 {
        return new vec3(this.x, this.y, this.z);
    }
    //scale
    static scale(a: vec3, scalar: number): vec3 {
        return a.scale(scalar);
    }
    scale(scalar: number, out?: vec3): vec3 {
        if (!out)
            out = this.clone();
        out.x = this.x * scalar;
        out.y = this.y * scalar;
        out.z = this.z * scalar;
        return out;
    }
    scaleSelf(scalar: number): vec3 {
        this.scale(scalar, this);
        return this;
    }
    //add
    static add(a: vec3, b: vec3): vec3 {
        return a.add(b);
    }
    add(vec: vec3, out?: vec3): vec3 {
        if (!out)
            out = this.clone();
        out.x = this.x + vec.x;
        out.y = this.y + vec.y;
        out.z = this.z + vec.z;
        return out;
    }
    addSelf(v: vec3): vec3 {
        return this.add(v, this);
    }
    //sub
    static sub(a: vec3, b: vec3): vec3 {
        return a.sub(b);
    }
    sub(v: vec3, out?: vec3): vec3 {
        if (!out)
            out = this.clone();
        out.x = this.x - v.x;
        out.y = this.y - v.y;
        out.z = this.z - v.z;
        return out;
    }
    subSelf(v: vec3): vec3 {
        return this.sub(v, this);
    }
    //mul
    mul(v: vec3, out?: vec3): vec3 {
        if (!out)
            out = this.clone();
        out.x = this.x * v.x;
        out.y = this.y * v.y;
        out.z = this.z * v.z;
        return out;
    }
    static mul(a: vec3, b: vec3): vec3 {
        return a.mul(b);
    }
    mulSelf(v: vec3): vec3 {
        return this.mul(v, this);
    }
    //div
    div(v: vec3, out?: vec3): vec3 {
        if (!out)
            out = this.clone();
        out.x = this.x / v.x;
        out.y = this.y / v.y;
        out.z = this.z / v.z;
        return out;
    }
    static div(a: vec3, b: vec3): vec3 {
        return a.div(b);
    }
    divSelf(v: vec3): vec3 {
        return this.div(v, this);
    }
    //dot
    static dot(a: vec3, b: vec3): number {
        return a.x * b.x + a.y * b.y + a.z * b.z;
    }
    dot(v: vec3): number {
        return vec3.dot(this, v);
    }
    //cross
    static cross(a: vec3, b: vec3): vec3 {
        return new vec3(
            a.y * b.z - a.z * b.y,
            -a.x * b.z + a.z * b.x,
            a.x * b.y - a.y * b.x
        );
    }
    cross(v: vec3): vec3 {
        return vec3.cross(this, v);
    }
    lpnorm(p: number): number {
        return Math.pow(
            Math.pow(Math.abs(this.x), p) +
            Math.pow(Math.abs(this.y), p) +
            Math.pow(Math.abs(this.z), p)
            , 1.0 / p);
    }
    l1norm(): number {
        return Math.abs(this.x) + Math.abs(this.y) + Math.abs(this.z);
    }
    l2norm(): number {
        return Math.sqrt(this.squaredLength());
    }
    lInfnorm(): number {
        return Math.max(Math.abs(this.x), Math.max(Math.abs(this.y), Math.abs(this.z)));
    }
    length(): number {
        return this.l2norm();
    }
    squaredLength(): number {
        return this.x * this.x + this.y * this.y + this.z * this.z;
    }
    static distance(a: vec3, b: vec3): number {
        return a.sub(b).length();
    }
    static squaredDistance(a: vec3, b: vec3): number {
        return a.sub(b).squaredLength();
    }
    toString(): string {
        return `[${this.x.toFixed(4)}, ${this.y.toFixed(4)}, ${this.z.toFixed(4)}]`;
    }
    rotateEuler(point: vec3): vec3 {
        let ax = new axisAngle(new vec3(1, 0, 0), this.x);
        let ay = new axisAngle(new vec3(0, 1, 0), this.y);
        let az = new axisAngle(new vec3(0, 0, 1), this.z);
        return ay.rotate(ax.rotate(az.rotate(point)));
    }
    apply(op: (a: number) => number): vec3 {
        return new vec3(op(this.x), op(this.y), op(this.z));
    }
    static apply(a: vec3, b: vec3, op: (a: number, b: number) => number): vec3 {
        return new vec3(op(a.x, b.x), op(a.y, b.y), op(a.z, b.z));
    }
}