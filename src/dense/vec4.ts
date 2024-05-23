import { Tolerance, lerp, SmallTolerance, SmallestTolerance } from "../utils";
import vector from "./vector";

export default class vec4 {
    x: number;
    y: number;
    z: number;
    w: number;
    constructor(x: number, y: number, z: number, w: number) {
        this.x = x;
        this.y = y;
        this.z = z;
        this.w = w;
    }
    set(v: vec4): void {
        this.x = v.x;
        this.y = v.y;
        this.z = v.z;
        this.w = v.w;
    }
    toArray(): number[] {
        return [this.x, this.y, this.z, this.w];
    }
    toVector(): vector {
        return new vector(this.toArray());
    }
    negate(): vec4 {
        this.x = -this.x;
        this.y = -this.y;
        this.z = -this.z;
        this.w = -this.w;
        return this;
    }
    static negate(v: vec4) {
        return new vec4(-v.x, -v.y, -v.z, -v.w);
    }
    static lerp(a: vec4, b: vec4, t: number): vec4 {
        return new vec4(
            lerp(a.x, b.x, t),
            lerp(a.y, b.y, t),
            lerp(a.z, b.z, t),
            lerp(a.w, b.w, t)
        );
    }
    static near(a: vec4, b: vec4, absTolerance: number = SmallTolerance, relTolerance: number = 0): boolean {
        let norm = vec4.sub(a, b).lInfnorm();
        if (relTolerance == 0)
            return norm <= absTolerance;
        return norm <= Math.max(absTolerance, relTolerance * Math.max(a.lInfnorm(), b.lInfnorm()));
    }
    static empty(): vec4 {
        return new vec4(0., 0., 0., 0.);
    }
    clone(): vec4 {
        return new vec4(this.x, this.y, this.z, this.w);
    }
    //scale
    static scale(a: vec4, scalar: number): vec4 {
        return a.scale(scalar);
    }
    scale(scalar: number, out?: vec4): vec4 {
        if (!out) {
            out = this.clone();
        }
        out.x = this.x * scalar;
        out.y = this.y * scalar;
        out.z = this.z * scalar;
        out.w = this.w * scalar;
        return out;
    }
    scaleSelf(scalar: number): vec4 {
        this.scale(scalar, this);
        return this;
    }
    //add
    static add(a: vec4, b: vec4): vec4 {
        return a.add(b);
    }
    add(vec: vec4, out?: vec4): vec4 {
        if (!out) {
            out = this.clone();
        }
        out.x = this.x + vec.x;
        out.y = this.y + vec.y;
        out.z = this.z + vec.z;
        out.w = this.w + vec.w;
        return out;
    }
    addSelf(v: vec4): vec4 {
        return this.add(v, this);
    }
    //sub
    static sub(a: vec4, b: vec4): vec4 {
        return a.sub(b);
    }
    sub(v: vec4, out?: vec4): vec4 {
        if (!out) {
            out = this.clone();
        }
        out.x = this.x - v.x;
        out.y = this.y - v.y;
        out.z = this.z - v.z;
        out.w = this.w - v.w;
        return out;
    }
    subSelf(v: vec4): vec4 {
        return this.sub(v, this);
    }
    //mul
    mul(v: vec4, out?: vec4): vec4 {
        if (!out) {
            out = this.clone();
        }
        out.x = this.x * v.x;
        out.y = this.y * v.y;
        out.z = this.z * v.z;
        out.z = this.w * v.w;
        return out;
    }
    static mul(a: vec4, b: vec4): vec4 {
        return a.mul(b);
    }
    mulSelf(v: vec4): vec4 {
        return this.mul(v, this);
    }
    //div
    div(v: vec4, out?: vec4): vec4 {
        if (!out) {
            out = this.clone();
        }
        out.x = this.x / v.x;
        out.y = this.y / v.y;
        out.z = this.z / v.z;
        out.z = this.w / v.w;
        return out;
    }
    static div(a: vec4, b: vec4): vec4 {
        return a.div(b);
    }
    divSelf(v: vec4): vec4 {
        return this.div(v, this);
    }
    static normalize(v: vec4): vec4 {
        return v.clone().normalize();
    }
    normalize(): vec4 {
        let l = this.l2norm();
        if (l < SmallestTolerance) {
            this.x = 0;
            this.y = 0;
            this.z = 0;
            this.w = 0;
            return this;
        }
        return this.scaleSelf(1 / l);
    }
    lpnorm(p: number): number {
        return Math.pow(
            Math.pow(Math.abs(this.x), p) +
            Math.pow(Math.abs(this.y), p) +
            Math.pow(Math.abs(this.z), p) +
            Math.pow(Math.abs(this.w), p)
            , 1.0 / p);
    }
    l1norm(): number {
        return Math.abs(this.x) + Math.abs(this.y) + Math.abs(this.z) + Math.abs(this.w);
    }
    l2norm(): number {
        return Math.sqrt(this.squaredLength());
    }
    lInfnorm(): number {
        return Math.max(Math.max(Math.abs(this.x), Math.abs(this.y)), Math.max(Math.abs(this.z), Math.abs(this.w)));
    }
    length(): number {
        return this.l2norm();
    }
    squaredLength(): number {
        return this.x * this.x + this.y * this.y + this.z * this.z + this.w * this.w;
    }
    static distance(a: vec4, b: vec4): number {
        return a.sub(b).length();
    }
    static squaredDistance(a: vec4, b: vec4): number {
        return a.sub(b).squaredLength();
    }
    static dot(a: vec4, b: vec4): number {
        return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
    }
    dot(v: vec4): number {
        return vec4.dot(this, v);
    }
    toString(): string {
        return `[${this.x.toPrecision(4)}, ${this.y.toPrecision(4)}, ${this.z.toPrecision(4)}, ${this.w.toPrecision(4)}]`;
    }
    apply(op: (a: number) => number): vec4 {
        return new vec4(op(this.x), op(this.y), op(this.z), op(this.w));
    }
    static apply(a: vec4, b: vec4, op: (a: number, b: number) => number): vec4 {
        return new vec4(op(a.x, b.x), op(a.y, b.y), op(a.z, b.z), op(a.w, b.w));
    }
}