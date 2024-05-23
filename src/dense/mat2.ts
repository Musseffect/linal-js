import mat3 from "./mat3";
import Matrix from "./denseMatrix";
import { determinant2x2, Tolerance, SmallTolerance } from "../utils";
import vec2 from "./vec2";
import AbstractDenseMatrix from "./abstractDenseMatrix";

export default class mat2 extends AbstractDenseMatrix {
    constructor(m11: number, m12: number, m21: number, m22: number) {
        super([
            m11, m12,
            m21, m22
        ], 2, 2);
    }
    isIdentity(tolerance: number = SmallTolerance): boolean {
        let diff = 0;
        for (let i = 0; i < 2; ++i) {
            for (let j = 0; j < 2; ++j) {
                diff = Math.max(Math.abs(this.get(i, j) - (i == j ? 0 : 1)), diff);
            }
        }
        return diff < tolerance;
    }
    clone(): mat2 {
        return new mat2(
            this.get(0, 0), this.get(0, 1),
            this.get(1, 0), this.get(1, 1),
        );
    }
    static empty(): mat2 {
        return new mat2(
            0, 0,
            0, 0);
    }
    static identity(): mat2 {
        return new mat2(
            1, 0,
            0, 1);
    }
    determinant(): number {
        return determinant2x2(this.get(0, 0),
            this.get(0, 1), this.get(1, 0), this.get(1, 1));
    }
    static scale(a: mat2, l: number): mat2 {
        return a.scale(l);
    }
    scale(l: number, out?: mat2): mat2 {
        if (!out) {
            out = this.clone();
        }
        for (let i = 0; i < 2; i++) {
            for (let j = 0; j < 2; j++) {
                out.set(i, j, this.get(i, j) * l);
            }
        }
        return out;
    }
    scaleSelf(l: number): mat2 {
        return this.scale(l, this);
    }
    inverse(): mat2 {
        let d = this.determinant();
        return new mat2(
            this.get(1, 1), -this.get(0, 1),
            -this.get(1, 0), this.get(0, 0)
        ).scaleSelf(1.0 / d);
    }
    transpose(): mat2 {
        return new mat2(
            this.get(0, 0), this.get(1, 0),
            this.get(0, 1), this.get(1, 1)
        );
    }
    toString(): string {
        return `[
    [${this.get(0, 0).toFixed(4)}, ${this.get(0, 1).toFixed(4)}]
    [${this.get(1, 0).toFixed(4)}, ${this.get(1, 1).toFixed(4)}]
]`;
    }
    preMulVec(v: vec2): vec2 {
        return new vec2(v.x * this.get(0, 0) + v.y * this.get(1, 0), v.x * this.get(0, 1) + v.y * this.get(1, 1));
    }
    postMulVec(v: vec2): vec2 {
        return new vec2(v.x * this.get(0, 0) + v.y * this.get(0, 1), v.x * this.get(1, 0) + v.y * this.get(1, 1));
    }
    toMatrix(): Matrix {
        let data = [];
        for (let j = 0; j < 4; ++j) {
            for (let i = 0; i < 4; ++i) {
                data.push(this.get(i, j));
            }
        }
        return new Matrix(data, 4, 4);
    }
    static rotation(angle: number): mat2 {
        let ca = Math.cos(angle);
        let sa = Math.sin(angle);
        return new mat2(ca, -sa, sa, ca);
    }
    toMat3(): mat3 {
        return new mat3(
            this.get(0, 0), this.get(0, 1), 0.,
            this.get(1, 0), this.get(1, 1), 0.,
            0., 0., 1.
        );
    }
    transformPoint2D(p: vec2): vec2 {
        return this.postMulVec(p);
    }
    static add(a: mat2, b: mat2): mat2 {
        return a.add(b);
    }
    add(m: mat2, out?: mat2): mat2 {
        if (!out)
            out = this.clone();
        return out.addSelf(m);
    }
    addSelf(m: mat2): mat2 {
        super.addSelf(m);
        return this;
    }
    static sub(a: mat2, b: mat2): mat2 {
        return a.sub(b);
    }
    sub(m: mat2, out?: mat2): mat2 {
        if (!out)
            out = this.clone();
        return out.subSelf(m);
    }
    subSelf(m: mat2): mat2 {
        super.subSelf(m);
        return this;
    }
}