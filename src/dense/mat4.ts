import mat3 from "./mat3";
import quat from "../quat";
import vec3 from "./vec3";
import { determinant3x3, determinant4x4, Tolerance, SmallTolerance } from "../utils";
import vec4 from "./vec4";
import Matrix from "./denseMatrix";
import AbstractDenseMatrix from "./abstractDenseMatrix";


export default class mat4 extends AbstractDenseMatrix {
    constructor(m11: number, m12: number, m13: number, m14: number,
        m21: number, m22: number, m23: number, m24: number,
        m31: number, m32: number, m33: number, m34: number,
        m41: number, m42: number, m43: number, m44: number) {
        super([
            m11, m12, m13, m14,
            m21, m22, m23, m24,
            m31, m32, m33, m34,
            m41, m42, m43, m44
        ], 4, 4);
    }
    isIdentity(tolerance: number = SmallTolerance): boolean {
        let diff = 0;
        for (let i = 0; i < 4; ++i) {
            for (let j = 0; j < 4; ++j) {
                diff = Math.max(Math.abs(this.get(i, j) - (i == j ? 0 : 1)), diff);
            }
        }
        return diff < tolerance;
    }
    clone(): mat4 {
        return new mat4(
            this.get(0, 0), this.get(0, 1), this.get(0, 2), this.get(0, 3),
            this.get(1, 0), this.get(1, 1), this.get(1, 2), this.get(2, 3),
            this.get(2, 0), this.get(2, 1), this.get(2, 2), this.get(2, 3),
            this.get(3, 0), this.get(3, 1), this.get(3, 2), this.get(3, 3)
        );
    }
    static empty(): mat4 {
        return new mat4(
            0, 0, 0, 0,
            0, 0, 0, 0,
            0, 0, 0, 0,
            0, 0, 0, 0);
    }
    static identity(): mat4 {
        return new mat4(
            1, 0, 0, 0,
            0, 1, 0, 0,
            0, 0, 1, 0,
            0, 0, 0, 1);
    }
    extractMat3(): mat3 {
        return new mat3(
            this.get(0, 0), this.get(0, 1), this.get(0, 2),
            this.get(1, 0), this.get(1, 1), this.get(1, 2),
            this.get(2, 0), this.get(2, 1), this.get(2, 2)
        );
    }
    determinant(): number {
        return determinant4x4(
            this.get(0, 0), this.get(0, 1), this.get(0, 2), this.get(0, 3),
            this.get(1, 0), this.get(1, 1), this.get(1, 2), this.get(1, 3),
            this.get(2, 0), this.get(2, 1), this.get(2, 2), this.get(2, 3),
            this.get(3, 0), this.get(3, 1), this.get(3, 2), this.get(3, 3)
        );
    }
    extractTranslation(): vec3 {
        return new vec3(this.get(0, 3), this.get(1, 3), this.get(2, 3));
    }
    transformPoint3D(p: vec3): vec3 {
        let result = [0, 0, 0, 0];
        for (let i = 0; i < 4; i++) {
            result[i] += this.get(i, 0) * p.x;
            result[i] += this.get(i, 1) * p.y;
            result[i] += this.get(i, 2) * p.z;
            result[i] += this.get(i, 3);
        }
        return new vec3(result[0] / result[3], result[1] / result[3], result[2] / result[3]);
    }
    transformVector3D(v: vec3): vec3 {
        let result = [0, 0, 0];
        for (let i = 0; i < 3; i++) {
            result[i] += this.get(i, 0) * v.x;
            result[i] += this.get(i, 1) * v.y;
            result[i] += this.get(i, 2) * v.z;
        }
        return new vec3(result[0], result[1], result[2]);
    }
    preMulVec(p: vec4): vec4 {
        let result = [0, 0, 0, 0];
        for (let i = 0; i < 4; i++) {
            result[i] += this.get(0, i) * p.x;
            result[i] += this.get(1, i) * p.y;
            result[i] += this.get(2, i) * p.z;
            result[i] += this.get(3, i) * p.w;
        }
        return new vec4(result[0], result[1], result[2], result[3]);
    }
    postMulVec(p: vec4): vec4 {
        let result = [0, 0, 0, 0];
        for (let i = 0; i < 4; i++) {
            result[i] += this.get(i, 0) * p.x;
            result[i] += this.get(i, 1) * p.y;
            result[i] += this.get(i, 2) * p.z;
            result[i] += this.get(i, 3) * p.w;
        }
        return new vec4(result[0], result[1], result[2], result[3]);
    }
    transpose(): mat4 {
        return new mat4(
            this.get(0, 0), this.get(1, 0), this.get(2, 0), this.get(3, 0),
            this.get(0, 1), this.get(1, 1), this.get(2, 1), this.get(3, 1),
            this.get(0, 2), this.get(1, 2), this.get(2, 2), this.get(3, 2),
            this.get(0, 3), this.get(1, 3), this.get(2, 3), this.get(3, 3)
        );
    }
    toMatrix(): Matrix {
        let data = [];
        for (let j = 0; j < 4; ++j) {
            for (let i = 0; i < 4; ++i) {
                data.push(this.get(j, i));
            }
        }
        return new Matrix(data, 4, 4);
    }
    static scale(a: mat4, l: number): mat4 {
        return a.scale(l);
    }
    scale(l: number, out?: mat4): mat4 {
        if (!out) {
            out = this.clone();
        }

        for (let i = 0; i < 4; i++) {
            for (let j = 0; j < 4; j++)
                out.set(i, j, this.get(i, j) * l);
        }
        return out;
    }
    scaleSelf(l: number): mat4 {
        return this.scale(l, this);
    }
    static mul(a: mat4, b: mat4): mat4 {
        return a.mul(b);
    }
    mul(m: mat4, out?: mat4): mat4 {
        if (!out) {
            out = this.clone();
        }

        for (let i = 0; i < 4; i++) {
            for (let j = 0; j < 4; j++) {
                let result = 0.0;
                result += this.get(i, 0) * m.get(0, j);
                result += this.get(i, 1) * m.get(1, j);
                result += this.get(i, 2) * m.get(2, j);
                result += this.get(i, 3) * m.get(3, j);
                out.set(i, j, result);
            }
        }
        return out;
    }
    mulSelf(m: mat4): mat4 {
        return this.clone().mul(m, this);
    }
    trace(): number {
        return this.get(0, 0) + this.get(1, 1) + this.get(2, 2) + this.get(3, 3);
    }
    static fromMat3(m: mat3): mat4 {
        return new mat4(
            m.get(0, 0), m.get(0, 1), m.get(0, 2), 0,
            m.get(1, 0), m.get(1, 1), m.get(1, 2), 0,
            m.get(2, 0), m.get(2, 1), m.get(2, 2), 0,
            0, 0, 0, 1);
    }
    static fromTranslation3D(t: vec3): mat4 {
        return new mat4(
            1, 0, 0, t.x,
            0, 1, 0, t.y,
            0, 0, 1, t.z,
            0, 0, 0, 1);
    }
    static fromRotation3D(r: quat): mat4 {
        return r.toMat4();
    }
    static fromScale3D(s: vec3): mat4 {
        return new mat4(
            s.x, 0, 0, 0,
            0, s.y, 0, 0,
            0, 0, s.z, 0,
            0, 0, 0, 1);
    }
    static fromTRS(t: vec3, r: quat, s: vec3): mat4 {
        let rMat = r.toMat3();
        return new mat4(
            rMat.get(0, 0) * s.x, rMat.get(0, 1) * s.y, rMat.get(0, 2) * s.z, t.x,
            rMat.get(1, 0) * s.x, rMat.get(1, 1) * s.y, rMat.get(1, 2) * s.z, t.y,
            rMat.get(2, 0) * s.x, rMat.get(2, 1) * s.y, rMat.get(2, 2) * s.z, t.z,
            0, 0, 0, 1
        );
    }
    inverse(): mat4 {
        let d = this.determinant();
        let m11 = determinant3x3(
            this.get(1, 1), this.get(1, 2), this.get(1, 3),
            this.get(2, 1), this.get(2, 2), this.get(2, 3),
            this.get(3, 1), this.get(3, 2), this.get(3, 3)
        );
        let m21 = determinant3x3(
            this.get(1, 0), this.get(1, 2), this.get(1, 3),
            this.get(2, 0), this.get(2, 2), this.get(2, 3),
            this.get(3, 0), this.get(3, 2), this.get(3, 3)
        );
        let m31 = determinant3x3(
            this.get(1, 0), this.get(1, 1), this.get(1, 3),
            this.get(2, 0), this.get(2, 1), this.get(2, 3),
            this.get(3, 0), this.get(3, 1), this.get(3, 3)
        );
        let m41 = determinant3x3(
            this.get(1, 0), this.get(1, 1), this.get(1, 2),
            this.get(2, 0), this.get(2, 1), this.get(2, 2),
            this.get(3, 0), this.get(3, 1), this.get(3, 2)
        );
        let m12 = determinant3x3(
            this.get(0, 1), this.get(0, 2), this.get(0, 3),
            this.get(2, 1), this.get(2, 2), this.get(2, 3),
            this.get(3, 1), this.get(3, 2), this.get(3, 3)
        );
        let m22 = determinant3x3(
            this.get(0, 0), this.get(0, 2), this.get(0, 3),
            this.get(2, 0), this.get(2, 2), this.get(2, 3),
            this.get(3, 0), this.get(3, 2), this.get(3, 3)
        );
        let m32 = determinant3x3(
            this.get(0, 0), this.get(0, 1), this.get(0, 3),
            this.get(2, 0), this.get(2, 1), this.get(2, 3),
            this.get(3, 0), this.get(3, 1), this.get(3, 3)
        );
        let m42 = determinant3x3(
            this.get(0, 0), this.get(0, 1), this.get(0, 2),
            this.get(2, 0), this.get(2, 1), this.get(2, 2),
            this.get(3, 0), this.get(3, 1), this.get(3, 2)
        );
        let m13 = determinant3x3(
            this.get(0, 1), this.get(0, 2), this.get(0, 3),
            this.get(1, 1), this.get(1, 2), this.get(1, 3),
            this.get(3, 1), this.get(3, 2), this.get(3, 3)
        );
        let m23 = determinant3x3(
            this.get(0, 0), this.get(0, 2), this.get(0, 3),
            this.get(1, 0), this.get(1, 2), this.get(1, 3),
            this.get(3, 0), this.get(3, 2), this.get(3, 3)
        );
        let m33 = determinant3x3(
            this.get(0, 0), this.get(0, 1), this.get(0, 3),
            this.get(1, 0), this.get(1, 1), this.get(1, 3),
            this.get(3, 0), this.get(3, 1), this.get(3, 3)
        );
        let m43 = determinant3x3(
            this.get(0, 0), this.get(0, 1), this.get(0, 2),
            this.get(1, 0), this.get(1, 1), this.get(1, 2),
            this.get(3, 0), this.get(3, 1), this.get(3, 2)
        );
        let m14 = determinant3x3(
            this.get(0, 1), this.get(0, 2), this.get(0, 3),
            this.get(1, 1), this.get(1, 2), this.get(1, 3),
            this.get(2, 1), this.get(2, 2), this.get(2, 3)
        );
        let m24 = determinant3x3(
            this.get(0, 0), this.get(0, 2), this.get(0, 3),
            this.get(1, 0), this.get(1, 2), this.get(1, 3),
            this.get(2, 0), this.get(2, 2), this.get(2, 3)
        );
        let m34 = determinant3x3(
            this.get(0, 0), this.get(0, 1), this.get(0, 3),
            this.get(1, 0), this.get(1, 1), this.get(1, 3),
            this.get(2, 0), this.get(2, 1), this.get(2, 3)
        );
        let m44 = determinant3x3(
            this.get(0, 0), this.get(0, 1), this.get(0, 2),
            this.get(1, 0), this.get(1, 1), this.get(1, 2),
            this.get(2, 0), this.get(2, 1), this.get(2, 2)
        );
        return new mat4(
            m11, -m12, m13, -m14,
            -m21, m22, -m23, m24,
            m31, -m32, m33, -m34,
            -m41, m42, -m43, m44
        ).scaleSelf(1 / d);
    }
    toString(): string {
        return `[
    [${this.get(0, 0).toFixed(4)}, ${this.get(0, 1).toFixed(4)}, ${this.get(0, 2).toFixed(4)}, ${this.get(0, 3).toFixed(4)}]
    [${this.get(1, 0).toFixed(4)}, ${this.get(1, 1).toFixed(4)}, ${this.get(1, 2).toFixed(4)}, ${this.get(1, 3).toFixed(4)}]
    [${this.get(2, 0).toFixed(4)}, ${this.get(2, 1).toFixed(4)}, ${this.get(2, 2).toFixed(4)}, ${this.get(2, 3).toFixed(4)}]
    [${this.get(3, 0).toFixed(4)}, ${this.get(3, 1).toFixed(4)}, ${this.get(3, 2).toFixed(4)}, ${this.get(3, 3).toFixed(4)}]
]`;
    }
}