import mat3 from "./dense/mat3";
import { Tolerance, near } from "./utils";
import vec3 from "./dense/vec3";

export default class axisAngle {
    axis: vec3;
    angle: number;
    constructor(axis: vec3, angle: number) {
        this.axis = vec3.normalize(axis);
        this.angle = angle;
    }
    static near(a: axisAngle, b: axisAngle, threshold?: number): boolean {
        if (!threshold)
            threshold = Tolerance;
        return vec3.near(a.axis, b.axis, threshold) && near(a.angle, b.angle, threshold);
    }
    rotate(point: vec3): vec3 {
        return vec3.add(vec3.lerp(this.axis.scale(vec3.dot(this.axis, point)), point, Math.cos(this.angle)),
            vec3.cross(this.axis, point).scaleSelf(Math.sin(this.angle)));
    }
    toString(): string {
        return `{ axis:${this.axis.toString()}, angle:${this.angle.toFixed(4)} }`;
    }
    toMat3(): mat3 {
        const { x, y, z } = this.axis;
        let c = Math.cos(this.angle);
        let s = Math.sin(this.angle);
        let t = 1 - c;
        return new mat3(
            t * x * x + c,
            t * x * y - s * z,
            t * x * z + s * y,

            t * x * y + s * z,
            t * y * y + c,
            t * y * z - s * x,

            t * x * z - s * y,
            t * y * z + s * x,
            t * z * z + c
        );
    }
}