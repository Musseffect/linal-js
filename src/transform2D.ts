import mat3 from "./dense/mat3";
import { assert, near } from "./utils";
import vec2 from "./dense/vec2";

export default class transform2D {
    translation: vec2;
    rotation: number;
    scale: vec2;
    constructor(translation: vec2, rotation: number, scale: vec2) {
        this.translation = translation;
        this.rotation = rotation;
        this.scale = scale;
    }
    isRigid(): boolean {
        return near(this.scale.x, 1.0) && near(this.scale.y, 1.0);
    }
    hasUniformScale(): boolean {
        return near(this.scale.x, this.scale.y);
    }
    clone(): transform2D {
        return new transform2D(this.translation.clone(), this.rotation, this.scale.clone());
    }
    static identity(): transform2D {
        return new transform2D(new vec2(0, 0), 0.0, new vec2(1, 1));
    }
    toAffine(): mat3 {
        return mat3.fromTRS(this.translation, this.rotation, this.scale);
    }
    transformPoint2D(point: vec2): vec2 {
        return vec2.rotate2D(vec2.mul(point, this.scale), this.rotation).addSelf(this.translation);
    }
    transformVector2D(vector: vec2): vec2 {
        return vec2.rotate2D(vec2.mul(vector, this.scale), this.rotation);
    }
    transformInversePoint2D(point: vec2): vec2 {
        return vec2.rotate2D(vec2.sub(point, this.translation), -this.rotation).divSelf(this.scale);
    }
    transformInverseVector2D(vector: vec2): vec2 {
        return vec2.rotate2D(vector, -this.rotation).divSelf(this.scale);
    }
    // only possible with uniform scale
    inverse(): transform2D {
        assert(this.hasUniformScale(), "can't inverse transform2D with non-uniform scale");
        return new transform2D(
            vec2.rotate2D(vec2.div(vec2.negate(this.translation), this.scale), -this.rotation),
            -this.rotation,
            new vec2(1.0 / this.scale.x, 1.0 / this.scale.y)
        );
    }
    toString(): string {
        return `{ t:${this.translation.toString()}, r:${this.rotation}, s:${this.scale.toString()} }`;
    }
}