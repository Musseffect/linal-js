import mat4 from "./dense/mat4";
import quat from "./quat";
import { assert, near } from "./utils";
import vec2 from "./dense/vec2";
import vec3 from "./dense/vec3";

export default class transform3D {
    translation: vec3;
    rotation: quat;
    scale: vec3;
    constructor(translation: vec3, rotation: quat, scale: vec3) {
        this.translation = translation;
        this.rotation = rotation;
        this.scale = scale;
    }
    isRigid(): boolean {
        return near(this.scale.x, 1.0) && near(this.scale.y, 1.0) && near(this.scale.z, 1.0);
    }
    hasUniformScale(): boolean {
        return near(Math.abs(this.scale.x), Math.abs(this.scale.y))
            && near(Math.abs(this.scale.x), Math.abs(this.scale.z))
            && near(Math.abs(this.scale.y), Math.abs(this.scale.z));
    }
    clone(): transform3D {
        return new transform3D(this.translation.clone(), this.rotation.clone(), this.scale.clone());
    }
    static identity(): transform3D {
        return new transform3D(new vec3(0, 0, 0), quat.identity(), new vec3(1, 1, 1));
    }
    toAffine(): mat4 {
        return mat4.fromTRS(this.translation, this.rotation, this.scale);
    }
    transformPoint3D(point: vec3): vec3 {
        return this.rotation.rotate(vec3.mul(point, this.scale)).addSelf(this.translation);
    }
    transformVector3D(vector: vec3): vec3 {
        return this.rotation.rotate(vec3.mul(vector, this.scale));
    }
    transformInversePoint3D(point: vec3): vec3 {
        return quat.inverse(this.rotation).rotate(vec3.sub(point, this.translation)).divSelf(this.scale);
    }
    transformInverseVector3D(vector: vec3): vec3 {
        return quat.inverse(this.rotation).rotate(vector).divSelf(this.scale);
    }
    // only possible with uniform scale
    inverse(): transform3D {
        assert(this.hasUniformScale(), "can't inverse transform3D with non-uniform scale");
        let invRot = quat.inverse(this.rotation);
        return new transform3D(
            invRot.rotate(vec3.div(vec3.negate(this.translation), this.scale)),
            invRot,
            new vec3(1.0 / this.scale.x, 1.0 / this.scale.y, 1.0 / this.scale.z)
        );
    }
    toString(): string {
        return `{ t:${this.translation.toString()}, r:${this.rotation.toString()}, s:${this.scale.toString()} }`;
    }
}