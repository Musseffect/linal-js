import vector from "../../dense/vector";
import { eode, odeSolver, odeState } from "./ode";


export default class rk4 extends odeSolver {
    system: eode;
    h: number;
    constructor(system: eode, h: number) {
        super();
        this.system = system;
        this.h = h;
    }
    step(x: vector, t: number): odeState {
        let k1 = this.system.f(x, t);
        let k2 = this.system.f(vector.scale(k1, this.h * 0.5).addSelf(x), t + this.h * 0.5);
        let k3 = this.system.f(vector.scale(k2, this.h * 0.5).addSelf(x), t + this.h * 0.5);
        let k4 = this.system.f(vector.scale(k3, this.h).addSelf(x), t + this.h);
        let xNext = k2.addSelf(k3).scaleSelf(2.0).addSelf(k1).addSelf(k4).scaleSelf(this.h / 6).addSelf(x);
        return { x: xNext, t: t + this.h };
    }
}