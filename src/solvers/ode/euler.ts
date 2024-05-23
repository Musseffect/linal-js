import vector from "../../dense/vector";
import { eode, odeSolver, odeState } from "./ode";


export default class euler extends odeSolver {
    system: eode;
    h: number;
    constructor(system: eode, h: number) {
        super();
        this.system = system;
        this.h = h;
    }
    step(x: vector, t: number): odeState {
        return { x: this.system.f(x, t).scaleSelf(this.h).addSelf(x), t: t + this.h };
    }
}