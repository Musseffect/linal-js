import Matrix from "../../dense/denseMatrix";
import { lerp } from "../../utils";
import vector from "../../dense/vector";

export interface eode {
    f(x: vector, t: number): vector;
    size(): number;
}
export interface odeState {
    x: vector;
    t: number;
}
export interface iode extends eode {
    dfdx(x: vector, t: number): Matrix;
}

export abstract class odeSolver {
    abstract step(x: vector, t: number): odeState;
    solve(x0: vector, t0: number, t1: number): vector {
        let xCur = x0.clone();
        let tCur = t0;
        let xNext = xCur;
        let t = t0;
        while (t < t1) {
            xCur = xNext;
            tCur = t;
            let nextState = this.step(xCur, tCur);
            t = nextState.t;
            xNext = nextState.x;
        }
        return vector.lerp(xCur, xNext, (t1 - tCur) / (t - tCur));
    }
}