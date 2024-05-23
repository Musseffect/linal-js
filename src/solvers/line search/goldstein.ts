import Vector from "../../dense/vector";
import { BacktrackingLineSearch, LineSearch, LineSearchProblem } from "./lineSearch";


export default class GoldsteinLineSearch extends BacktrackingLineSearch {
    constructor(problem: LineSearchProblem) {
        super(problem);
    }
    protected decreaseCondition(initialValue: number, x: Vector, direction: Vector, step: number, cosAngle: number): boolean {
        const nextValue = this.problem.f(Vector.add(x, Vector.scale(direction, step)));
        return initialValue + (1 - this.gradCoeff) * step * cosAngle <= nextValue &&
            nextValue <= initialValue + this.gradCoeff * step * cosAngle;
    }
}