import { SmallestTolerance } from "../../utils";
import Vector from "../../dense/vector";
import { BacktrackingLineSearch, LineSearch } from "./lineSearch";


export default class WolfeLineSearch extends BacktrackingLineSearch {
    curvatureCoeff: number = 0.9;
    protected decreaseCondition(initialValue: number, x: Vector, direction: Vector, step: number, cosAngle: number): boolean {
        let nextArg = Vector.add(x, Vector.scale(direction, step));
        let curValue = this.problem.f(nextArg);
        return (initialValue - curValue) >= step * -this.gradCoeff * cosAngle &&
            -Vector.dot(direction, this.problem.grad(nextArg)) <= -this.curvatureCoeff * cosAngle;
    }
}