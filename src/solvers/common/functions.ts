import Matrix from "../../dense/denseMatrix";
import Vector from "../../dense/vector";

export interface ScalarFunction {
    f(x: number): number;
    dfdx(x: number): number;
    dfddx(x: number): number;
}

export interface VectorArgumentFunction {
    f(x: Vector): number;
    dfdx(x: Vector): Vector;
    dfddx(x: Vector): Matrix;
}