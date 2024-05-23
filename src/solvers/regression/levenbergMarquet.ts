import Matrix from "../../dense/denseMatrix";
import Vector from "../../dense/vector";
import { assertFail } from "../../utils";
import PartialPivLU from "../linear systems/partialPivLU";
import { LeastSquaresResiduals } from "./problem";

// from The Levenberg-Marquardt algorithm for
// nonlinear least squares curve-fitting problems
enum LMLambdaUpdateType {
    First = 0,
    Second = 1,
    Third = 2
}

export class LevenbergMarquet {
    numIters: number;
    errAbsTol: number;
    initialLambda: number;
    method: LMLambdaUpdateType;
    lambdaIncreaseFactor: number = 11;
    lambdaDecreaseFactor: number = 9;
    roTolerance = 1e-4;
    method1(residuals: LeastSquaresResiduals, p0: Vector): { solution: Vector, error: number } {
        let p = p0.clone();
        let lambda = this.initialLambda;
        let numParams = residuals.numParams();
        let curError = residuals.error(p);
        for (let i = 0; i < this.numIters; i++) {
            let rhs = Vector.empty(numParams);
            let H = Matrix.empty(numParams, numParams);
            for (let resIdx = 0; resIdx < residuals.numResiduals(); resIdx++) {
                let r = residuals.f(p, resIdx);
                let drdp = residuals.dfdp(p, resIdx);
                rhs.addSelf(Vector.scale(drdp, -r));
                H.addSelf(Vector.outer(drdp, drdp));
            }
            if (curError < this.errAbsTol)
                break;
            const hDiag = H.diag().scaleSelf(lambda);
            for (let j = 0; j < numParams; ++j)
                H.set(j, j, H.get(j, j) * (1 + lambda));
            let dp = PartialPivLU.solve(H, rhs);
            let pNew = Vector.add(p, dp);
            let newError = residuals.error(pNew);
            let ro = (curError - newError) / (Vector.dot(dp, hDiag.mul(dp).addSelf(rhs)));
            if (ro > this.roTolerance) {
                p = pNew;
                lambda = Math.max(lambda / this.lambdaDecreaseFactor, 1e-7);
                curError = newError;
            } else {
                lambda = Math.min(lambda * this.lambdaIncreaseFactor, 1e7);
            }
        }
        return { solution: p, error: curError };
    }
    method2(residuals: LeastSquaresResiduals, p0: Vector): { solution: Vector, error: number } {
        let p = p0.clone();
        let lambda = this.initialLambda;
        let numParams = residuals.numParams();
        let curError = residuals.error(p);
        for (let i = 0; i < this.numIters; i++) {
            let rhs = Vector.empty(numParams);
            let H = Matrix.empty(numParams, numParams);
            for (let resIdx = 0; resIdx < residuals.numResiduals(); resIdx++) {
                let r = residuals.f(p, resIdx);
                let drdp = residuals.dfdp(p, resIdx);
                rhs.addSelf(Vector.scale(drdp, -r));
                H.addSelf(Vector.outer(drdp, drdp));
            }
            if (curError < this.errAbsTol)
                break;
            if (i == 0) {
                let maxDiag = 0.0;
                for (let j = 0; j < numParams; ++j)
                    maxDiag = Math.max(maxDiag, Math.abs(H.get(j, j)));
                lambda *= maxDiag;
            }
            for (let j = 0; j < numParams; ++j)
                H.set(j, j, H.get(j, j) + lambda);
            let dp = PartialPivLU.solve(H, rhs);
            let pNew = Vector.add(p, dp);
            let newError = residuals.error(pNew);
            const rhsProj = Vector.dot(rhs, dp);
            let alpha = rhsProj / ((newError - curError) / 2 + 2 * rhsProj);
            dp.scaleSelf(alpha);
            pNew = Vector.add(p, dp);
            newError = residuals.error(pNew);
            let ro = (curError - newError) / Vector.dot(dp, dp.scaleSelf(lambda).addSelf(rhs));

            if (ro > this.roTolerance) {
                p = pNew;
                lambda = Math.max(lambda / (1 + alpha), 1e-7);
                curError = newError;
            } else {
                lambda += Math.abs(curError - newError) / (2 * alpha);
            }
        }
        return { solution: p, error: curError };
    }
    method3(residuals: LeastSquaresResiduals, p0: Vector): { solution: Vector, error: number } {
        let p = p0.clone();
        let lambda = this.initialLambda;
        let numParams = residuals.numParams();
        let curError = residuals.error(p);
        let v = 2;
        for (let i = 0; i < this.numIters; i++) {
            let rhs = Vector.empty(numParams);
            let H = Matrix.empty(numParams, numParams);
            for (let resIdx = 0; resIdx < residuals.numResiduals(); resIdx++) {
                let r = residuals.f(p, resIdx);
                let drdp = residuals.dfdp(p, resIdx);
                rhs.addSelf(Vector.scale(drdp, -r));
                H.addSelf(Vector.outer(drdp, drdp));
            }
            if (curError < this.errAbsTol)
                break;
            if (i == 0) {
                let maxDiag = 0.0;
                for (let j = 0; j < numParams; ++j)
                    maxDiag = Math.max(maxDiag, Math.abs(H.get(j, j)));
                lambda *= maxDiag;
            }
            for (let j = 0; j < numParams; ++j)
                H.set(j, j, H.get(j, j) + lambda);
            let dp = PartialPivLU.solve(H, rhs);
            let pNew = Vector.add(p, dp);
            let newError = residuals.error(pNew);
            let ro = (curError - newError) / Vector.dot(dp, dp.scaleSelf(lambda).addSelf(rhs));

            if (ro > this.roTolerance) {
                p = pNew;
                lambda = lambda * Math.max(1 - Math.pow(2 * ro - 1, 3), 0.333);
                v = 2;
                curError = newError;
            } else {
                lambda = lambda * v;
                v = 2 * v;
            }
        }
        return { solution: p, error: curError };
    }
    solve(residuals: LeastSquaresResiduals, p0: Vector): { solution: Vector, error: number } {
        switch (this.method) {
            case LMLambdaUpdateType.First: return this.method1(residuals, p0);
            case LMLambdaUpdateType.Second: return this.method2(residuals, p0);
            case LMLambdaUpdateType.Third: return this.method3(residuals, p0);
        }
        assertFail("Unknown method");
    }
}