import Matrix from "../../dense/denseMatrix";
import Vector from "../../dense/vector";
import PartialPivLU from "../linear systems/partialPivLU";
import { LeastSquaresResiduals } from "./problem";

export class GaussNewton {
    numIters: number;
    errAbsTol: number;
    alpha: number;
    solve(residuals: LeastSquaresResiduals, p0: Vector): { solution: Vector, error: number } {
        let p = p0.clone();
        let numParams = residuals.numParams();
        for (let i = 0; i < this.numIters; i++) {
            let rhs = Vector.empty(numParams);
            let H = Matrix.empty(numParams, numParams);
            let error = 0.0;
            for (let resIdx = 0; resIdx < residuals.numResiduals(); resIdx++) {
                let r = residuals.f(p, resIdx);
                let drdp = residuals.dfdp(p, resIdx);
                error += r * r;
                rhs.addSelf(Vector.scale(drdp, -r * this.alpha));
                H.addSelf(Vector.outer(drdp, drdp));
            }
            if (error < this.errAbsTol)
                break;
            let dp = PartialPivLU.solve(H, rhs);
            p.addSelf(dp);
        }
        let error = residuals.error(p);
        return { solution: p, error: error };
    }
}