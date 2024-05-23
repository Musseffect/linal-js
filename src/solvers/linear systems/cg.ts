import Matrix from "../../dense/denseMatrix";
import { assert, SmallTolerance } from "../../utils";
import Vector from "../../dense/vector";
import { ConvergenseFailureException } from "./exceptions";

const SolverName = "'Cholesky'";

export enum CGPreconditioner {
    Identity,
    Jacobi
    //, IncompleteCholesky
};

interface Preconditioner {
    calc(r: Vector): Vector;
}

class IdentityPreconditioner implements Preconditioner {
    // will return reference to residual vector
    calc(r: Vector): Vector {
        return r;
    }
}
class JacobiPreconditioner implements Preconditioner {
    A: Matrix;
    constructor(A: Matrix) {
        this.A = A;
    };
    calc(r: Vector): Vector {
        let result = r.clone();
        for (let i = 0; i < r.size(); ++i)
            result.set(i, result.get(i) / this.A.get(i, i));
        return result;
    }
}

export class CG {
    A: Matrix;
    preconditioner: Preconditioner
    constructor(A: Matrix, preconditioner: CGPreconditioner) {
        assert(A.isSquare(), "Non-square matrix");
        switch (preconditioner) {
            case CGPreconditioner.Identity:
                this.preconditioner = new IdentityPreconditioner();
                break;
            case CGPreconditioner.Jacobi:
                this.preconditioner = new JacobiPreconditioner(A);
                break;
            default:
                throw Error("Unkown preconditioner");
        }
        this.A = A;
    }
    compute(b: Vector, maxIterations: number = 20, tolerance: number = SmallTolerance): Vector {
        assert(this.A.width() == b.data.length, "Width of matrix isn't compatible with vector's length");
        let x = Vector.empty(b.size());
        let r = Vector.sub(b, Matrix.postMulVec(this.A, x))
        if (r.lInfNorm() < tolerance) return x;
        let z = this.preconditioner.calc(r);
        let p = z.clone();
        let iter = 0;
        while (iter < maxIterations) {
            let ap = Matrix.postMulVec(this.A, p);
            let rDotZ = Vector.dot(r, z);
            let alpha = rDotZ / Vector.dot(p, ap);
            x.addSelf(Vector.scale(p, alpha));
            r.subSelf(Vector.scale(ap, alpha));
            if (r.lInfNorm() < tolerance) return x;
            z = this.preconditioner.calc(r);
            let rDotZNew = Vector.dot(r, z);
            let beta = rDotZNew / rDotZ;
            p = p.scaleSelf(beta).addSelf(z);
            rDotZ = rDotZNew;
            ++iter;
        }
        throw new ConvergenseFailureException(SolverName);
    }
    static solve(A: Matrix, b: Vector, maxIterations: number = 20, tolerance: number = SmallTolerance): Vector {
        let solver = new CG(A, CGPreconditioner.Jacobi);
        return solver.compute(b, maxIterations, tolerance);
    }
}