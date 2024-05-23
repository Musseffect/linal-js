import Matrix from "../../dense/denseMatrix";
import Vector from "../../dense/vector";
import { LeastSquaresFunction, LeastSquaresSamples } from "./problem";

// TODO:
describe.skip('Least squares', () => {
    test('Linear', () => {
        // z = aN + a0x0 + a1x1 + a2x2 + a3x3
        const parameters = new Vector([1, 2, 3, 0, -1]);
        let func: LeastSquaresFunction = {
            numParameters() {
                return 5;
            },
            numVariables() {
                return 4;
            },
            f: function (x: Vector, p: Vector): number {
                let value = 0;
                for (let i = 0; i < 4; ++i)
                    value += x.get(i) * p.get(i);
                return value + p.get(5);
            },
            dfdp: function (x: Vector, p: Vector): Vector {
                let v = x.clone();
                v.resize(5, 1);
                return v;
            },
            ddfdpdp: function (x: Vector, p: Vector): Matrix {
                return Matrix.empty(5, 5);
            }
        };
        let samples: LeastSquaresSamples = new LeastSquaresSamples();
        for (let i = 0; i < 1000; ++i) {
            let x = Vector.generate(4, (i) => { return Math.random(); });
            samples.arguments.push(x);
            samples.values.push(func.f(x, parameters));
        }

    });
    test('Lin log', () => {
        // z = a_0 + a_1 * log(x) + a_2 * log(y)
        const parameters = new Vector([1, 2, 3])
        let func: LeastSquaresFunction = {
            numParameters() {
                return 3;
            },
            numVariables() {
                return 2;
            },
            f: function (x: Vector, p: Vector): number {
                return p.get(0) + p.get(1) * Math.log(x.get(0)) + p.get(2) * Math.log(x.get(1));
            },
            dfdp: function (x: Vector, p: Vector): Vector {
                return new Vector([1, Math.log(x.get(0)), Math.log(x.get(1))]);
            },
            ddfdpdp: function (x: Vector, p: Vector): Matrix {
                return Matrix.empty(3, 3);
            }
        };
    });
    test('Bilinear', () => {
        // z = a00 + a10x + a01y + a11xy
        let func: LeastSquaresFunction = {
            numParameters() {
                return 4;
            },
            numVariables() {
                return 2;
            },
            f: function (x: Vector, p: Vector): number {
                let value = p.get(0);
                value += x.get(0) * (p.get(1) + p.get(3) * x.get(1));
                value += x.get(1) * p.get(2);
                return value;
            },
            dfdp: function (x: Vector, p: Vector): Vector {
                return new Vector([1, x.get(0), x.get(0), x.get(0) * x.get(1)]);
            },
            ddfdpdp: function (x: Vector, p: Vector): Matrix {
                return Matrix.empty(4, 4);
            }
        };
    });
    test('Gaussian', () => {
        function unpackParams(p: Vector): { a: number, b: number, c: number } {
            return { a: p.get(0), b: p.get(1), c: p.get(2) };
        }
        // z = a * exp(-b * (x - c)^2)
        let func: LeastSquaresFunction = {
            numParameters() {
                return 3;
            },
            numVariables() {
                return 1;
            },
            f: function (x: Vector, p: Vector): number {
                const { a, b, c } = unpackParams(p);
                return a * Math.exp(-b * Math.pow(x.get(0) - c, 2));
            },
            dfdp: function (x: Vector, p: Vector): Vector {
                const { a, b, c } = unpackParams(p);
                const v = x.get(0);
                const arg = Math.pow(v - c, 2);
                const exp = Math.exp(-b * arg);

                return new Vector([exp, -this.f(x, p) * arg,
                    this.f(x, p) * b * 2 * (v - c)
                ]);
            },
            ddfdpdp: function (x: Vector, p: Vector): Matrix {
                return Matrix.empty(3, 3);
            }
        };

    });
});
