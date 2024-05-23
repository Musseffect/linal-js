import { complex } from "../complex";
import vec2 from "../dense/vec2";


test('Complex cartesian operations', () => {
    let z1 = new complex(1, 2);
    let z2 = complex.polar(2.0, 1.1);
    expect(z2.length()).toBeCloseTo(2.0);
    expect(z2.arg()).toBeCloseTo(1.1);
    let zDiv = complex.div(z1, z2);
    let zMul = complex.mul(z1, z2);
    let zExp = complex.exp(z1);
    let zLog = complex.log(z1);
    let zPow = complex.pow(z1, z2);
    expect(vec2.near(z1, complex.mul(zDiv, z2))).toBeTruthy();
    expect(vec2.near(z1, complex.div(zMul, z2))).toBeTruthy();
    expect(vec2.near(z1, complex.log(zExp))).toBeTruthy();
    expect(vec2.near(z1, complex.exp(zLog))).toBeTruthy();
    expect(vec2.near(z2, complex.div(complex.log(zPow), complex.log(z1)))).toBeTruthy();
});