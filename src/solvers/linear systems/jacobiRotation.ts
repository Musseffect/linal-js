/** compute c = cos(theta), s = sin(theta) such that
 * [c -s] [a1 b] [c  s] = [d1  0]
 * [s  c] [b a2] [-s c] = [0  d2]
 * @param a1 
 * @param a2 
 * @param b 
 */
export function jacobiRotation(a1: number, a2: number, b: number): { c: number, s: number } {
    if (b == 0) return { c: 1, s: 0 };
    const beta = (a2 - a1) / (2 * b);
    const t = Math.sign(beta) / (Math.abs(beta) + Math.sqrt(beta * beta + 1));
    const c = 1 / Math.sqrt(t * t + 1);
    return { c: c, s: c * t };
}