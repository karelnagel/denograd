import { expect } from 'expect'
import { mnist } from '../../src/nn/datasets.ts'
Deno.test(
  'mnist',
  async () => {
    const [X_train, Y_train, X_test, Y_test] = await mnist()
    expect(X_train.shape).toEqual([60_000, 1, 28, 28])
    expect(Y_train.shape).toEqual([60_000])
    expect(X_test.shape).toEqual([10_000, 1, 28, 28])
    expect(Y_test.shape).toEqual([10_000])

    const train = X_train.get(69).tolist() as any
    expect(train.length).toBe(1)
    for (const row of train[0]) {
      for (const x of row) expect(x).toBeGreaterThanOrEqual(0), expect(x).toBeLessThanOrEqual(255)
    }

    for (const x of Y_train.tolist() as number[]) expect(x).toBeLessThanOrEqual(9), expect(x).toBeGreaterThanOrEqual(0)
    for (const x of Y_test.tolist() as number[]) expect(x).toBeLessThanOrEqual(9), expect(x).toBeGreaterThanOrEqual(0)
  },
)