// https://github.com/thesephist/tsqdm/blob/main/src/tqdm.ts

import { Env } from './env/index.ts'
import { string_to_bytes } from './helpers.ts'

type RenderBarOptions = {
  i: number
  label?: string
  size?: number
  width: number
  elapsed: number
}

export type TqdmOptions = {
  label?: string
  size?: number
  width?: number
}

const markers = ['', '▏', '▎', '▍', '▌', '▋', '▊', '▉', '█']
const filledMarker = markers.at(-1)

function renderBarWithSize({ i, label, size, width, elapsed }: RenderBarOptions & { size: number }): string {
  const n = Math.max(i * 8 * width / size, 0)
  const whole = Math.floor(n / 8)
  const rem = Math.round(n % 8)
  const bar = new Array(whole).fill(filledMarker).join('') + markers[rem]
  const gap = new Array(width - bar.length).fill(' ').join('')
  const rate = i / elapsed
  const remaining = (size - i) / rate
  const percent = i / size * 100
  const graph = `${label ? label + ': ' : ''}${percent.toFixed(1)}% |${bar}${gap}| ${i}/${size} | ${elapsed.toFixed(2)}>${remaining.toFixed(2)}s ${rate.toFixed(2)}it/s`
  if (graph === '' && n > 0) return '▏'
  return graph
}

function renderBarWithoutSize({ i, label, elapsed }: RenderBarOptions & { size: undefined }): string {
  const rate = i / elapsed
  const graph = `${label ? label + ': ' : ''}${i} | ${elapsed.toFixed(2)}s ${rate.toFixed(2)}it/s`
  if (graph === '' && i > 0) return '▏'
  return graph
}

/**
 * TQDM bar rendering logic extracted out for easy testing and modularity.
 * Renders the full bar string given all necessary inputs.
 */
function renderBar({ size, ...options }: RenderBarOptions): string {
  if (size === undefined) return renderBarWithoutSize({ size: undefined, ...options })
  return renderBarWithSize({ size, ...options })
}

function* arrayToIterableIterator<T>(iter: T[]): IterableIterator<T> {
  yield* iter
}

function isIterableIterator<T>(value: IterableIterator<T> | AsyncIterableIterator<T>): value is IterableIterator<T> {
  return value !== null &&
    typeof (value as IterableIterator<T>)[Symbol.iterator] === 'function' &&
    typeof value.next === 'function'
}

async function* toAsyncIterableIterator<T>(iter: IterableIterator<T>): AsyncIterableIterator<T> {
  for (const it of iter) {
    yield it
  }
}

export class Tqdm<T> implements AsyncIterableIterator<T> {
  private iter: AsyncIterableIterator<T>
  private i: number
  private start: number
  private label: string | undefined
  private size: number | undefined
  private width: number

  constructor(iter: Array<T> | IterableIterator<T> | AsyncIterableIterator<T>, { label, size, width = 16 }: TqdmOptions = {}) {
    if (Array.isArray(iter)) {
      size = iter.length
      iter = arrayToIterableIterator(iter)
    }
    if (isIterableIterator(iter)) {
      iter = toAsyncIterableIterator(iter)
    }
    this.iter = iter
    this.i = 0
    this.start = Date.now()
    this.label = label
    this.size = size
    this.width = width
  }

  private print = async (s: string) => await Env.writeStdout(string_to_bytes(s))

  set_description = (label: string) => this.label = label

  next = async (): Promise<IteratorResult<T>> => {
    const result = await this.iter.next()
    if (!result.done) {
      const elapsed = (Date.now() - this.start) / 1000
      await this.print(renderBar({ i: this.i, label: this.label, size: this.size, width: this.width, elapsed }) + '\x1b[1G')
      this.i++
    } else {
      await this.print('\n')
    }
    return result
  };

  [Symbol.asyncIterator] = (): AsyncIterableIterator<T> => this
}
