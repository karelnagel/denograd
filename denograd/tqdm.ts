// https://github.com/thesephist/tsqdm/blob/main/src/tqdm.ts

import { env } from './env/index.ts'
import { round } from './helpers.ts'

export type RenderBarOptions = {
  i: number
  label?: string
  size?: number
  width: number
  elapsed: number
  format?: (val: number) => string
}
export type TqdmProgress = {
  i?: number
  label?: string
  size?: number
  elapsed?: number
}
export type TqdmOnProgress = (progress: TqdmProgress) => void

export type TqdmOptions = {
  label?: string
  size?: number
  width?: number
  onProgress?: TqdmOnProgress
  format?: (val: number) => string
}

const markers = ['', '▏', '▎', '▍', '▌', '▋', '▊', '▉', '█']
const filledMarker = markers.at(-1)

function renderBar({ i, label, size, width, elapsed, format = (val) => round(val, 2).toString() }: RenderBarOptions): string {
  const rate = i / elapsed
  if (size === undefined) {
    const graph = `${label ? label + ': ' : ''}${i} | ${elapsed.toFixed(2)}s ${rate.toFixed(2)}it/s`
    if (graph === '' && i > 0) return '▏'
    return graph
  }
  const n = Math.max(i * 8 * width / size, 0)
  const whole = Math.floor(n / 8)
  const rem = Math.round(n % 8)
  const bar = new Array(whole).fill(filledMarker).join('') + markers[rem]
  const gap = new Array(width - bar.length).fill(' ').join('')
  const remaining = (size - i) / rate
  const percent = i / size * 100
  const graph = `${label ? label + ': ' : ''}${percent.toFixed(1)}% |${bar}${gap}| ${format(i)}/${format(size)} | ${elapsed.toFixed(2)}>${remaining.toFixed(2)}s ${format(rate)}/s`
  if (graph === '' && n > 0) return '▏'
  return graph
}

function* arrayToIterableIterator<T>(iter: T[]): IterableIterator<T> {
  yield* iter
}

function* iterator(len = Infinity): IterableIterator<number> {
  for (let i = 0; i < len; i++) yield i
}

export class Tqdm<T> implements IterableIterator<T> {
  private iter: IterableIterator<T>
  private i: number
  private start: number
  private label: string | undefined
  private size: number | undefined
  private width: number
  private format?: (val: number) => string
  private onProgress?: TqdmOnProgress

  constructor(iter: Array<T> | IterableIterator<T> | undefined | number, { label, size, width = 16, onProgress, format }: TqdmOptions = {}) {
    if (typeof iter === 'undefined' || typeof iter === 'number') {
      size = iter
      iter = iterator(iter) as IterableIterator<T>
    } else if (Array.isArray(iter)) {
      size = iter.length
      iter = arrayToIterableIterator(iter)
    }
    this.iter = iter
    this.i = 0
    this.start = Date.now()
    this.label = label
    this.size = size
    this.width = width
    this.format = format
    this.onProgress = onProgress
  }

  private print = (s: string) => env.writeStdout(s)

  set_description = (label: string) => this.label = label

  render = (newI?: number) => {
    if (newI) this.i = newI
    const elapsed = (Date.now() - this.start) / 1000
    this.onProgress?.({ i: this.i, label: this.label, size: this.size, elapsed })
    this.print(renderBar({ i: this.i, label: this.label, size: this.size, width: this.width, elapsed, format: this.format }) + '\x1b[1G')
  }

  next = (): IteratorResult<T> => {
    const result = this.iter.next()
    this.render()
    if (result.done) this.print('\n')
    this.i++
    return result
  };

  [Symbol.iterator] = (): IterableIterator<T> => this
}
