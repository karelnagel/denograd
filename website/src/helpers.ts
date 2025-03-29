import fs from 'node:fs/promises'
import { bytes_to_string } from '../../jsgrad/helpers/helpers.ts'

export const getExample = async (file: string) => bytes_to_string(await fs.readFile(`src/examples/${file}`))
export const getExamples = async () => {
  const examples = await fs.readdir('src/examples')
  return await Promise.all(examples.map(async (file) => ({ file, code: await getExample(file) })))
}

export type JSDoc = { doc: string }
export type Location = { filename: string; line: number; col: number; byteIndex: number }
export type ReferenceDef = { target: Location }
export type Element = { name: string; isDefault: boolean; location: Location; declarationKind: string; kind: string; jsDoc?: JSDoc; reference_def?: ReferenceDef }
export type NamespaceDef = { elements: Element[] }
export type Node = { name: string; isDefault: boolean; location: Location; declarationKind: string; kind: string; namespaceDef?: NamespaceDef }
export type Doc = { version: number; nodes: Node[] }
export const getDocs = async () => {
  return JSON.parse(bytes_to_string(await fs.readFile('.astro/docs.json'))) as Doc
}
