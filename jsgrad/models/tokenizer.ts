import { env } from '../env/index.ts'
import { bytes_to_string, range, string_to_bytes } from '../helpers.ts'

export class Tokenizer {
  decode_map: Record<number, string>
  static llama = async (path: string) => {
    const data = await env.readTextFile(path)
    const ranks = data.split('\n').filter(Boolean).map((x) => x.split(' ')).map(([k, v]) => [atob(k), Number(v)])
    const specialTokensList = [
      '<|begin_of_text|>',
      '<|end_of_text|>',
      '<|reserved_special_token_0|>',
      '<|reserved_special_token_1|>',
      '<|reserved_special_token_2|>',
      '<|reserved_special_token_3|>',
      '<|start_header_id|>',
      '<|end_header_id|>',
      '<|reserved_special_token_4|>',
      '<|eot_id|>',
      ...range(5, 256 - 5).map((_, i) => `<|reserved_special_token_${i}|>`),
    ]
    return new Tokenizer(
      new RegExp("'s|'t|'re|'ve|'m|'ll|'d|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+", 'giu'),
      Object.fromEntries(ranks),
      Object.fromEntries(specialTokensList.map((token, i) => [token, ranks.length + i])),
    )
  }
  constructor(public pat: RegExp, public mergeable_ranks: Record<string, number>, public special_tokens: Record<string, number>) {
    this.decode_map = Object.fromEntries(Object.entries(this.mergeable_ranks).map(([token, id]) => [id, token]))
  }

  get bos_id(): number {
    return this.special_tokens['<|begin_of_text|>']
  }

  get stop_tokens() {
    return [this.special_tokens['<|end_of_text|>'], this.special_tokens['<|eot_id|>']]
  }

  decode(toks: number[]): string {
    const byteArrays = toks.filter((t) => t < Object.keys(this.mergeable_ranks).length).map((t) => Uint8Array.from(this.decode_map[t], (c) => c.charCodeAt(0)))

    let allBytes = new Uint8Array()
    for (const curr of byteArrays) {
      const combined = new Uint8Array(allBytes.length + curr.length)
      combined.set(allBytes)
      combined.set(curr, allBytes.length)
      allBytes = combined
    }

    return bytes_to_string(allBytes)
  }

  encode(text: string, allowSpecial: boolean = false): number[] {
    const pieces = [...text.matchAll(this.pat)].map((match) => match[0])

    const tokens: number[] = []
    for (const piece of pieces) {
      if (allowSpecial && piece in this.special_tokens) tokens.push(this.special_tokens[piece])
      else tokens.push(...this.bpe_encode(string_to_bytes(piece)))
    }

    return tokens
  }

  private bpe_encode(bytes: Uint8Array): number[] {
    let tokens: string[] = Array.from(bytes, (b) => String.fromCharCode(b))

    while (true) {
      let minRank: number = Infinity
      let mergePair: [string, string] | undefined

      for (let i = 0; i < tokens.length - 1; i++) {
        const pair = tokens[i] + tokens[i + 1]
        const rank = this.mergeable_ranks[pair]
        if (rank !== undefined && rank < minRank) {
          minRank = rank
          mergePair = [tokens[i], tokens[i + 1]]
        }
      }

      if (mergePair === undefined) break

      const newTokens: string[] = []
      let i = 0
      while (i < tokens.length) {
        if (
          i < tokens.length - 1 &&
          tokens[i] === mergePair[0] &&
          tokens[i + 1] === mergePair[1]
        ) {
          newTokens.push(tokens[i] + tokens[i + 1])
          i += 2
        } else {
          newTokens.push(tokens[i])
          i += 1
        }
      }
      tokens = newTokens
    }

    return tokens.map((token) => this.mergeable_ranks[token])
  }
  encode_role = (role: string) => {
    return [this.special_tokens['<|start_header_id|>'], ...this.encode(role), this.special_tokens['<|end_header_id|>'], ...this.encode('\n\n')]
  }
  encode_message = (role: string, content: string) => {
    return [...this.encode_role(role), ...this.encode(content.trim()), this.special_tokens['<|eot_id|>']]
  }
}
