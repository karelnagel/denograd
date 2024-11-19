import { assertEquals } from 'jsr:@std/assert@^1.0.8'
import { exec } from 'node:child_process'

export const execAsync = (cmd: string, opt?: any) => new Promise<string>((res, rej) => exec(cmd, opt, (error, stdout, stderr) => error || stderr ? rej(error) : res(stdout as any as string)))
export const python = async (code: string) => JSON.parse((await execAsync(`cd /Users/karel/Documents/denograd/tinygrad && python3 -c '${code}'`)).trim())
export const tiny = async (strings: TemplateStringsArray, ...values: any[]): Promise<any> => {
    const code = `
import tinygrad as tiny
import json
def out(o):
    print(json.dumps(o))

${String.raw({ raw: strings }, ...values)}
`
    return await python(code)
}

export const tinyTest = <T extends any[]>(name: string, inputs: T[], fn: (...args: T) => any, python: (...args: T) => string) => {
    Deno.test(name, async () => {
        for (const input of inputs) {
            assertEquals(fn(...input), await tiny`${python(...input)}`)
        }
    })
}
