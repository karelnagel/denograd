import { exec } from 'node:child_process'

export const execAsync = (cmd: string, opt?: any) => new Promise<string>((res, rej) => exec(cmd, opt, (error, stdout, stderr) => error || stderr ? rej(error) : res(stdout as any as string)))
export const tiny = async (strings: TemplateStringsArray, ...values: any[]): Promise<any> => {
    const code = `
import tinygrad as tiny
import json
def out(o):
    print(json.dumps(o))

${String.raw({ raw: strings }, ...values)}
`
    const res = await execAsync(`cd /Users/karel/Documents/denograd/tinygrad && python3 -c '${code}'`)
    return JSON.parse(res.trim())
}

const res = await tiny`out("hello")`
console.log(res)
