import { argv } from 'node:process'
import { readFile, writeFile } from 'node:fs/promises'

export const main = async () => {
  const inputFile = argv[2]
  const ouptutFile = argv[3]
  if (!inputFile || !ouptutFile) throw new Error('Input or output file missing!')

  let code = (await readFile(inputFile)).toString()

  code = code.replace(/def (.*?)\(self,? ?/g, 'def $1(')

  //   functions
  code = code.replaceAll('.append(', '.push(')
  code = code.replaceAll('.startswith(', '.startsWith(')
  code = code.replace(/len\(((?:[^()]|\([^()]*\))*)\)/g, '$1.length')
  code = code.replace(/enumerate\(((?:[^()]|\([^()]*\))*)\)/g, '$1.entries()')
  code = code.replace(/assert (.*?)\n/g, 'assert($1)\n')

  code = code.replace(/\s*"""([\s\S]*?)"""/g, (_, content) => {
    const clean = content.trim().split('\n').map((line: string) => line.trim()).join('\n * ')
    return `\n/**\n * ${clean}\n */`
  })

  code = code.replace(/for (.*?) in (.*?): /g, ' for (const $1 of $2){ ')

  code = code.replace(/ (.*?) not in \[(.*?)\] /g, ' ![($2)].includes($1) ')
  code = code.replace(/ (.*?) in \[(.*?)\] /g, ' [($2)].includes($1) ')
  code = code.replace(/ (.*?) not in \{(.*?)\} /g, ' ![($2)].includes($1) ')
  code = code.replace(/ (.*?) in \{(.*?)\} /g, ' [($2)].includes($1) ')

  // TYPES
  code = code.replace(/: ?Optional\[(.*?)\] ?= ?None/g, '?: $1')
  code = code.replace(/: ?Optional\[(.*?)\]/g, '?: $1')

  code = code.replace(/: ?Tuple\[(.*?), ?...\]/g, ': $1[]')
  code = code.replace(/: ?Tuple\[(.*?)\]/g, ': [$1]')
  code = code.replace(/: ?List\[(.*?)\]/g, ': $1[]')
  code = code.replace(/: ?Dict\[(.*?)\]/g, ': Map<$1>')

  code = code.replace(/: ?int/, ':number')
  code = code.replace(/: ?float/, ':number')
  code = code.replace(/: ?str/, ':string')
  code = code.replace(/: ?bool/, ':boolean')
  code = code.replace(/: ?Any/, ':any')

  code = code.replace(':=', '696996969696')
  code = code.replace(/if (.*?): /g, 'if ($1) ')
  code = code.replace(/elif (.*?): /g, 'else if ($1) ')
  code = code.replace(/else: /g, 'else ')
  code = code.replace('696996969696', ':=')

  code = code.replace(/class (.*?)\((.*?)\):/, 'class $1 extends $2 {')
  code = code.replace(/class (.*?):/, 'class $1 {')

  code = code.replace(/self/g, 'this')
  code = code.replace(/None/g, 'undefined')
  code = code.replace(/False/g, 'false')
  code = code.replace(/True/g, 'true')
  code = code.replaceAll(' and ', ' && ')
  code = code.replaceAll(' or ', ' || ')
  code = code.replaceAll(' is not ', ' !== ')
  code = code.replaceAll(' is ', ' == ')
  code = code.replaceAll('not ', '!')
  code = code.replaceAll(' != ', ' !== ')
  code = code.replaceAll(' == ', ' === ')

  code = code.replaceAll(/f"(.*?)"/g, (_, content) => {
    return '`' + content.replace(/{([^}]*)}/g, '${$1}') + '`'
  })

  // one liner functions
  code = code.replace(/def (.*?)\((.*?)\) ?-> ?(.*?): return/g, 'const $1 = ($2): $3 => ')
  code = code.replace(/def (.*?)\((.*?)\) ?: return/g, 'const $1 = ($2) => ')

  code = code.replace(/def (.*?)\((.*?)\) ?-> ?(.*?):/g, 'const $1 = ($2): $3 => {')
  code = code.replace(/def (.*?)\((.*?)\) ?:/g, 'const $1 = ($2) => {')

  code = code.replace(/def (.*?)\(/g, 'const $1 = (')

  code = code.replace(/\[-(.*?)\]/g, '.at(-$1)!')

  code = code.replaceAll('print(', 'console.log(')
  code = code.replaceAll('#', '//')

  await writeFile(ouptutFile, code)
}

main()
