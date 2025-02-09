const getFiles = (dir = '.'): Iterable<string> => {
  return Deno.readDirSync(dir).flatMap((file) => file.isDirectory ? getFiles(`${dir}/${file.name}`) : file.name === 'deno.json' ? [`${dir}/${file.name}`] : [])
}

const version = Deno.args[0]
if (!version) throw new Error(`No version, use with deno task update 1.0.0`)

for (const file of getFiles()) {
  let json = Deno.readTextFileSync(file)
  json = json.replace(/"version": "[^"]*"/, `"version": "${version}"`)
  Deno.writeTextFileSync(file, json)
}
