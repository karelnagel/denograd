export default {
  name: 'custom-lint-rules',
  rules: {
    'no-floating-promises': {
      create(context: any) {
        const asyncFunctions = new Set<string>()

        return {
          FunctionDeclaration(node: any) {
            if (node.async && node.id) {
              asyncFunctions.add(node.id.name)
            }
          },
          VariableDeclarator(node: any) {
            if (node.init && (node.init.type === 'ArrowFunctionExpression' || node.init.type === 'FunctionExpression') && node.init.async && node.id.type === 'Identifier') {
              asyncFunctions.add(node.id.name)
            }
          },
          CallExpression(node: any) {
            if (node.callee.type === 'Identifier' && asyncFunctions.has(node.callee.name)) {
              const parent = node.parent
              if (parent.type !== 'AwaitExpression' && !(parent.type === 'MemberExpression' && parent.property.type === 'Identifier' && (parent.property.name === 'then' || parent.property.name === 'catch') && parent.parent.type === 'CallExpression')) {
                context.report({
                  node,
                  message: 'Floating promise: await or handle the promise returned by this function call',
                })
              }
            }
          },
        }
      },
    },
    'no-negative-indexes': {
      create(context: any) {
        return {
          MemberExpression(node: any) {
            if (node.property.type === 'UnaryExpression' && node.property.operator === '-' && node.property.argument.type === 'Literal' && typeof node.property.argument.value === 'number') {
              context.report({
                node: node.property,
                message: 'Negative index used in member expression',
              })
            }
          },
        }
      },
    },
    'no-if-array': {
      create(context: any) {
        const strictArraysStack = [new Set<string>()]

        // Helper to identify strict array types (not in union with undefined/null)
        function isStrictArrayType(type: any) {
          return (type.type === 'TSArrayType' || (type.type === 'TSTypeReference' && type.typeName.type === 'Identifier' && type.typeName.name === 'Array'))
        }

        return {
          // Enter function scope and check parameters
          FunctionDeclaration(node: any) {
            strictArraysStack.push(new Set())
            for (const param of node.params) {
              if (param.type === 'Identifier' && param.typeAnnotation && isStrictArrayType(param.typeAnnotation.typeAnnotation)) {
                strictArraysStack[strictArraysStack.length - 1].add(param.name)
              }
            }
          },
          'FunctionDeclaration:exit'(node: any) {
            strictArraysStack.pop()
          },
          // Enter block scope
          BlockStatement(node: any) {
            strictArraysStack.push(new Set())
          },
          'BlockStatement:exit'(node: any) {
            strictArraysStack.pop()
          },
          // Track variable declarations with strict array types
          VariableDeclarator(node: any) {
            if (node.id.type === 'Identifier' && node.id.typeAnnotation && isStrictArrayType(node.id.typeAnnotation.typeAnnotation)) {
              strictArraysStack[strictArraysStack.length - 1].add(node.id.name)
            }
          },
          // Check if statements
          IfStatement(node: any) {
            if (node.test.type === 'Identifier') {
              const name = node.test.name
              for (let i = strictArraysStack.length - 1; i >= 0; i--) {
                if (strictArraysStack[i].has(name)) {
                  context.report({ node: node.test, message: 'Use list.length > 0 instead of if (list) for arrays' })
                  break // Stop at the nearest scope
                }
              }
            }
          },
        }
      },
    },
    'no-null': {
      create(context: any) {
        return {
          Literal(node: any) {
            // deno-lint-ignore custom-lint-rules/no-null
            if (node.value === null) {
              context.report({ node, message: "Prefer 'undefined' over 'null'" })
            }
          },
        }
      },
    },
  },
}
