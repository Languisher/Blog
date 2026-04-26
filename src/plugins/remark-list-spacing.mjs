import { visit } from 'unist-util-visit'

function appendClassName(node, className) {
  node.data ||= {}
  node.data.hProperties ||= {}

  const { className: existingClassName } = node.data.hProperties

  if (Array.isArray(existingClassName)) {
    if (!existingClassName.includes(className)) {
      existingClassName.push(className)
    }
    return
  }

  if (typeof existingClassName === 'string' && existingClassName.trim()) {
    node.data.hProperties.className = `${existingClassName} ${className}`
    return
  }

  node.data.hProperties.className = className
}

export function remarkListSpacing() {
  return (tree) => {
    visit(tree, 'list', (node, index, parent) => {
      if (!parent || typeof index !== 'number' || index <= 0) {
        return
      }

      const previous = parent.children[index - 1]
      const previousEndLine = previous?.position?.end?.line
      const listStartLine = node.position?.start?.line

      if (
        typeof previousEndLine === 'number'
        && typeof listStartLine === 'number'
        && listStartLine === previousEndLine + 1
      ) {
        appendClassName(node, 'list-attached')
      }
    })
  }
}
