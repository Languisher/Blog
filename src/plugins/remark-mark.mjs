import { visit } from 'unist-util-visit'

const markPattern = /==([^\n=](?:(?!==)[^\n])*?)==/g

function createMarkNode(children) {
  return {
    type: 'mark',
    data: {
      hName: 'mark',
    },
    children,
  }
}

function splitMarkedText(value) {
  const nodes = []
  let lastIndex = 0

  for (const match of value.matchAll(markPattern)) {
    const matchIndex = match.index ?? 0

    if (matchIndex > lastIndex) {
      nodes.push({ type: 'text', value: value.slice(lastIndex, matchIndex) })
    }

    nodes.push(createMarkNode([{ type: 'text', value: match[1] }]))
    lastIndex = matchIndex + match[0].length
  }

  if (lastIndex < value.length) {
    nodes.push({ type: 'text', value: value.slice(lastIndex) })
  }

  return nodes
}

export function remarkMark() {
  return (tree) => {
    visit(tree, 'text', (node, index, parent) => {
      if (!parent || typeof index !== 'number' || !node.value.includes('==')) {
        return
      }

      const replacement = splitMarkedText(node.value)
      if (replacement.length <= 1) {
        return
      }

      parent.children.splice(index, 1, ...replacement)
      return index + replacement.length
    })
  }
}
