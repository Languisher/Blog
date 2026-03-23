import { visit } from 'unist-util-visit'
import { attachHeadingNumbers } from '../utils/heading-numbering.mjs'

export function rehypeHeadingNumbering() {
  return (tree) => {
    const headingNodes = []

    visit(tree, 'element', (node) => {
      if (/^h[1-6]$/.test(node.tagName)) {
        headingNodes.push(node)
      }
    })

    const numberedHeadings = attachHeadingNumbers(
      headingNodes.map(node => ({ depth: Number.parseInt(node.tagName.slice(1), 10) })),
    )

    headingNodes.forEach((node, index) => {
      const headingNumber = numberedHeadings[index]?.number
      if (!headingNumber) {
        return
      }

      node.properties = {
        ...node.properties,
        dataHeadingNumber: headingNumber,
      }
    })
  }
}
