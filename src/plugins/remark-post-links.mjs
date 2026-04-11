import path from 'node:path'
import { visit } from 'unist-util-visit'
import { createPostReferenceRegistry, resolvePostReferenceUrl } from '../utils/post-references.js'

export function remarkPostLinks({ postsDir, base = '', defaultLocale }) {
  return (tree, file) => {
    if (!file.path) {
      return
    }

    const sourceFilePath = path.normalize(file.path)
    const registry = createPostReferenceRegistry(postsDir)

    visit(tree, 'link', (node) => {
      if (typeof node.url !== 'string') {
        return
      }

      const rewrittenUrl = resolvePostReferenceUrl(node.url, sourceFilePath, registry, {
        base,
        defaultLocale,
      })

      if (rewrittenUrl) {
        node.url = rewrittenUrl
      }
    })
  }
}
