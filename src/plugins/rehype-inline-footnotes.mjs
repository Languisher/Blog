import { SKIP, visit } from 'unist-util-visit'

const blockTags = new Set([
  'p',
  'blockquote',
  'li',
  'div',
  'section',
  'article',
  'details',
])

function toClassList(className) {
  if (Array.isArray(className)) {
    return className
  }

  if (typeof className === 'string') {
    return className.split(/\s+/).filter(Boolean)
  }

  return []
}

function hasClass(node, className) {
  return toClassList(node?.properties?.className).includes(className)
}

function appendClassName(node, className) {
  node.properties ||= {}

  const classNames = toClassList(node.properties.className)
  if (!classNames.includes(className)) {
    classNames.push(className)
  }

  node.properties.className = classNames
}

function getFootnoteIdsFromHref(href) {
  if (typeof href !== 'string' || !href.startsWith('#')) {
    return []
  }

  const rawId = href.slice(1)
  const decodedId = decodeURIComponent(rawId)

  return rawId === decodedId ? [rawId] : [rawId, decodedId]
}

function isFootnoteSection(node) {
  if (node?.type !== 'element' || node.tagName !== 'section') {
    return false
  }

  return Object.hasOwn(node.properties ?? {}, 'dataFootnotes') || hasClass(node, 'footnotes')
}

function isFootnoteBackref(node) {
  if (node?.type !== 'element' || node.tagName !== 'a') {
    return false
  }

  return Object.hasOwn(node.properties ?? {}, 'dataFootnoteBackref') || hasClass(node, 'data-footnote-backref')
}

function isFootnoteRef(node) {
  if (node?.type !== 'element' || node.tagName !== 'a') {
    return false
  }

  return Object.hasOwn(node.properties ?? {}, 'dataFootnoteRef')
}

function textFromNode(node) {
  if (!node) {
    return ''
  }

  if (node.type === 'text') {
    return node.value
  }

  if (!Array.isArray(node.children)) {
    return ''
  }

  return node.children.map(textFromNode).join('')
}

function cloneWithoutBackrefs(node) {
  if (!node || isFootnoteBackref(node)) {
    return null
  }

  if (!Array.isArray(node.children)) {
    return { ...node }
  }

  const children = node.children
    .map(cloneWithoutBackrefs)
    .filter(Boolean)
    .filter(child => child.type !== 'text' || child.value.trim() !== '')

  return {
    ...node,
    children,
  }
}

function collectFootnotes(tree) {
  const footnotes = new Map()

  visit(tree, 'element', (node, index, parent) => {
    if (!isFootnoteSection(node) || !parent || typeof index !== 'number') {
      return
    }

    visit(node, 'element', (child) => {
      if (child.tagName !== 'li' || typeof child.properties?.id !== 'string') {
        return
      }

      footnotes.set(child.properties.id, child)
    })

    parent.children.splice(index, 1)
    return SKIP
  })

  return footnotes
}

function collectFootnoteUses(root, footnotes) {
  const usesByBlock = new WeakMap()
  const blocks = []
  const seenFootnotes = new Set()

  function walk(node, ancestors) {
    if (!node) {
      return
    }

    if (node.type === 'element' && isFootnoteRef(node)) {
      const id = getFootnoteIdsFromHref(node.properties?.href)
        .find(candidate => footnotes.has(candidate))
      const footnote = id ? footnotes.get(id) : null
      if (footnote && !seenFootnotes.has(id)) {
        const block = [...ancestors].reverse().find(ancestor => blockTags.has(ancestor.node.tagName))
        if (block) {
          const uses = usesByBlock.get(block.node) ?? []
          uses.push({
            id,
            label: textFromNode(node) || String(uses.length + 1),
            footnote,
          })
          if (!usesByBlock.has(block.node)) {
            blocks.push(block.node)
          }
          usesByBlock.set(block.node, uses)
          seenFootnotes.add(id)
        }
      }
    }

    if (!Array.isArray(node.children)) {
      return
    }

    node.children.forEach((child, index) => {
      const nextAncestors = node.type === 'element'
        ? [...ancestors, { node, index }]
        : ancestors

      walk(child, nextAncestors)
    })
  }

  walk(root, [])
  return { blocks, usesByBlock }
}

function createFootnoteAside(uses) {
  return {
    type: 'element',
    tagName: 'aside',
    properties: {
      className: ['inline-footnote'],
      role: 'note',
    },
    children: uses.map(({ id, label, footnote }) => ({
      type: 'element',
      tagName: 'div',
      properties: {
        className: ['inline-footnote__item'],
        id,
      },
      children: [
        {
          type: 'element',
          tagName: 'span',
          properties: { className: ['inline-footnote__marker'] },
          children: [{ type: 'text', value: label }],
        },
        {
          type: 'element',
          tagName: 'div',
          properties: { className: ['inline-footnote__content'] },
          children: footnote.children
            .map(cloneWithoutBackrefs)
            .filter(Boolean),
        },
      ],
    })),
  }
}

export function rehypeInlineFootnotes() {
  return (tree) => {
    const footnotes = collectFootnotes(tree)
    if (footnotes.size === 0) {
      return
    }

    const { blocks, usesByBlock } = collectFootnoteUses(tree, footnotes)
    if (blocks.length === 0) {
      return
    }

    visit(tree, 'element', (node, index, parent) => {
      const uses = usesByBlock.get(node)
      if (!uses || !parent || typeof index !== 'number') {
        return
      }

      appendClassName(node, 'inline-footnote__main')

      parent.children[index] = {
        type: 'element',
        tagName: 'div',
        properties: { className: ['inline-footnote-block'] },
        children: [
          createFootnoteAside(uses),
          node,
        ],
      }

      return SKIP
    })
  }
}
