import fs from 'node:fs'
import path from 'node:path'
import fg from 'fast-glob'

function normalizeFilePath(filePath) {
  return path.normalize(filePath)
}

function normalizeRoutePath(routePath) {
  return routePath.replace(/\\/g, '/')
}

function stripMarkdownExtension(filePath) {
  return filePath.replace(/\.(md|mdx)$/i, '')
}

function parseFrontmatterValue(value) {
  const trimmedValue = value.trim()

  if (
    (trimmedValue.startsWith('"') && trimmedValue.endsWith('"'))
    || (trimmedValue.startsWith('\'') && trimmedValue.endsWith('\''))
  ) {
    return trimmedValue.slice(1, -1)
  }

  return trimmedValue
}

function parseFrontmatter(source) {
  const frontmatterMatch = source.match(/^---\r?\n([\s\S]*?)\r?\n---/)
  if (!frontmatterMatch) {
    return {}
  }

  const frontmatter = {}

  for (const line of frontmatterMatch[1].split(/\r?\n/)) {
    const match = line.match(/^([A-Za-z0-9_-]+):\s*(.*)$/)
    if (!match) {
      continue
    }

    const [, key, rawValue] = match
    frontmatter[key] = parseFrontmatterValue(rawValue)
  }

  return frontmatter
}

function buildPostPath(slug, lang, { base = '', defaultLocale }) {
  const langPrefix = !lang || lang === defaultLocale ? '' : `/${lang}`
  const postPath = `${langPrefix}/posts/${slug}/`
  return base ? `${base}${postPath}` : postPath
}

export function createPostReferenceRegistry(postsDir) {
  const registry = new Map()
  const postFiles = fg.sync('**/*.{md,mdx}', {
    cwd: postsDir,
    absolute: false,
  })

  for (const relativeFilePath of postFiles) {
    const absoluteFilePath = normalizeFilePath(path.resolve(postsDir, relativeFilePath))
    const source = fs.readFileSync(absoluteFilePath, 'utf8')
    const frontmatter = parseFrontmatter(source)
    const postId = normalizeRoutePath(stripMarkdownExtension(relativeFilePath))

    registry.set(absoluteFilePath, {
      filePath: absoluteFilePath,
      id: postId,
      slug: frontmatter.abbrlink || postId,
      lang: frontmatter.lang || '',
    })
  }

  return registry
}

function splitLinkTarget(url) {
  const match = url.match(/^([^?#]*)([?#].*)?$/)
  return {
    pathname: match?.[1] ?? url,
    suffix: match?.[2] ?? '',
  }
}

function isMarkdownFileLink(pathname) {
  if (!pathname || pathname.startsWith('/') || pathname.startsWith('#') || pathname.startsWith('?')) {
    return false
  }

  if (/^[a-z][a-z\d+.-]*:/i.test(pathname)) {
    return false
  }

  return /\.(md|mdx)$/i.test(pathname)
}

function decodeLinkPath(pathname) {
  try {
    return decodeURI(pathname)
  }
  catch {
    return pathname
  }
}

export function resolvePostReferenceUrl(url, sourceFilePath, registry, options) {
  const { pathname, suffix } = splitLinkTarget(url)
  if (!isMarkdownFileLink(pathname)) {
    return null
  }

  const source = registry.get(normalizeFilePath(sourceFilePath))
  if (!source) {
    return null
  }

  const resolvedFilePath = normalizeFilePath(path.resolve(path.dirname(source.filePath), decodeLinkPath(pathname)))
  const target = registry.get(resolvedFilePath)
  if (!target) {
    return null
  }

  if (source.lang && target.lang && source.lang !== target.lang) {
    return `${buildPostPath(target.slug, target.lang, options)}${suffix}`
  }

  if (!source.lang && target.lang) {
    return `${buildPostPath(target.slug, target.lang, options)}${suffix}`
  }

  return `../${target.slug}/${suffix}`
}
