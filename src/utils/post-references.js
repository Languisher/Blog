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
    const separatorIndex = line.indexOf(':')
    if (separatorIndex === -1) {
      continue
    }

    const key = line.slice(0, separatorIndex)
    if (!/^[\w-]+$/.test(key)) {
      continue
    }

    const rawValue = line.slice(separatorIndex + 1).trimStart()
    frontmatter[key] = parseFrontmatterValue(rawValue)
  }

  return frontmatter
}

function buildPostPath(slug, lang, { base = '', defaultLocale }) {
  const langPrefix = !lang || lang === defaultLocale ? '' : `/${lang}`
  const postPath = `${langPrefix}/posts/${slug}/`
  return base ? `${base}${postPath}` : postPath
}

function buildPostAssetPath(slug, assetPath, { base = '' }) {
  const postAssetPath = `/post-assets/${slug}/${normalizeRoutePath(assetPath)}`
  return base ? `${base}${postAssetPath}` : postAssetPath
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

  return /\.(?:md|mdx)$/i.test(pathname)
}

function isLocalPostAssetLink(pathname) {
  if (!pathname || pathname.startsWith('/') || pathname.startsWith('#') || pathname.startsWith('?')) {
    return false
  }

  if (/^[a-z][a-z\d+.-]*:/i.test(pathname)) {
    return false
  }

  return !/\.(?:md|mdx)$/i.test(pathname)
}

function decodeLinkPath(pathname) {
  try {
    return decodeURI(pathname)
  }
  catch {
    return pathname
  }
}

function getMarkdownLinkUrls(markdown) {
  const urls = new Set()
  const markdownLinkPattern = /(!?)\[[^\]\n]*\]\(([^)\n]+)\)/g
  const htmlLinkPattern = /\bhref=["']([^"']+)["']/g

  for (const match of markdown.matchAll(markdownLinkPattern)) {
    if (match[1] === '!') {
      continue
    }

    const rawUrl = match[2]?.trim()
    if (rawUrl) {
      urls.add(rawUrl.startsWith('<') && rawUrl.endsWith('>') ? rawUrl.slice(1, -1) : rawUrl)
    }
  }

  for (const match of markdown.matchAll(htmlLinkPattern)) {
    const rawUrl = match[1]?.trim()
    if (rawUrl) {
      urls.add(rawUrl)
    }
  }

  return urls
}

function resolvePostAsset(url, sourceFilePath, registry, options) {
  const { pathname, suffix } = splitLinkTarget(url)
  if (!isLocalPostAssetLink(pathname)) {
    return null
  }

  const source = registry.get(normalizeFilePath(sourceFilePath))
  if (!source) {
    return null
  }

  const sourceDir = path.dirname(source.filePath)
  const resolvedAssetPath = normalizeFilePath(path.resolve(sourceDir, decodeLinkPath(pathname)))
  const relativeAssetPath = path.relative(sourceDir, resolvedAssetPath)
  if (relativeAssetPath.startsWith('..') || path.isAbsolute(relativeAssetPath)) {
    return null
  }

  if (!fs.existsSync(resolvedAssetPath) || !fs.statSync(resolvedAssetPath).isFile()) {
    return null
  }

  const postAssetPath = buildPostAssetPath(source.slug, relativeAssetPath, options)

  return {
    filePath: resolvedAssetPath,
    routePath: encodeURI(postAssetPath),
    suffix,
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

export function resolvePostAssetUrl(url, sourceFilePath, registry, options) {
  const asset = resolvePostAsset(url, sourceFilePath, registry, options)
  return asset ? `${asset.routePath}${asset.suffix}` : null
}

export function collectPostAssets(postsDir, options = {}) {
  const registry = createPostReferenceRegistry(postsDir)
  const assets = new Map()

  for (const source of registry.values()) {
    const markdown = fs.readFileSync(source.filePath, 'utf8')
    for (const url of getMarkdownLinkUrls(markdown)) {
      const asset = resolvePostAsset(url, source.filePath, registry, options)
      if (asset) {
        assets.set(asset.routePath, asset.filePath)
      }
    }
  }

  return assets
}
