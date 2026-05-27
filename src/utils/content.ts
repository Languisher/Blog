import type { CollectionEntry } from 'astro:content'
import type { Language } from '@/i18n/config'
import type { Post } from '@/types'
import { getCollection, render } from 'astro:content'
import { defaultLocale } from '@/config'
import { getPostPath } from '@/i18n/path'
import { memoize } from '@/utils/cache'
import categoriesSource from '../data/categories.toml?raw'

const metaCache = new Map<string, { minutes: number }>()

/**
 * Add metadata including reading time to a post
 *
 * @param post The post to enhance with metadata
 * @returns Enhanced post with reading time information
 */
async function addMetaToPost(post: CollectionEntry<'posts'>): Promise<Post> {
  const cacheKey = `${post.id}-${post.data.lang || 'universal'}`
  const cachedMeta = metaCache.get(cacheKey)
  if (cachedMeta) {
    return {
      ...post,
      remarkPluginFrontmatter: cachedMeta,
    }
  }

  const { remarkPluginFrontmatter } = await render(post)
  const meta = remarkPluginFrontmatter as { minutes: number }
  metaCache.set(cacheKey, meta)

  return {
    ...post,
    remarkPluginFrontmatter: meta,
  }
}

/**
 * Find duplicate post slugs within the same language
 *
 * @param posts Array of blog posts to check
 * @returns Array of descriptive error messages for duplicate slugs
 */
export async function checkPostSlugDuplication(posts: CollectionEntry<'posts'>[]): Promise<string[]> {
  const slugMap = new Map<string, Set<string>>()
  const duplicates: string[] = []

  posts.forEach((post) => {
    const lang = post.data.lang
    const slug = post.data.abbrlink || post.id

    let slugSet = slugMap.get(lang)
    if (!slugSet) {
      slugSet = new Set()
      slugMap.set(lang, slugSet)
    }

    if (!slugSet.has(slug)) {
      slugSet.add(slug)
      return
    }

    if (!lang) {
      duplicates.push(`Duplicate slug "${slug}" found in universal post (applies to all languages)`)
    }
    else {
      duplicates.push(`Duplicate slug "${slug}" found in "${lang}" language post`)
    }
  })

  return duplicates
}

/**
 * Get all posts (including pinned ones, excluding drafts in production)
 *
 * @param lang The language code to filter by, defaults to site's default language
 * @returns Posts filtered by language, enhanced with metadata, sorted by date
 */
async function _getPosts(lang?: Language) {
  const currentLang = lang || defaultLocale

  const filteredPosts = await getCollection(
    'posts',
    ({ data }: CollectionEntry<'posts'>) => {
      // Show drafts in dev mode only
      const shouldInclude = import.meta.env.DEV || !data.draft
      return shouldInclude && (data.lang === currentLang || data.lang === '')
    },
  )

  const enhancedPosts = await Promise.all(filteredPosts.map(addMetaToPost))

  return enhancedPosts.sort((a, b) =>
    b.data.published.valueOf() - a.data.published.valueOf(),
  )
}

export const getPosts = memoize(_getPosts)

/**
 * Get all non-pinned posts
 *
 * @param lang The language code to filter by, defaults to site's default language
 * @returns Regular posts (non-pinned), filtered by language
 */
async function _getRegularPosts(lang?: Language) {
  const posts = await getPosts(lang)
  return posts.filter(post => !post.data.pin)
}

export const getRegularPosts = memoize(_getRegularPosts)

/**
 * Get pinned posts sorted by pin priority
 *
 * @param lang The language code to filter by, defaults to site's default language
 * @returns Pinned posts sorted by pin value in descending order
 */
async function _getPinnedPosts(lang?: Language) {
  const posts = await getPosts(lang)
  return posts
    .filter(post => post.data.pin && post.data.pin > 0)
    .sort((a, b) => (b.data.pin ?? 0) - (a.data.pin ?? 0))
}

export const getPinnedPosts = memoize(_getPinnedPosts)

/**
 * Group posts by year and sort within each year
 *
 * @param lang The language code to filter by, defaults to site's default language
 * @returns Map of posts grouped by year (descending), sorted by date within each year
 */
async function _getPostsByYear(lang?: Language): Promise<Map<number, Post[]>> {
  const posts = await getRegularPosts(lang)
  const yearMap = new Map<number, Post[]>()

  posts.forEach((post: Post) => {
    const year = post.data.published.getFullYear()
    let yearPosts = yearMap.get(year)
    if (!yearPosts) {
      yearPosts = []
      yearMap.set(year, yearPosts)
    }
    yearPosts.push(post)
  })

  // Sort posts within each year by date
  yearMap.forEach((yearPosts) => {
    yearPosts.sort((a, b) => {
      const aDate = a.data.published
      const bDate = b.data.published
      return bDate.getMonth() - aDate.getMonth() || bDate.getDate() - aDate.getDate()
    })
  })

  return new Map([...yearMap.entries()].sort((a, b) => b[0] - a[0]))
}

export const getPostsByYear = memoize(_getPostsByYear)

/**
 * Group posts by their tags
 *
 * @param lang The language code to filter by, defaults to site's default language
 * @returns Map where keys are tag names and values are arrays of posts with that tag
 */
async function _getPostsGroupByTags(lang?: Language) {
  const posts = await getPosts(lang)
  const tagMap = new Map<string, Post[]>()

  posts.forEach((post: Post) => {
    post.data.tags?.forEach((tag: string) => {
      let tagPosts = tagMap.get(tag)
      if (!tagPosts) {
        tagPosts = []
        tagMap.set(tag, tagPosts)
      }
      tagPosts.push(post)
    })
  })

  return tagMap
}

export const getPostsGroupByTags = memoize(_getPostsGroupByTags)

/**
 * Get all tags sorted by post count
 *
 * @param lang The language code to filter by, defaults to site's default language
 * @returns Array of tags sorted by popularity (most posts first)
 */
async function _getAllTags(lang?: Language) {
  const tagMap = await getPostsGroupByTags(lang)
  const tagsWithCount = Array.from(tagMap.entries())

  tagsWithCount.sort((a, b) => b[1].length - a[1].length)
  return tagsWithCount.map(([tag]) => tag)
}

export const getAllTags = memoize(_getAllTags)

/**
 * Get all posts that contain a specific tag
 *
 * @param tag The tag name to filter posts by
 * @param lang The language code to filter by, defaults to site's default language
 * @returns Array of posts that contain the specified tag
 */
async function _getPostsByTag(tag: string, lang?: Language) {
  const tagMap = await getPostsGroupByTags(lang)
  return tagMap.get(tag) ?? []
}

export const getPostsByTag = memoize(_getPostsByTag)

export interface CategoryNode {
  name: string
  slug: string
  aliases: string[]
  children: CategoryNode[]
  depth: number
}

export interface CategorySummary extends CategoryNode {
  count: number
}

interface RawCategoryNode {
  name: string
  slug?: string
  aliases?: string[]
  children?: RawCategoryNode[]
}

function slugifyCategory(name: string) {
  return encodeURIComponent(name.trim().toLowerCase().replace(/\s+/g, '-'))
}

function parseTomlString(value: string) {
  return JSON.parse(value) as string
}

function parseTomlStringArray(value: string) {
  const strings = value.match(/"(?:\\.|[^"])*"/g) ?? []
  return strings.map(parseTomlString)
}

function parseCategoryConfig() {
  const roots: RawCategoryNode[] = []
  const stack: RawCategoryNode[] = []

  let currentNode: RawCategoryNode | undefined

  categoriesSource.split(/\r?\n/).forEach((rawLine) => {
    const line = rawLine.trim()

    if (!line || line.startsWith('#')) {
      return
    }

    const headerMatch = line.match(/^\[\[(categories(?:\.children)*)\]\]$/)
    if (headerMatch) {
      const depth = (headerMatch[1].match(/children/g) ?? []).length
      if (depth > 1) {
        throw new Error(`Category hierarchy supports at most two levels. Found deeper category near "${line}"`)
      }

      const node: RawCategoryNode = {
        name: '',
        children: [],
      }

      if (depth === 0) {
        roots.push(node)
      }
      else {
        const parent = stack[depth - 1]
        if (!parent) {
          throw new Error(`Invalid category TOML hierarchy near "${line}"`)
        }
        parent.children = parent.children ?? []
        parent.children.push(node)
      }

      stack[depth] = node
      stack.length = depth + 1
      currentNode = node
      return
    }

    if (!currentNode) {
      return
    }

    const separatorIndex = line.indexOf('=')
    if (separatorIndex === -1) {
      return
    }

    const key = line.slice(0, separatorIndex).trim()
    const value = line.slice(separatorIndex + 1).trim()

    if (!/^[a-z][\w-]*$/i.test(key)) {
      return
    }

    if (key === 'name' && value.startsWith('"')) {
      currentNode.name = parseTomlString(value)
    }
    else if (key === 'slug' && value.startsWith('"')) {
      currentNode.slug = parseTomlString(value)
    }
    else if (key === 'aliases' && value.startsWith('[')) {
      currentNode.aliases = parseTomlStringArray(value)
    }
  })

  return roots
}

function normalizeCategoryNodes(nodes: RawCategoryNode[], depth = 0): CategoryNode[] {
  if (nodes.length > 0 && depth > 1) {
    throw new Error('Category hierarchy supports at most two levels')
  }

  return nodes.filter(node => node.name).map((node) => {
    const slug = node.slug?.trim() || slugifyCategory(node.name)

    return {
      name: node.name,
      slug,
      aliases: node.aliases ?? [],
      depth,
      children: normalizeCategoryNodes(node.children ?? [], depth + 1),
    }
  })
}

function flattenCategoryNodes(nodes: CategoryNode[]): CategoryNode[] {
  return nodes.flatMap(node => [node, ...flattenCategoryNodes(node.children)])
}

async function _getCategoryTree() {
  return normalizeCategoryNodes(parseCategoryConfig())
}

export const getCategoryTree = memoize(_getCategoryTree)

async function _getAllCategories() {
  return flattenCategoryNodes(await getCategoryTree())
}

export const getAllCategories = memoize(_getAllCategories)

async function getCategoryByInput(category: string) {
  const normalized = category.trim()
  const categories = await getAllCategories()

  return categories.find(item =>
    item.slug === normalized
    || item.name === normalized
    || item.aliases.includes(normalized),
  )
}

async function getCategoryPostKey(post: Post) {
  const categories = await getAllCategories()
  const categoryByInput = new Map<string, CategoryNode>()

  categories.forEach((category) => {
    categoryByInput.set(category.slug, category)
    categoryByInput.set(category.name, category)
    category.aliases.forEach(alias => categoryByInput.set(alias, category))
  })

  const candidates = [
    post.data.category,
    post.id.split('/')[0] ?? '',
  ].map(candidate => candidate.trim()).filter(Boolean)

  for (const candidate of candidates) {
    const configuredCategory = categoryByInput.get(candidate)

    if (configuredCategory) {
      return configuredCategory.slug
    }
  }

  return candidates[0] ? slugifyCategory(candidates[0]) : ''
}

function getPostCategoryInput(post: Post) {
  return post.data.category || post.id.split('/')[0] || ''
}

function getCategoryDescendantSlugs(category: CategoryNode): string[] {
  return category.children.flatMap(child => [child.slug, ...getCategoryDescendantSlugs(child)])
}

/**
 * Group posts by their configured category.
 *
 * @param lang The language code to filter by, defaults to site's default language
 * @returns Map where keys are category slugs and values are arrays of posts
 */
async function _getPostsGroupByCategories(lang?: Language) {
  const posts = await getPosts(lang)
  const categoryMap = new Map<string, Post[]>()

  for (const post of posts) {
    const categoryInput = getPostCategoryInput(post)

    if (!categoryInput) {
      continue
    }

    const categoryKey = await getCategoryPostKey(post)

    if (!categoryKey) {
      continue
    }

    const categoryPosts = categoryMap.get(categoryKey) ?? []
    categoryPosts.push(post)
    categoryMap.set(categoryKey, categoryPosts)
  }

  return categoryMap
}

export const getPostsGroupByCategories = memoize(_getPostsGroupByCategories)

/**
 * Get categories from the configured tree with post counts.
 *
 * @param lang The language code to filter by, defaults to site's default language
 * @returns Flattened category summaries in configured tree order
 */
async function _getAllCategoriesWithCount(lang?: Language): Promise<CategorySummary[]> {
  const categoryTree = await getCategoryTree()
  const configuredCategories = flattenCategoryNodes(categoryTree)
  const categoryMap = await getPostsGroupByCategories(lang)
  const configuredSlugs = new Set(configuredCategories.map(category => category.slug))
  const summaries = configuredCategories.map(category => ({
    ...category,
    count: [
      category.slug,
      ...getCategoryDescendantSlugs(category),
    ].reduce((count, slug) => count + (categoryMap.get(slug)?.length ?? 0), 0),
  }))

  for (const [slug, posts] of categoryMap.entries()) {
    if (configuredSlugs.has(slug)) {
      continue
    }

    summaries.push({
      name: getPostCategoryInput(posts[0]!) || slug,
      slug,
      aliases: [],
      children: [],
      depth: 0,
      count: posts.length,
    })
  }

  return summaries.filter(category => category.count > 0)
}

export const getAllCategoriesWithCount = memoize(_getAllCategoriesWithCount)

/**
 * Get all posts that belong to a specific category slug.
 *
 * @param categorySlug The category slug to filter posts by
 * @param lang The language code to filter by, defaults to site's default language
 * @returns Array of posts that belong to the category
 */
async function _getPostsByCategory(categorySlug: string, lang?: Language) {
  const categoryMap = await getPostsGroupByCategories(lang)
  const category = await getCategoryByInput(categorySlug)
  const categorySlugs = category
    ? [category.slug, ...getCategoryDescendantSlugs(category)]
    : [categorySlug]
  const posts = categorySlugs.flatMap(slug => categoryMap.get(slug) ?? [])

  return Array.from(new Map(posts.map(post => [post.id, post])).values())
}

export const getPostsByCategory = memoize(_getPostsByCategory)

/**
 * Get all tags with post counts.
 *
 * @param lang The language code to filter by, defaults to site's default language
 * @returns Array of tags and counts sorted by popularity
 */
async function _getAllTagsWithCount(lang?: Language) {
  const tagMap = await getPostsGroupByTags(lang)
  return Array.from(tagMap.entries())
    .sort((a, b) => b[1].length - a[1].length)
    .map(([name, posts]) => ({
      name,
      count: posts.length,
    }))
}

export const getAllTagsWithCount = memoize(_getAllTagsWithCount)

function splitLinkTarget(url: string) {
  const match = url.match(/^([^?#]*)([?#].*)?$/)
  return {
    pathname: match?.[1] ?? url,
  }
}

function isMarkdownFileLink(pathname: string) {
  if (!pathname || pathname.startsWith('/') || pathname.startsWith('#') || pathname.startsWith('?')) {
    return false
  }

  if (/^[a-z][a-z\d+.-]*:/i.test(pathname)) {
    return false
  }

  return /\.(?:md|mdx)$/i.test(pathname)
}

function decodeLinkPath(pathname: string) {
  try {
    return decodeURI(pathname)
  }
  catch {
    return pathname
  }
}

function normalizeRawLinkUrl(url: string) {
  const trimmedUrl = url.trim()
  return trimmedUrl.startsWith('<') && trimmedUrl.endsWith('>')
    ? trimmedUrl.slice(1, -1).trim()
    : trimmedUrl
}

function getMarkdownLinkUrls(markdown: string) {
  const urls = new Set<string>()
  const markdownLinkPattern = /(!?)\[[^\]\n]*\]\((<[^>\n]+>|(?:\\.|[^()\\\n]|\([^()\n]*\))+)\)/g
  const referenceDefinitionPattern = /^[ \t]{0,3}\[([^\]\n]+)\]:[ \t]*(<[^>\n]+>|[^ \t\n]+)(?:[ \t].*)?$/gm
  const referenceLinkPattern = /(!?)\[[^\]\n]+\]\[([^\]\n]*)\]/g
  const htmlLinkPattern = /\bhref=["']([^"']+)["']/g
  const referenceDefinitions = new Map<string, string>()

  for (const match of markdown.matchAll(referenceDefinitionPattern)) {
    const label = match[1]?.trim().toLowerCase()
    const rawUrl = match[2]
    if (label && rawUrl) {
      referenceDefinitions.set(label, normalizeRawLinkUrl(rawUrl))
    }
  }

  for (const match of markdown.matchAll(markdownLinkPattern)) {
    if (match[1] === '!') {
      continue
    }

    const rawUrl = match[2]?.trim()
    if (rawUrl) {
      urls.add(normalizeRawLinkUrl(rawUrl))
    }
  }

  for (const match of markdown.matchAll(referenceLinkPattern)) {
    if (match[1] === '!') {
      continue
    }

    const label = (match[2] || match[0].replace(/^\[|\]\[\]$/g, '')).trim().toLowerCase()
    const rawUrl = referenceDefinitions.get(label)
    if (rawUrl) {
      urls.add(rawUrl)
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

function stripMarkdownExtension(filePath: string) {
  return filePath.replace(/\.(md|mdx)$/i, '')
}

function normalizeRoutePath(pathname: string) {
  let normalizedPath = decodeLinkPath(pathname)
  normalizedPath = normalizedPath.replace(/\/index(?:\.html)?$/i, '/')
  return normalizedPath.endsWith('/') ? normalizedPath : `${normalizedPath}/`
}

function resolveMarkdownPostId(url: string, sourcePostId: string, postIds: Set<string>) {
  const { pathname } = splitLinkTarget(url)
  if (!isMarkdownFileLink(pathname)) {
    return null
  }

  const resolvedSegments = decodeLinkPath(pathname).split('/').reduce((segments, segment) => {
    if (!segment || segment === '.') {
      return segments
    }
    if (segment === '..') {
      segments.pop()
      return segments
    }
    segments.push(segment)
    return segments
  }, sourcePostId.split('/').slice(0, -1))
  const resolvedPath = stripMarkdownExtension(resolvedSegments.join('/'))

  return postIds.has(resolvedPath) ? resolvedPath : null
}

function getPostRouteLookup(posts: Post[], lang: Language) {
  const lookup = new Map<string, string>()

  for (const post of posts) {
    const slug = post.data.abbrlink || post.id
    const routePath = normalizeRoutePath(getPostPath(slug, lang))
    lookup.set(routePath, post.id)

    const decodedRoutePath = normalizeRoutePath(decodeLinkPath(routePath))
    lookup.set(decodedRoutePath, post.id)
  }

  return lookup
}

function resolveRoutedPostId(url: string, sourcePost: Post, posts: Post[], lang: Language) {
  const { pathname } = splitLinkTarget(url)
  if (!pathname || pathname.startsWith('#') || pathname.startsWith('?') || /^[a-z][a-z\d+.-]*:/i.test(pathname)) {
    return null
  }

  const sourceSlug = sourcePost.data.abbrlink || sourcePost.id
  const sourcePath = getPostPath(sourceSlug, lang)
  const resolvedPath = new URL(pathname, `https://example.com${sourcePath}`).pathname
  const normalizedPath = normalizeRoutePath(resolvedPath)
  const routeLookup = getPostRouteLookup(posts, lang)

  return routeLookup.get(normalizedPath) ?? null
}

function getPostLinkSource(post: Post) {
  const renderedHtml = (post as Post & { rendered?: { html?: string } }).rendered?.html ?? ''
  return `${post.body ?? ''}\n${renderedHtml}`
}

function getLinkedPostIds(post: Post, posts: Post[], lang: Language) {
  const postIds = new Set(posts.map(post => post.id))
  const linkedPostIds = new Set<string>()

  for (const url of getMarkdownLinkUrls(getPostLinkSource(post))) {
    const postId = resolveMarkdownPostId(url, post.id, postIds)
      ?? resolveRoutedPostId(url, post, posts, lang)

    if (postId && postId !== post.id) {
      linkedPostIds.add(postId)
    }
  }

  return linkedPostIds
}

export interface RelatedPosts {
  backlinks: Post[]
  outgoing: Post[]
  tagGroups: {
    tag: string
    posts: Post[]
  }[]
}

/**
 * Get posts related to the current post by internal links and shared tags.
 *
 * @param currentPost The post currently being rendered
 * @param lang The language code to filter by
 * @returns Related posts grouped by backlinks, outgoing links, and tags
 */
export async function getRelatedPosts(currentPost: CollectionEntry<'posts'>, lang: Language): Promise<RelatedPosts> {
  const posts = await getPosts(lang)
  const current = posts.find(post => post.id === currentPost.id)

  if (!current) {
    return {
      backlinks: [],
      outgoing: [],
      tagGroups: [],
    }
  }

  const postById = new Map(posts.map(post => [post.id, post]))
  const linkedIdsByPost = new Map(posts.map(post => [post.id, getLinkedPostIds(post, posts, lang)]))
  const outgoingIds = linkedIdsByPost.get(current.id) ?? new Set<string>()

  const outgoing = posts.filter(post => outgoingIds.has(post.id))
  const backlinks = posts.filter(post => linkedIdsByPost.get(post.id)?.has(current.id))
  const tagGroups = current.data.tags.map(tag => ({
    tag,
    posts: posts.filter(post => post.id !== current.id && post.data.tags.includes(tag)),
  })).filter(group => group.posts.length > 0)

  return {
    backlinks: backlinks.filter(post => postById.has(post.id)),
    outgoing: outgoing.filter(post => postById.has(post.id)),
    tagGroups,
  }
}

/**
 * Check which languages support a specific tag
 *
 * @param tag The tag name to check language support for
 * @returns Array of language codes that support the specified tag
 */
async function _getTagSupportedLangs(tag: string): Promise<Language[]> {
  const posts = await getCollection(
    'posts',
    ({ data }) => !data.draft,
  )
  const { allLocales } = await import('@/config')

  return allLocales.filter(locale =>
    posts.some(post =>
      post.data.tags?.includes(tag)
      && (post.data.lang === locale || post.data.lang === ''),
    ),
  )
}

export const getTagSupportedLangs = memoize(_getTagSupportedLangs)
