import fs from 'node:fs'
import fsp from 'node:fs/promises'
import path from 'node:path'
import mdx from '@astrojs/mdx'
import partytown from '@astrojs/partytown'
import sitemap from '@astrojs/sitemap'
import Compress from 'astro-compress'
import { defineConfig } from 'astro/config'
import rehypeKatex from 'rehype-katex'
import rehypeMermaid from 'rehype-mermaid'
import rehypeSlug from 'rehype-slug'
import remarkDirective from 'remark-directive'
import remarkMath from 'remark-math'
import UnoCSS from 'unocss/astro'
import { base, defaultLocale, themeConfig } from './src/config'
import { langMap } from './src/i18n/config'
import { rehypeCodeCopyButton } from './src/plugins/rehype-code-copy-button.mjs'
import { rehypeExternalLinks } from './src/plugins/rehype-external-links.mjs'
import { rehypeHeadingAnchor } from './src/plugins/rehype-heading-anchor.mjs'
import { rehypeHeadingNumbering } from './src/plugins/rehype-heading-numbering.mjs'
import { rehypeImageProcessor } from './src/plugins/rehype-image-processor.mjs'
import { rehypeInlineFootnotes } from './src/plugins/rehype-inline-footnotes.mjs'
import { remarkContainerDirectives } from './src/plugins/remark-container-directives.mjs'
import { remarkLeafDirectives } from './src/plugins/remark-leaf-directives.mjs'
import { remarkListSpacing } from './src/plugins/remark-list-spacing.mjs'
import { remarkMark } from './src/plugins/remark-mark.mjs'
import { remarkPostLinks } from './src/plugins/remark-post-links.mjs'
import { remarkReadingTime } from './src/plugins/remark-reading-time.mjs'
import { collectPostAssets } from './src/utils/post-references.js'

const { url: site } = themeConfig.site
const { imageHostURL } = themeConfig.preload ?? {}
const imageConfig = imageHostURL
  ? { image: { domains: [imageHostURL], remotePatterns: [{ protocol: 'https' }] } }
  : {}
const postsDir = path.resolve('./src/content/posts')

const contentTypes: Record<string, string> = {
  '.avif': 'image/avif',
  '.gif': 'image/gif',
  '.jpeg': 'image/jpeg',
  '.jpg': 'image/jpeg',
  '.pdf': 'application/pdf',
  '.png': 'image/png',
  '.svg': 'image/svg+xml',
  '.webp': 'image/webp',
}

function getContentType(filePath: string) {
  return contentTypes[path.extname(filePath).toLowerCase()] ?? 'application/octet-stream'
}

interface DevServer {
  middlewares: {
    use: (
      handler: (
        req: { url?: string },
        res: NodeJS.WritableStream & {
          statusCode: number
          setHeader: (name: string, value: string) => void
        },
        next: () => void,
      ) => void,
    ) => void
  }
  watcher: {
    add: (filePath: string) => void
    on: (event: string, handler: (event: string, filePath: string) => void) => void
  }
}

function postAssetsPlugin() {
  let assets = new Map<string, string>()

  function refreshAssets() {
    assets = collectPostAssets(postsDir)
  }

  return {
    name: 'post-assets',
    buildStart() {
      refreshAssets()
    },
    configureServer(server: DevServer) {
      refreshAssets()

      server.watcher.add(postsDir)
      server.watcher.on('all', (_event, filePath) => {
        if (path.resolve(filePath).startsWith(postsDir)) {
          refreshAssets()
        }
      })

      server.middlewares.use((req, res, next) => {
        const requestPath = decodeURI((req.url ?? '').split('?')[0] ?? '')
        const routePath = base && requestPath.startsWith(base)
          ? requestPath.slice(base.length)
          : requestPath
        const assetPath = assets.get(encodeURI(routePath))

        if (!assetPath) {
          next()
          return
        }

        res.statusCode = 200
        res.setHeader('Content-Type', getContentType(assetPath))
        fs.createReadStream(assetPath).pipe(res)
      })
    },
    async closeBundle() {
      refreshAssets()

      await Promise.all(Array.from(assets.entries()).map(async ([routePath, assetPath]) => {
        const outputPath = path.join('dist', decodeURI(routePath))
        await fsp.mkdir(path.dirname(outputPath), { recursive: true })
        await fsp.copyFile(assetPath, outputPath)
      }))
    },
  }
}

export default defineConfig({
  site,
  base,
  trailingSlash: 'always', // Not recommended to change
  prefetch: {
    prefetchAll: true,
    defaultStrategy: 'viewport', // hover, tap, viewport, load
  },
  ...imageConfig,
  i18n: {
    locales: Object.entries(langMap).map(([path, codes]) => ({
      path,
      codes: [...codes] as [string, ...string[]],
    })),
    defaultLocale,
  },
  integrations: [
    UnoCSS({
      injectReset: true,
    }),
    mdx(),
    partytown({
      config: {
        forward: ['dataLayer.push', 'gtag'],
      },
    }),
    sitemap(),
    Compress({
      CSS: true,
      HTML: true,
      Image: false,
      JavaScript: true,
      SVG: false,
    }),
  ],
  markdown: {
    remarkPlugins: [
      remarkDirective,
      remarkMath,
      [remarkPostLinks, { postsDir: path.resolve('./src/content/posts'), base, defaultLocale }],
      remarkContainerDirectives,
      remarkLeafDirectives,
      remarkMark,
      remarkListSpacing,
      remarkReadingTime,
    ],
    rehypePlugins: [
      rehypeSlug,
      rehypeHeadingAnchor,
      rehypeHeadingNumbering,
      rehypeKatex,
      [rehypeMermaid, { strategy: 'pre-mermaid' }],
      rehypeImageProcessor,
      rehypeInlineFootnotes,
      rehypeExternalLinks,
      rehypeCodeCopyButton,
    ],
    syntaxHighlight: {
      type: 'shiki',
      excludeLangs: ['mermaid'],
    },
    shikiConfig: {
      // Available themes: https://shiki.style/themes
      themes: {
        light: 'github-light',
        dark: 'github-dark',
      },
    },
  },
  vite: {
    plugins: [
      postAssetsPlugin(),
      {
        name: 'prefix-font-urls-with-base',
        transform(code, id) {
          if (!id.split('?')[0].endsWith('src/styles/font.css')) {
            return null
          }

          return code.replace(/url\(\s*(['"]?)\/fonts\//g, `url($1${base}/fonts/`)
        },
      },
    ],
  },
  devToolbar: {
    enabled: false,
  },
})
