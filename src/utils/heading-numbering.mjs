/**
 * @typedef {{ depth: number }} HeadingLike
 */

/**
 * Attach hierarchical section numbers to headings.
 *
 * The numbering base follows the shallowest heading depth in the document,
 * so articles that start at `##` still render as `1`, `1.1`, `1.1.1`.
 *
 * @template {HeadingLike} T
 * @param {T[]} headings
 * @param {{ minDepth?: number, maxDepth?: number }} [options]
 * @returns {(T & { number?: string })[]}
 */
export function attachHeadingNumbers(headings, options = {}) {
  const { minDepth = 1, maxDepth = 6 } = options
  const relevantHeadings = headings.filter(
    heading => heading.depth >= minDepth && heading.depth <= maxDepth,
  )

  if (relevantHeadings.length === 0) {
    return headings.map(heading => ({ ...heading }))
  }

  const baseDepth = Math.max(
    minDepth,
    Math.min(...relevantHeadings.map(heading => heading.depth)),
  )
  const counters = Array.from({ length: maxDepth - baseDepth + 1 }, () => 0)

  return headings.map((heading) => {
    if (heading.depth < baseDepth || heading.depth > maxDepth) {
      return { ...heading }
    }

    const depthIndex = heading.depth - baseDepth
    counters[depthIndex] += 1

    for (let index = depthIndex + 1; index < counters.length; index += 1) {
      counters[index] = 0
    }

    const number = counters
      .slice(0, depthIndex + 1)
      .filter(counter => counter > 0)
      .join('.')

    return {
      ...heading,
      number,
    }
  })
}
