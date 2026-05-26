# Categories

Edit the JSON block below to control the category list and parent-child relationships.
Posts can opt into a category with `category: category-slug` in their frontmatter.

```json
[
  {
    "name": "AI Infra",
    "slug": "ai-infra",
    "children": [
      {
        "name": "LLM",
        "slug": "llm"
      },
      {
        "name": "LLM Infra",
        "slug": "llm-infra"
      },
      {
        "name": "vLLM",
        "slug": "vLLM"
      },
      {
        "name": "CUDA",
        "slug": "cuda"
      },
      {
        "name": "Parallelism",
        "slug": "parallelism"
      }
    ]
  },
  {
    "name": "CS Basics",
    "slug": "cs-basics"
  },
  {
    "name": "Algorithms",
    "slug": "algos",
    "children": [
      {
        "name": "DSA",
        "slug": "DSA"
      },
      {
        "name": "Reinforcement Learning",
        "slug": "rl"
      }
    ]
  },
  {
    "name": "Paper Reading",
    "slug": "paper-reading"
  },
  {
    "name": "Misc",
    "slug": "misc"
  }
]
```
