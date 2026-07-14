import { useEffect } from 'react'
import './Blog.css'
import { BLOG_POSTS, AUTHORS, authorInitials } from '../data/blogData'

const resolveHref = (href) =>
  href.startsWith('http') ? href : `${import.meta.env.BASE_URL}${href}`

export function AuthorChips({ slugs }) {
  return (
    <div className="author-chips">
      {slugs.map((slug) => {
        const author = AUTHORS[slug]
        if (!author) return null
        return (
          <a key={slug} href={`#author/${slug}`} className="author-chip" onClick={(e) => e.stopPropagation()}>
            <span className="author-avatar">{authorInitials(author.name)}</span>
            <span className="author-chip-name">{author.name}</span>
          </a>
        )
      })}
    </div>
  )
}

function BlogCard({ post }) {
  const href = resolveHref(post.href)
  const external = post.href.startsWith('http')
  return (
    <article className="blog-card">
      <a
        className="blog-card-link"
        href={href}
        {...(external ? { target: '_blank', rel: 'noopener noreferrer' } : {})}
      >
        <div className="blog-card-top">
          <span className={`blog-card-badge badge-${post.badge.toLowerCase()}`}>{post.badge}</span>
          <span className="blog-card-date">{post.date}</span>
        </div>
        <h2 className="blog-card-title">
          {post.title}
          {external && <span className="external-marker" title="Opens on sierra.ai">↗</span>}
        </h2>
        <p className="blog-card-description">{post.description}</p>
      </a>
      <AuthorChips slugs={post.authorSlugs} />
    </article>
  )
}

function Blog() {
  useEffect(() => {
    window.scrollTo(0, 0)
  }, [])

  return (
    <div className="blog-page">
      <header className="blog-page-header">
        <h1 className="blog-page-title">Blog</h1>
        <p className="blog-page-subtitle">
          Research updates, benchmark releases, and engineering notes from the τ-bench team.
        </p>
      </header>
      <div className="blog-grid">
        {BLOG_POSTS.map((post) => (
          <BlogCard key={post.slug} post={post} />
        ))}
      </div>
    </div>
  )
}

export default Blog
