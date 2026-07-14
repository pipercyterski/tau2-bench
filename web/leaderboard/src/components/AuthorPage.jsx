import { useEffect } from 'react'
import './Blog.css'
import { AUTHORS, PAPERS, postsByAuthor, authorInitials } from '../data/blogData'
import { AuthorChips } from './Blog'

const resolveHref = (href) =>
  href.startsWith('http') ? href : `${import.meta.env.BASE_URL}${href}`

function AuthorPage({ slug }) {
  const author = AUTHORS[slug]

  useEffect(() => {
    window.scrollTo(0, 0)
  }, [slug])

  if (!author) {
    return (
      <div className="blog-page">
        <header className="blog-page-header">
          <h1 className="blog-page-title">Author not found</h1>
          <p className="blog-page-subtitle">
            No author profile exists at this address. <a href="#blog">Back to the blog</a>.
          </p>
        </header>
      </div>
    )
  }

  const posts = postsByAuthor(slug)
  const papers = author.paperKeys.map((key) => PAPERS[key])

  return (
    <div className="blog-page author-page">
      <a href="#blog" className="author-back-link">← All posts</a>
      <header className="author-header">
        <div className="author-avatar author-avatar-large">{authorInitials(author.name)}</div>
        <div className="author-header-text">
          <h1 className="author-name">{author.name}</h1>
          {author.affiliation && <p className="author-affiliation">{author.affiliation}</p>}
        </div>
      </header>
      <p className="author-bio">{author.bio}</p>

      {papers.length > 0 && (
        <section className="author-section">
          <h2 className="author-section-title">Papers</h2>
          <ul className="author-paper-list">
            {papers.map((paper) => (
              <li key={paper.href}>
                <a href={paper.href} target="_blank" rel="noopener noreferrer" className="author-paper-link">
                  {paper.title}
                </a>
                <span className="author-paper-venue">{paper.venue}</span>
              </li>
            ))}
          </ul>
        </section>
      )}

      {posts.length > 0 && (
        <section className="author-section">
          <h2 className="author-section-title">Posts</h2>
          <div className="blog-grid">
            {posts.map((post) => {
              const external = post.href.startsWith('http')
              return (
                <article key={post.slug} className="blog-card">
                  <a
                    className="blog-card-link"
                    href={resolveHref(post.href)}
                    {...(external ? { target: '_blank', rel: 'noopener noreferrer' } : {})}
                  >
                    <div className="blog-card-top">
                      <span className={`blog-card-badge badge-${post.badge.toLowerCase()}`}>{post.badge}</span>
                      <span className="blog-card-date">{post.date}</span>
                    </div>
                    <h3 className="blog-card-title">
                      {post.title}
                      {external && <span className="external-marker" title="Opens on sierra.ai">↗</span>}
                    </h3>
                    <p className="blog-card-description">{post.description}</p>
                  </a>
                  <AuthorChips slugs={post.authorSlugs} />
                </article>
              )
            })}
          </div>
        </section>
      )}
    </div>
  )
}

export default AuthorPage
