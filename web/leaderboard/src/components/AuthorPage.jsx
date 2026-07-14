import { useEffect } from 'react'
import './Blog.css'
import { AUTHORS, PAPERS, postsByAuthor, authorPhoto } from '../data/blogData'
import { BlogCard } from './Blog'

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
        <img src={authorPhoto(slug)} alt={author.name} className="author-photo" />
        <div className="author-header-text">
          <h1 className="author-name">{author.name}</h1>
          <p className="author-role">{author.role}</p>
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
            {posts.map((post) => (
              <BlogCard key={post.slug} post={post} />
            ))}
          </div>
        </section>
      )}
    </div>
  )
}

export default AuthorPage
