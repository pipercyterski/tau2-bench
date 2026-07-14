// Blog posts and author profiles for the Blog page and author bio pages.
// Post authorship follows the corresponding paper's author list (see the
// Citation section of the repo README).

export const PAPERS = {
  tauBench: {
    title: 'τ-bench: A Benchmark for Tool-Agent-User Interaction in Real-World Domains',
    href: 'https://arxiv.org/abs/2406.12045',
    venue: 'arXiv 2024',
    authorSlugs: ['shunyu-yao', 'noah-shinn', 'pedram-razavi', 'karthik-narasimhan'],
  },
  tau2Bench: {
    title: 'τ²-Bench: Evaluating Conversational Agents in a Dual-Control Environment',
    href: 'https://arxiv.org/abs/2506.07982',
    venue: 'arXiv 2025',
    authorSlugs: ['victor-barres', 'honghua-dong', 'soham-ray', 'xujie-si', 'karthik-narasimhan'],
  },
  tauKnowledge: {
    title: 'τ-Knowledge: Evaluating Conversational Agents over Unstructured Knowledge',
    href: 'https://arxiv.org/abs/2603.04370',
    venue: 'arXiv 2026',
    authorSlugs: ['quan-shi', 'alexandra-zytek', 'pedram-razavi', 'karthik-narasimhan', 'victor-barres'],
  },
  tauVoice: {
    title: 'τ-Voice: Benchmarking Full-Duplex Voice Agents on Real-World Domains',
    href: 'https://arxiv.org/abs/2603.13686',
    venue: 'arXiv 2026',
    authorSlugs: ['soham-ray', 'keshav-dhandhania', 'victor-barres', 'karthik-narasimhan'],
  },
  saber: {
    title: 'SABER: Small Actions, Big Errors — Safeguarding Mutating Steps in LLM Agents (τ-Bench Verified)',
    href: 'https://arxiv.org/abs/2512.07850',
    venue: 'ICLR 2026 Workshop',
    authorSlugs: ['alejandro-cuadron', 'pengfei-yu', 'yang-liu', 'arpit-gupta'],
  },
}

export const AUTHORS = {
  'victor-barres': {
    name: 'Victor Barres',
    affiliation: 'Sierra',
    bio: 'Victor is a researcher at Sierra working on agent benchmarking. He is a lead author of τ²-Bench and a co-author of τ-Knowledge and τ-Voice.',
    paperKeys: ['tau2Bench', 'tauKnowledge', 'tauVoice'],
  },
  'karthik-narasimhan': {
    name: 'Karthik Narasimhan',
    affiliation: 'Sierra',
    bio: 'Karthik leads research at Sierra and is a co-author of every paper in the τ-bench family, from the original τ-bench through τ²-Bench, τ-Knowledge, and τ-Voice.',
    paperKeys: ['tauBench', 'tau2Bench', 'tauKnowledge', 'tauVoice'],
  },
  'soham-ray': {
    name: 'Soham Ray',
    affiliation: 'Sierra',
    bio: 'Soham is a researcher at Sierra working on voice agents and agent evaluation. He is the lead author of τ-Voice and a co-author of τ²-Bench.',
    paperKeys: ['tauVoice', 'tau2Bench'],
  },
  'keshav-dhandhania': {
    name: 'Keshav Dhandhania',
    affiliation: 'Sierra',
    bio: 'Keshav works at Sierra on real-time voice agents and is a co-author of τ-Voice.',
    paperKeys: ['tauVoice'],
  },
  'pedram-razavi': {
    name: 'Pedram Razavi',
    affiliation: 'Sierra',
    bio: 'Pedram works at Sierra on agent benchmarking and is a co-author of the original τ-bench and τ-Knowledge.',
    paperKeys: ['tauBench', 'tauKnowledge'],
  },
  'quan-shi': {
    name: 'Quan Shi',
    bio: 'Quan is the lead author of τ-Knowledge, a benchmark for evaluating conversational agents over large unstructured knowledge bases.',
    paperKeys: ['tauKnowledge'],
  },
  'alexandra-zytek': {
    name: 'Alexandra Zytek',
    bio: 'Alexandra is a co-author of τ-Knowledge, a benchmark for evaluating conversational agents over large unstructured knowledge bases.',
    paperKeys: ['tauKnowledge'],
  },
  'shunyu-yao': {
    name: 'Shunyu Yao',
    bio: 'Shunyu is the lead author of the original τ-bench, the benchmark for tool-agent-user interaction that started the τ-bench family.',
    paperKeys: ['tauBench'],
  },
  'noah-shinn': {
    name: 'Noah Shinn',
    bio: 'Noah is a co-author of the original τ-bench, the benchmark for tool-agent-user interaction that started the τ-bench family.',
    paperKeys: ['tauBench'],
  },
  'honghua-dong': {
    name: 'Honghua Dong',
    bio: 'Honghua is a co-author of τ²-Bench, which evaluates conversational agents in dual-control environments.',
    paperKeys: ['tau2Bench'],
  },
  'xujie-si': {
    name: 'Xujie Si',
    bio: 'Xujie is a co-author of τ²-Bench, which evaluates conversational agents in dual-control environments.',
    paperKeys: ['tau2Bench'],
  },
  'alejandro-cuadron': {
    name: 'Alejandro Cuadron',
    bio: 'Alejandro is the lead author of τ-Bench Verified (SABER), the systematic audit of τ-bench tasks that drove most of the τ³ airline and retail task fixes.',
    paperKeys: ['saber'],
  },
  'pengfei-yu': {
    name: 'Pengfei Yu',
    affiliation: 'Amazon',
    bio: 'Pengfei is a co-author of τ-Bench Verified (SABER), the systematic audit of τ-bench tasks that drove most of the τ³ airline and retail task fixes.',
    paperKeys: ['saber'],
  },
  'yang-liu': {
    name: 'Yang Liu',
    affiliation: 'Amazon',
    bio: 'Yang is a co-author of τ-Bench Verified (SABER), the systematic audit of τ-bench tasks that drove most of the τ³ airline and retail task fixes.',
    paperKeys: ['saber'],
  },
  'arpit-gupta': {
    name: 'Arpit Gupta',
    affiliation: 'Amazon',
    bio: 'Arpit is a co-author of τ-Bench Verified (SABER), the systematic audit of τ-bench tasks that drove most of the τ³ airline and retail task fixes.',
    paperKeys: ['saber'],
  },
}

// Newest first. `href` values starting with http are external; everything else
// is resolved against import.meta.env.BASE_URL.
export const BLOG_POSTS = [
  {
    slug: 'tau-knowledge',
    title: 'τ-knowledge',
    badge: 'Research',
    date: 'February 2026',
    description:
      'A benchmark for evaluating AI agents in knowledge-intensive customer support: a realistic fintech knowledge base of 698 documents paired with tasks requiring multi-step reasoning, policy application, and tool use. The best frontier model reaches only ~26% pass^1.',
    href: 'blog/tau-knowledge.html',
    authorSlugs: PAPERS.tauKnowledge.authorSlugs,
  },
  {
    slug: 'tau-voice-examples',
    title: 'τ-voice Examples',
    badge: 'Research',
    date: 'February 2026',
    description:
      'τ-Voice extends τ-bench to live, full-duplex voice interactions — overlapping speech, interruptions, accents, and background noise. These annotated examples show how the same task can succeed with clean audio and fail under realistic conditions.',
    href: 'blog/tau-voice-examples.html',
    authorSlugs: PAPERS.tauVoice.authorSlugs,
  },
  {
    slug: 'tau3-task-fixes',
    title: 'τ³-Bench: Fixing Airline + Retail',
    badge: 'Engineering',
    date: 'February 2026',
    description:
      'We audited and fixed 50+ tasks across the airline and retail domains, addressing incorrect expected actions, ambiguous instructions, impossible constraints, and missing fallback behaviors — most sourced from τ-Bench Verified (SABER) and community pull requests.',
    href: 'blog/tau3-task-fixes.html',
    authorSlugs: PAPERS.saber.authorSlugs,
  },
  {
    slug: 'tau3-bench-announcement',
    title: 'τ³-bench: Advancing Agent Benchmarking to Knowledge and Voice',
    badge: 'Announcement',
    date: 'March 2026',
    description:
      'The Sierra blog announcement of τ³-bench, which extends the benchmark family with the τ-Knowledge and τ-Voice tracks.',
    href: 'https://sierra.ai/blog/bench-advancing-agent-benchmarking-to-knowledge-and-voice',
    authorSlugs: ['quan-shi', 'alexandra-zytek', 'soham-ray', 'keshav-dhandhania', 'pedram-razavi', 'victor-barres', 'karthik-narasimhan'],
  },
  {
    slug: 'tau2-bench-announcement',
    title: 'τ²-bench: Benchmarking Agents in Collaborative Real-World Scenarios',
    badge: 'Announcement',
    date: 'June 2025',
    description:
      'The Sierra blog announcement of τ²-bench, which evaluates conversational agents in dual-control environments where both the agent and the user can act.',
    href: 'https://sierra.ai/blog/benchmarking-agents-in-collaborative-real-world-scenarios',
    authorSlugs: PAPERS.tau2Bench.authorSlugs,
  },
  {
    slug: 'tau-bench-announcement',
    title: 'τ-bench: Benchmarking AI Agents',
    badge: 'Announcement',
    date: 'June 2024',
    description:
      'The Sierra blog announcement of the original τ-bench, a benchmark for tool-agent-user interaction in real-world domains.',
    href: 'https://sierra.ai/blog/benchmarking-ai-agents',
    authorSlugs: PAPERS.tauBench.authorSlugs,
  },
]

export const postsByAuthor = (slug) => BLOG_POSTS.filter((p) => p.authorSlugs.includes(slug))

export const authorInitials = (name) =>
  name
    .split(' ')
    .map((part) => part[0])
    .join('')
    .toUpperCase()
