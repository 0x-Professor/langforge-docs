const Fuse = require('fuse.js');
const fs = require('fs');
const path = require('path');

/**
 * Professional Search Service for LangForge Documentation
 * Provides fast, fuzzy search capabilities with relevance scoring
 */
class SearchService {
  constructor() {
    this.searchIndex = null;
    this.documents = [];
    this.isInitialized = false;
    this.indexPath = path.join(__dirname, '..', 'data', 'search-index.json');
    
    // Fuse.js configuration for optimal search experience
    this.fuseOptions = {
      keys: [
        { name: 'title', weight: 0.4 },
        { name: 'content', weight: 0.3 },
        { name: 'tags', weight: 0.2 },
        { name: 'category', weight: 0.1 }
      ],
      threshold: 0.3, // Lower = more strict matching
      distance: 100,
      includeScore: true,
      includeMatches: true,
      minMatchCharLength: 2,
      shouldSort: true,
      findAllMatches: true
    };
  }

  /**
   * Initialize the search service by building the search index
   */
  async initialize() {
    try {
      // Try to load existing index first
      if (await this.loadIndex()) {
        console.log('✅ Search index loaded from cache');
        return;
      }

      // Build new index if cache doesn't exist
      console.log('🔍 Building search index...');
      await this.buildIndex();
      await this.saveIndex();
      console.log('✅ Search index built and cached');
    } catch (error) {
      console.error('❌ Failed to initialize search service:', error);
      throw error;
    }
  }

  /**
   * Build search index from documentation files
   */
  async buildIndex() {
    const docsPath = path.join(__dirname, '..', 'docs');
    const documents = [];

    // Recursively process all markdown files
    const files = await this.getAllMarkdownFiles(docsPath);
    
    for (const filePath of files) {
      try {
        const content = fs.readFileSync(filePath, 'utf8');
        const document = this.parseDocument(filePath, content);
        if (document) {
          documents.push(document);
        }
      } catch (error) {
        console.warn(`⚠️  Failed to process file ${filePath}:`, error.message);
      }
    }

    this.documents = documents;
    this.searchIndex = new Fuse(documents, this.fuseOptions);
    this.isInitialized = true;
  }

  /**
   * Parse a markdown document into searchable format
   */
  parseDocument(filePath, content) {
    const relativePath = path.relative(path.join(__dirname, '..', 'docs'), filePath);
    const urlPath = `/docs/${relativePath.replace(/\.md$/, '').replace(/\\/g, '/')}`;
    
    // Extract metadata
    const title = this.extractTitle(content);
    const description = this.extractDescription(content);
    const tags = this.extractTags(content);
    const category = this.extractCategory(relativePath);
    
    // Clean content for searching
    const cleanContent = this.cleanContent(content);
    
    return {
      id: relativePath,
      title,
      description,
      content: cleanContent,
      tags,
      category,
      path: urlPath,
      filePath: relativePath,
      lastModified: fs.statSync(filePath).mtime
    };
  }

  /**
   * Perform search query
   */
  search(query, options = {}) {
    if (!this.isInitialized || !this.searchIndex) {
      throw new Error('Search service not initialized');
    }

    const {
      limit = 10,
      category = null,
      includeContent = false,
      minScore = 0.5
    } = options;

    let results = this.searchIndex.search(query);

    // Filter by category if specified
    if (category) {
      results = results.filter(result => 
        result.item.category.toLowerCase().includes(category.toLowerCase())
      );
    }

    // Filter by minimum score
    results = results.filter(result => result.score <= minScore);

    // Limit results
    results = results.slice(0, limit);

    // Format results
    return results.map(result => ({
      title: result.item.title,
      description: result.item.description,
      path: result.item.path,
      category: result.item.category,
      tags: result.item.tags,
      score: result.score,
      matches: result.matches,
      content: includeContent ? result.item.content.substring(0, 300) + '...' : undefined,
      lastModified: result.item.lastModified
    }));
  }

  /**
   * Get search suggestions based on partial query
   */
  getSuggestions(partialQuery, limit = 5) {
    if (!this.isInitialized || partialQuery.length < 2) {
      return [];
    }

    // Search for titles and extract unique suggestions
    const titleResults = this.searchIndex.search(partialQuery, { 
      ...this.fuseOptions,
      keys: ['title'],
      threshold: 0.4
    });

    const suggestions = titleResults
      .slice(0, limit)
      .map(result => result.item.title)
      .filter((title, index, array) => array.indexOf(title) === index);

    return suggestions;
  }

  /**
   * Get popular search terms (would integrate with analytics)
   */
  getPopularSearches() {
    // This would typically come from analytics data
    return [
      'langchain basics',
      'agents',
      'memory',
      'chains',
      'getting started',
      'examples',
      'api reference',
      'troubleshooting'
    ];
  }

  /**
   * Utility methods
   */
  async getAllMarkdownFiles(dir) {
    const files = [];
    const items = await fs.promises.readdir(dir);
    
    for (const item of items) {
      const fullPath = path.join(dir, item);
      const stat = await fs.promises.stat(fullPath);
      
      if (stat.isDirectory()) {
        files.push(...await this.getAllMarkdownFiles(fullPath));
      } else if (item.endsWith('.md')) {
        files.push(fullPath);
      }
    }
    
    return files;
  }

  extractTitle(content) {
    const match = content.match(/^#\s+(.+)$/m);
    return match ? match[1].trim() : 'Untitled';
  }

  extractDescription(content) {
    // Look for description in frontmatter or first paragraph
    const frontmatterMatch = content.match(/^---\s*\n([\s\S]*?)\n---/);
    if (frontmatterMatch) {
      const descMatch = frontmatterMatch[1].match(/description:\s*(.+)/);
      if (descMatch) return descMatch[1].trim().replace(/['"]/g, '');
    }
    
    // Fallback to first paragraph
    const lines = content.split('\n');
    for (const line of lines) {
      const cleaned = line.trim();
      if (cleaned && !cleaned.startsWith('#') && !cleaned.startsWith('```')) {
        return cleaned.substring(0, 150);
      }
    }
    
    return '';
  }

  extractTags(content) {
    const frontmatterMatch = content.match(/^---\s*\n([\s\S]*?)\n---/);
    if (frontmatterMatch) {
      const tagsMatch = frontmatterMatch[1].match(/tags:\s*\[(.*?)\]/);
      if (tagsMatch) {
        return tagsMatch[1].split(',').map(tag => tag.trim().replace(/['"]/g, ''));
      }
    }
    return [];
  }

  extractCategory(filePath) {
    const parts = filePath.split(path.sep);
    if (parts.length > 1) {
      return parts[0].replace(/-/g, ' ').replace(/\b\w/g, l => l.toUpperCase());
    }
    return 'General';
  }

  cleanContent(content) {
    return content
      .replace(/^---\s*\n[\s\S]*?\n---\s*\n/, '') // Remove frontmatter
      .replace(/```[\s\S]*?```/g, '') // Remove code blocks
      .replace(/`[^`]+`/g, '') // Remove inline code
      .replace(/\[([^\]]+)\]\([^)]+\)/g, '$1') // Extract link text
      .replace(/[#*_~`]/g, '') // Remove markdown syntax
      .replace(/\s+/g, ' ') // Normalize whitespace
      .trim();
  }

  async saveIndex() {
    try {
      const dataDir = path.dirname(this.indexPath);
      await fs.promises.mkdir(dataDir, { recursive: true });
      
      const indexData = {
        documents: this.documents,
        timestamp: new Date().toISOString(),
        version: '1.0.0'
      };
      
      await fs.promises.writeFile(this.indexPath, JSON.stringify(indexData, null, 2));
    } catch (error) {
      console.warn('⚠️  Failed to save search index:', error.message);
    }
  }

  async loadIndex() {
    try {
      if (!fs.existsSync(this.indexPath)) {
        return false;
      }

      const indexData = JSON.parse(await fs.promises.readFile(this.indexPath, 'utf8'));
      this.documents = indexData.documents;
      this.searchIndex = new Fuse(this.documents, this.fuseOptions);
      this.isInitialized = true;
      
      return true;
    } catch (error) {
      console.warn('⚠️  Failed to load search index:', error.message);
      return false;
    }
  }

  /**
   * Rebuild index (useful for updates)
   */
  async rebuildIndex() {
    console.log('🔄 Rebuilding search index...');
    this.isInitialized = false;
    await this.buildIndex();
    await this.saveIndex();
    console.log('✅ Search index rebuilt');
  }
}

module.exports = SearchService;