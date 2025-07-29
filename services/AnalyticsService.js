const fs = require('fs');
const path = require('path');

/**
 * Professional Analytics Service for LangForge Documentation
 * Provides comprehensive tracking, storage, and reporting capabilities
 */
class AnalyticsService {
  constructor() {
    this.dataDir = path.join(__dirname, '..', 'data', 'analytics');
    this.dailyFile = null;
    this.sessionStore = new Map();
    this.eventQueue = [];
    this.isProcessing = false;
    
    // Ensure data directory exists
    this.ensureDataDirectory();
    
    // Process events every 5 seconds
    setInterval(() => this.processEventQueue(), 5000);
    
    // Rotate logs daily
    this.rotateLogs();
    setInterval(() => this.rotateLogs(), 24 * 60 * 60 * 1000); // Daily
  }

  /**
   * Track page view
   */
  trackPageView(data) {
    const event = {
      type: 'page_view',
      timestamp: new Date().toISOString(),
      sessionId: data.sessionId,
      userId: data.userId || null,
      page: data.page,
      referrer: data.referrer,
      userAgent: data.userAgent,
      ip: data.ip,
      country: data.country || null,
      loadTime: data.loadTime || null
    };

    this.queueEvent(event);
  }

  /**
   * Track search query
   */
  trackSearch(data) {
    const event = {
      type: 'search',
      timestamp: new Date().toISOString(),
      sessionId: data.sessionId,
      userId: data.userId || null,
      query: data.query,
      results: data.results || 0,
      clickedResult: data.clickedResult || null,
      page: data.page
    };

    this.queueEvent(event);
  }

  /**
   * Track user feedback
   */
  trackFeedback(data) {
    const event = {
      type: 'feedback',
      timestamp: new Date().toISOString(),
      sessionId: data.sessionId,
      userId: data.userId || null,
      page: data.page,
      rating: data.rating,
      comment: data.comment,
      email: data.email || null,
      category: data.category || 'general'
    };

    this.queueEvent(event);
    
    // Also store in separate feedback file for easy access
    this.storeFeedback(event);
  }

  /**
   * Track custom events
   */
  trackEvent(eventType, data) {
    const event = {
      type: eventType,
      timestamp: new Date().toISOString(),
      sessionId: data.sessionId,
      userId: data.userId || null,
      ...data
    };

    this.queueEvent(event);
  }

  /**
   * Track user session
   */
  trackSession(sessionId, data) {
    const session = {
      id: sessionId,
      startTime: new Date().toISOString(),
      lastActivity: new Date().toISOString(),
      userAgent: data.userAgent,
      ip: data.ip,
      country: data.country || null,
      referrer: data.referrer,
      pages: [],
      searches: [],
      events: []
    };

    this.sessionStore.set(sessionId, session);
    
    // Clean up old sessions (older than 24 hours)
    this.cleanupSessions();
  }

  /**
   * Update session activity
   */
  updateSession(sessionId, page) {
    const session = this.sessionStore.get(sessionId);
    if (session) {
      session.lastActivity = new Date().toISOString();
      if (!session.pages.includes(page)) {
        session.pages.push(page);
      }
    }
  }

  /**
   * Get analytics dashboard data
   */
  async getDashboardData(days = 30) {
    try {
      const data = {
        summary: await this.getSummaryStats(days),
        topPages: await this.getTopPages(days),
        topSearches: await this.getTopSearches(days),
        recentFeedback: await this.getRecentFeedback(days),
        userFlow: await this.getUserFlow(days),
        performance: await this.getPerformanceMetrics(days)
      };

      return data;
    } catch (error) {
      console.error('Failed to generate dashboard data:', error);
      return null;
    }
  }

  /**
   * Get summary statistics
   */
  async getSummaryStats(days) {
    const files = await this.getRecentLogFiles(days);
    let totalViews = 0;
    let uniqueUsers = new Set();
    let totalSearches = 0;
    let avgSessionTime = 0;

    for (const file of files) {
      const events = await this.readLogFile(file);
      
      for (const event of events) {
        if (event.type === 'page_view') {
          totalViews++;
          if (event.sessionId) uniqueUsers.add(event.sessionId);
        } else if (event.type === 'search') {
          totalSearches++;
        }
      }
    }

    return {
      totalViews,
      uniqueUsers: uniqueUsers.size,
      totalSearches,
      avgSessionTime,
      period: `${days} days`
    };
  }

  /**
   * Get top pages
   */
  async getTopPages(days, limit = 10) {
    const files = await this.getRecentLogFiles(days);
    const pageViews = new Map();

    for (const file of files) {
      const events = await this.readLogFile(file);
      
      for (const event of events) {
        if (event.type === 'page_view') {
          const count = pageViews.get(event.page) || 0;
          pageViews.set(event.page, count + 1);
        }
      }
    }

    return Array.from(pageViews.entries())
      .sort((a, b) => b[1] - a[1])
      .slice(0, limit)
      .map(([page, views]) => ({ page, views }));
  }

  /**
   * Get top searches
   */
  async getTopSearches(days, limit = 10) {
    const files = await this.getRecentLogFiles(days);
    const searches = new Map();

    for (const file of files) {
      const events = await this.readLogFile(file);
      
      for (const event of events) {
        if (event.type === 'search') {
          const count = searches.get(event.query) || 0;
          searches.set(event.query, count + 1);
        }
      }
    }

    return Array.from(searches.entries())
      .sort((a, b) => b[1] - a[1])
      .slice(0, limit)
      .map(([query, count]) => ({ query, count }));
  }

  /**
   * Get recent feedback
   */
  async getRecentFeedback(days, limit = 20) {
    try {
      const feedbackFile = path.join(this.dataDir, 'feedback.jsonl');
      if (!fs.existsSync(feedbackFile)) return [];

      const content = await fs.promises.readFile(feedbackFile, 'utf8');
      const lines = content.trim().split('\n').filter(line => line);
      
      const cutoff = new Date(Date.now() - days * 24 * 60 * 60 * 1000);
      
      return lines
        .map(line => JSON.parse(line))
        .filter(feedback => new Date(feedback.timestamp) > cutoff)
        .sort((a, b) => new Date(b.timestamp) - new Date(a.timestamp))
        .slice(0, limit);
    } catch (error) {
      console.error('Failed to read feedback:', error);
      return [];
    }
  }

  /**
   * Export data for external analytics tools
   */
  async exportData(startDate, endDate, format = 'json') {
    try {
      const start = new Date(startDate);
      const end = new Date(endDate);
      const allEvents = [];

      // Collect events from the date range
      const diffTime = Math.abs(end - start);
      const diffDays = Math.ceil(diffTime / (1000 * 60 * 60 * 24));

      for (let i = 0; i <= diffDays; i++) {
        const date = new Date(start);
        date.setDate(date.getDate() + i);
        const dateStr = date.toISOString().split('T')[0];
        const filename = path.join(this.dataDir, `analytics-${dateStr}.jsonl`);

        if (fs.existsSync(filename)) {
          const events = await this.readLogFile(filename);
          allEvents.push(...events.filter(event => {
            const eventDate = new Date(event.timestamp);
            return eventDate >= start && eventDate <= end;
          }));
        }
      }

      if (format === 'csv') {
        // Convert to CSV format
        if (allEvents.length === 0) return 'timestamp,type,sessionId,page,query,rating\n';
        
        const csvHeader = Object.keys(allEvents[0]).join(',') + '\n';
        const csvRows = allEvents.map(event => 
          Object.values(event).map(value => 
            typeof value === 'string' && value.includes(',') ? `"${value}"` : value
          ).join(',')
        ).join('\n');
        
        return csvHeader + csvRows;
      }

      return {
        exportDate: new Date().toISOString(),
        dateRange: { start: startDate, end: endDate },
        totalEvents: allEvents.length,
        events: allEvents
      };
    } catch (error) {
      console.error('Failed to export data:', error);
      return null;
    }
  }

  /**
   * Private helper methods
   */
  queueEvent(event) {
    this.eventQueue.push(event);
  }

  async processEventQueue() {
    if (this.isProcessing || this.eventQueue.length === 0) return;

    this.isProcessing = true;
    const events = [...this.eventQueue];
    this.eventQueue = [];

    try {
      await this.storeEvents(events);
    } catch (error) {
      console.error('Failed to store events:', error);
      // Re-queue events on failure
      this.eventQueue.unshift(...events);
    } finally {
      this.isProcessing = false;
    }
  }

  async storeEvents(events) {
    const logFile = this.getCurrentLogFile();
    const lines = events.map(event => JSON.stringify(event)).join('\n') + '\n';
    
    await fs.promises.appendFile(logFile, lines);
  }

  async storeFeedback(feedback) {
    const feedbackFile = path.join(this.dataDir, 'feedback.jsonl');
    const line = JSON.stringify(feedback) + '\n';
    
    await fs.promises.appendFile(feedbackFile, line);
  }

  getCurrentLogFile() {
    const today = new Date().toISOString().split('T')[0];
    return path.join(this.dataDir, `analytics-${today}.jsonl`);
  }

  rotateLogs() {
    this.dailyFile = this.getCurrentLogFile();
  }

  ensureDataDirectory() {
    if (!fs.existsSync(this.dataDir)) {
      fs.mkdirSync(this.dataDir, { recursive: true });
    }
  }

  cleanupSessions() {
    const cutoff = Date.now() - 24 * 60 * 60 * 1000; // 24 hours
    
    for (const [sessionId, session] of this.sessionStore.entries()) {
      if (new Date(session.lastActivity).getTime() < cutoff) {
        this.sessionStore.delete(sessionId);
      }
    }
  }

  async getRecentLogFiles(days) {
    const files = [];
    
    for (let i = 0; i < days; i++) {
      const date = new Date(Date.now() - i * 24 * 60 * 60 * 1000);
      const dateStr = date.toISOString().split('T')[0];
      const filename = path.join(this.dataDir, `analytics-${dateStr}.jsonl`);
      
      if (fs.existsSync(filename)) {
        files.push(filename);
      }
    }
    
    return files;
  }

  async readLogFile(filename) {
    try {
      const content = await fs.promises.readFile(filename, 'utf8');
      return content
        .trim()
        .split('\n')
        .filter(line => line)
        .map(line => JSON.parse(line));
    } catch (error) {
      console.error(`Failed to read log file ${filename}:`, error);
      return [];
    }
  }

  async getUserFlow(days) {
    const files = await this.getRecentLogFiles(days);
    const entryPages = new Map();
    const exitPages = new Map();
    const pathSequences = new Map();

    for (const file of files) {
      const events = await this.readLogFile(file);
      const sessions = new Map();

      // Group events by session
      for (const event of events) {
        if (event.type === 'page_view' && event.sessionId) {
          if (!sessions.has(event.sessionId)) {
            sessions.set(event.sessionId, []);
          }
          sessions.get(event.sessionId).push(event);
        }
      }

      // Analyze each session
      for (const [sessionId, sessionEvents] of sessions) {
        if (sessionEvents.length === 0) continue;

        // Sort by timestamp
        sessionEvents.sort((a, b) => new Date(a.timestamp) - new Date(b.timestamp));

        // Track entry and exit pages
        const firstPage = sessionEvents[0].page;
        const lastPage = sessionEvents[sessionEvents.length - 1].page;

        entryPages.set(firstPage, (entryPages.get(firstPage) || 0) + 1);
        exitPages.set(lastPage, (exitPages.get(lastPage) || 0) + 1);

        // Track common paths for sessions with multiple pages
        if (sessionEvents.length > 1) {
          const pathKey = sessionEvents.map(event => event.page).join(' > ');
          pathSequences.set(pathKey, (pathSequences.get(pathKey) || 0) + 1);
        }
      }
    }

    const topEntryPages = Array.from(entryPages.entries())
      .sort((a, b) => b[1] - a[1])
      .map(([page, count]) => ({ page, count }))
      .slice(0, 10);

    const topExitPages = Array.from(exitPages.entries())
      .sort((a, b) => b[1] - a[1])
      .map(([page, count]) => ({ page, count }))
      .slice(0, 10);

    const topPaths = Array.from(pathSequences.entries())
      .sort((a, b) => b[1] - a[1])
      .map(([path, count]) => ({ path, count }))
      .slice(0, 10);

    return { topEntryPages, topExitPages, topPaths };
  }

  async getPerformanceMetrics(days) {
    const files = await this.getRecentLogFiles(days);
    let totalLoadTime = 0;
    let totalCount = 0;

    for (const file of files) {
      const events = await this.readLogFile(file);
      
      for (const event of events) {
        if (event.type === 'page_view' && event.loadTime) {
          totalLoadTime += event.loadTime;
          totalCount++;
        }
      }
    }

    const avgLoadTime = totalCount > 0 ? totalLoadTime / totalCount : 0;

    return {
      totalLoadTime,
      avgLoadTime,
      totalCount,
      period: `${days} days`
    };
  }
}

module.exports = AnalyticsService;