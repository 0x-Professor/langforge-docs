// LangForge Documentation Assets Entry Point
console.log('LangForge Documentation loaded');

// Initialize search functionality
document.addEventListener('DOMContentLoaded', function() {
  // Add search functionality if search input exists
  const searchInput = document.querySelector('#search-input');
  if (searchInput) {
    let searchTimeout;
    
    searchInput.addEventListener('input', function(e) {
      clearTimeout(searchTimeout);
      searchTimeout = setTimeout(() => {
        performSearch(e.target.value);
      }, 300);
    });
  }
  
  // Add feedback functionality
  initializeFeedback();
  
  // Track page views
  trackPageView();
});

async function performSearch(query) {
  if (!query || query.length < 2) return;
  
  try {
    const response = await fetch(`/api/search?q=${encodeURIComponent(query)}`);
    const data = await response.json();
    displaySearchResults(data.results);
  } catch (error) {
    console.error('Search failed:', error);
  }
}

function displaySearchResults(results) {
  const resultsContainer = document.querySelector('#search-results');
  if (!resultsContainer) return;
  
  if (results.length === 0) {
    resultsContainer.innerHTML = '<p>No results found</p>';
    return;
  }
  
  const html = results.map(result => `
    <div class="search-result">
      <h3><a href="${result.path}">${result.title}</a></h3>
      <p>${result.description}</p>
      <span class="category">${result.category}</span>
    </div>
  `).join('');
  
  resultsContainer.innerHTML = html;
}

function initializeFeedback() {
  const feedbackForm = document.querySelector('#feedback-form');
  if (!feedbackForm) return;
  
  feedbackForm.addEventListener('submit', async function(e) {
    e.preventDefault();
    
    const formData = new FormData(feedbackForm);
    const data = Object.fromEntries(formData.entries());
    
    try {
      const response = await fetch('/api/feedback', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(data)
      });
      
      if (response.ok) {
        alert('Thank you for your feedback!');
        feedbackForm.reset();
      }
    } catch (error) {
      console.error('Feedback submission failed:', error);
    }
  });
}

function trackPageView() {
  fetch('/api/analytics', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      event: 'page_view',
      page: window.location.pathname,
      data: {
        title: document.title,
        referrer: document.referrer
      }
    })
  }).catch(console.error);
}

// Export for module systems
if (typeof module !== 'undefined' && module.exports) {
  module.exports = { performSearch, trackPageView };
}
