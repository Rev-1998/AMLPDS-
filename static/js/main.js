// static/js/main.js
/**
 * Main JavaScript file for AMLPDS
 */

// API Base URL
const API_BASE_URL = '/api';

// Authentication utilities
const auth = {
    getToken: () => localStorage.getItem('access_token'),
    setToken: (token) => localStorage.setItem('access_token', token),
    removeToken: () => localStorage.removeItem('access_token'),
    
    getRefreshToken: () => localStorage.getItem('refresh_token'),
    setRefreshToken: (token) => localStorage.setItem('refresh_token', token),
    removeRefreshToken: () => localStorage.removeItem('refresh_token'),
    
    isAuthenticated: () => !!auth.getToken(),
    
    getHeaders: () => ({
        'Authorization': `Bearer ${auth.getToken()}`,
        'Content-Type': 'application/json'
    }),
    
    logout: () => {
        auth.removeToken();
        auth.removeRefreshToken();
        window.location.href = '/';
    }
};

// API utilities
const api = {
    async request(endpoint, options = {}) {
        const url = `${API_BASE_URL}${endpoint}`;
        const config = {
            ...options,
            headers: {
                ...auth.getHeaders(),
                ...options.headers
            }
        };
        
        try {
            const response = await fetch(url, config);
            
            if (response.status === 401) {
                // Try to refresh token
                const refreshed = await this.refreshToken();
                if (refreshed) {
                    // Retry original request
                    config.headers.Authorization = `Bearer ${auth.getToken()}`;
                    return fetch(url, config);
                } else {
                    auth.logout();
                    throw new Error('Authentication failed');
                }
            }
            
            return response;
        } catch (error) {
            console.error('API request failed:', error);
            throw error;
        }
    },
    
    async refreshToken() {
        try {
            const response = await fetch('/auth/refresh', {
                method: 'POST',
                headers: {
                    'Authorization': `Bearer ${auth.getRefreshToken()}`,
                    'Content-Type': 'application/json'
                }
            });
            
            if (response.ok) {
                const data = await response.json();
                auth.setToken(data.access_token);
                return true;
            }
            
            return false;
        } catch (error) {
            console.error('Token refresh failed:', error);
            return false;
        }
    },
    
    async predict(emailText, urls = []) {
        const response = await this.request('/predict', {
            method: 'POST',
            body: JSON.stringify({ email_text: emailText, urls })
        });
        
        if (!response.ok) {
            throw new Error('Prediction failed');
        }
        
        return response.json();
    },
    
    async predictBatch(emails) {
        const response = await this.request('/predict/batch', {
            method: 'POST',
            body: JSON.stringify({ emails })
        });
        
        if (!response.ok) {
            throw new Error('Batch prediction failed');
        }
        
        return response.json();
    },
    
    async analyzeFile(file) {
        const formData = new FormData();
        formData.append('file', file);
        
        const response = await this.request('/analyze/file', {
            method: 'POST',
            headers: {
                'Authorization': `Bearer ${auth.getToken()}`
                // Don't set Content-Type, let browser set it for FormData
            },
            body: formData
        });
        
        if (!response.ok) {
            throw new Error('File analysis failed');
        }
        
        return response.json();
    },
    
    async getStats() {
        const response = await this.request('/stats');
        
        if (!response.ok) {
            throw new Error('Failed to get stats');
        }
        
        return response.json();
    },
    
    async reportPhishing(emailText, reportedAs, notes) {
        const response = await this.request('/report', {
            method: 'POST',
            body: JSON.stringify({
                email_text: emailText,
                reported_as: reportedAs,
                notes
            })
        });
        
        if (!response.ok) {
            throw new Error('Report submission failed');
        }
        
        return response.json();
    }
};

// UI utilities
const ui = {
    showLoading(elementId) {
        const element = document.getElementById(elementId);
        if (element) {
            element.innerHTML = '<div class="text-center"><div class="spinner"></div><p>Loading...</p></div>';
        }
    },
    
    showError(elementId, message) {
        const element = document.getElementById(elementId);
        if (element) {
            element.innerHTML = `<div class="alert alert-danger">${message}</div>`;
        }
    },
    
    showSuccess(elementId, message) {
        const element = document.getElementById(elementId);
        if (element) {
            element.innerHTML = `<div class="alert alert-success">${message}</div>`;
        }
    },
    
    formatConfidence(confidence) {
        return `${(confidence * 100).toFixed(1)}%`;
    },
    
    getRiskColorClass(riskLevel) {
        const colorMap = {
            'CRITICAL': 'risk-critical',
            'HIGH': 'risk-high',
            'MEDIUM': 'risk-medium',
            'LOW': 'risk-low',
            'UNCERTAIN': 'risk-medium'
        };
        return colorMap[riskLevel] || 'risk-medium';
    },
    
    displayPredictionResult(containerId, data) {
        const container = document.getElementById(containerId);
        if (!container) return;
        
        const isPhishing = data.prediction === 'phishing';
        const riskColorClass = this.getRiskColorClass(data.risk_level);
        
        let html = `
            <div class="result-card ${isPhishing ? 'result-phishing' : 'result-legitimate'} fade-in">
                <h3 class="${isPhishing ? 'text-danger' : 'text-success'}">
                    ${isPhishing ? '⚠️ PHISHING DETECTED' : '✓ LEGITIMATE EMAIL'}
                </h3>
                <div class="mt-3">
                    <p><strong>Confidence:</strong> ${this.formatConfidence(data.confidence)}</p>
                    <p><strong>Risk Level:</strong> <span class="${riskColorClass}">${data.risk_level}</span></p>
                </div>
            </div>
            
            <div class="card mt-4 fade-in">
                <h4>Summary</h4>
                <p>${data.explanation.summary}</p>
            </div>
            
            <div class="card mt-4 fade-in">
                <h4>Recommendations</h4>
                <ul>
                    ${data.explanation.recommendations.map(rec => `<li>${rec}</li>`).join('')}
                </ul>
            </div>
        `;
        
        if (data.explanation.suspicious_keywords && data.explanation.suspicious_keywords.length > 0) {
            html += `
                <div class="card mt-4 fade-in">
                    <h4>Suspicious Keywords Found</h4>
                    <div>
                        ${data.explanation.suspicious_keywords.map(kw => 
                            `<span class="keyword-tag">${kw.keyword}</span>`
                        ).join('')}
                    </div>
                </div>
            `;
        }
        
        container.innerHTML = html;
    }
};

// Form handlers
function setupFormHandlers() {
    // Login form
    const loginForm = document.getElementById('login-form');
    if (loginForm) {
        loginForm.addEventListener('submit', async (e) => {
            e.preventDefault();
            
            const username = document.getElementById('username').value;
            const password = document.getElementById('password').value;
            
            try {
                const response = await fetch('/auth/login', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ username, password })
                });
                
                const data = await response.json();
                
                if (response.ok) {
                    auth.setToken(data.access_token);
                    auth.setRefreshToken(data.refresh_token);
                    window.location.href = '/dashboard';
                } else {
                    ui.showError('error-message', data.message || 'Login failed');
                }
            } catch (error) {
                ui.showError('error-message', 'Network error. Please try again.');
            }
        });
    }
    
    // Register form
    const registerForm = document.getElementById('register-form');
    if (registerForm) {
        registerForm.addEventListener('submit', async (e) => {
            e.preventDefault();
            
            const username = document.getElementById('username').value;
            const email = document.getElementById('email').value;
            const password = document.getElementById('password').value;
            const confirmPassword = document.getElementById('confirm-password').value;
            
            if (password !== confirmPassword) {
                ui.showError('error-message', 'Passwords do not match');
                return;
            }
            
            try {
                const response = await fetch('/auth/register', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ username, email, password })
                });
                
                const data = await response.json();
                
                if (response.ok) {
                    auth.setToken(data.access_token);
                    auth.setRefreshToken(data.refresh_token);
                    window.location.href = '/dashboard';
                } else {
                    ui.showError('error-message', data.message || 'Registration failed');
                }
            } catch (error) {
                ui.showError('error-message', 'Network error. Please try again.');
            }
        });
    }
    
    // Email analysis form
    const analyzeForm = document.getElementById('analyze-form');
    if (analyzeForm) {
        analyzeForm.addEventListener('submit', async (e) => {
            e.preventDefault();
            
            const emailText = document.getElementById('email-text').value;
            
            if (!auth.isAuthenticated()) {
                alert('Please login to analyze emails');
                window.location.href = '/auth/login';
                return;
            }
            
            ui.showLoading('results-content');
            document.getElementById('results-section').classList.remove('hidden');
            
            try {
                const data = await api.predict(emailText);
                ui.displayPredictionResult('results-content', data);
            } catch (error) {
                ui.showError('results-content', 'Analysis failed. Please try again.');
            }
        });
    }
    
    // File upload
    const fileInput = document.getElementById('email-file');
    if (fileInput) {
        fileInput.addEventListener('change', async (e) => {
            const file = e.target.files[0];
            if (!file) return;
            
            // Read file content for text files
            if (file.type === 'text/plain' || file.name.endsWith('.txt')) {
                const reader = new FileReader();
                reader.onload = (e) => {
                    document.getElementById('email-text').value = e.target.result;
                };
                reader.readAsText(file);
            } else {
                // For other file types, use file analysis API
                if (!auth.isAuthenticated()) {
                    alert('Please login to analyze files');
                    return;
                }
                
                ui.showLoading('results-content');
                document.getElementById('results-section').classList.remove('hidden');
                
                try {
                    const data = await api.analyzeFile(file);
                    ui.displayPredictionResult('results-content', data);
                } catch (error) {
                    ui.showError('results-content', 'File analysis failed. Please try again.');
                }
            }
        });
    }
}

// Dashboard functions
async function loadDashboardStats() {
    if (!auth.isAuthenticated()) return;
    
    try {
        const stats = await api.getStats();
        
        // Update stat cards
        document.getElementById('total-scans').textContent = stats.total_scans;
        document.getElementById('phishing-detected').textContent = stats.phishing_detected;
        document.getElementById('detection-rate').textContent = ui.formatConfidence(stats.detection_rate);
        document.getElementById('avg-confidence').textContent = ui.formatConfidence(stats.average_confidence);
    } catch (error) {
        console.error('Failed to load stats:', error);
    }
}

// Check authentication on page load
function checkAuth() {
    const protectedPaths = ['/dashboard', '/analyze'];
    const currentPath = window.location.pathname;
    
    if (protectedPaths.includes(currentPath) && !auth.isAuthenticated()) {
        window.location.href = '/auth/login';
    }
}

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    checkAuth();
    setupFormHandlers();
    
    // Load dashboard stats if on dashboard page
    if (window.location.pathname === '/dashboard') {
        loadDashboardStats();
    }
    
    // Update UI based on auth status
    if (auth.isAuthenticated()) {
        const userMenu = document.getElementById('user-menu');
        const loginButton = document.getElementById('login-button');
        
        if (userMenu) userMenu.classList.remove('hidden');
        if (loginButton) loginButton.classList.add('hidden');
        
        // Decode JWT to get username
        try {
            const token = auth.getToken();
            const payload = JSON.parse(atob(token.split('.')[1]));
            const usernameDisplay = document.getElementById('username-display');
            if (usernameDisplay) {
                usernameDisplay.textContent = payload.sub || 'User';
            }
        } catch (error) {
            console.error('Error decoding token:', error);
        }
    }
});

// Export for use in other scripts
window.AMLPDS = {
    auth,
    api,
    ui
};
