document.addEventListener('DOMContentLoaded', function() {
    const form = document.getElementById('storyForm');
    const generateBtn = document.getElementById('generateBtn');
    const loadingSpinner = document.getElementById('loadingSpinner');
    const btnText = document.getElementById('btnText');
    
    if (form && generateBtn) {
        form.addEventListener('submit', function(e) {
            // Show loading state
            generateBtn.disabled = true;
            loadingSpinner.classList.remove('d-none');
            btnText.textContent = 'Generating Story...';
            
            // Add progress bar
            showProgressBar();
            
            // Simulate progress (since we can't get real progress from backend)
            let progress = 0;
            const progressInterval = setInterval(() => {
                progress += Math.random() * 10;
                if (progress > 90) progress = 90;
                updateProgress(progress);
            }, 1000);
            
            // Store interval to clear later if needed
            form.progressInterval = progressInterval;
        });
    }
    
    // Form validation
    const textPrompt = document.getElementById('id_text_prompt');
    const audioFile = document.getElementById('id_audio_file');
    
    if (textPrompt && audioFile) {
        function validateForm() {
            const hasText = textPrompt.value.trim().length > 0;
            const hasAudio = audioFile.files.length > 0;
            
            if (generateBtn) {
                generateBtn.disabled = !(hasText || hasAudio);
            }
        }
        
        textPrompt.addEventListener('input', validateForm);
        audioFile.addEventListener('change', validateForm);
        
        // Initial validation
        validateForm();
    }
    
    // Image lazy loading
    const images = document.querySelectorAll('img');
    if ('IntersectionObserver' in window) {
        const imageObserver = new IntersectionObserver((entries, observer) => {
            entries.forEach(entry => {
                if (entry.isIntersecting) {
                    const img = entry.target;
                    if (img.dataset.src) {
                        img.src = img.dataset.src;
                        img.removeAttribute('data-src');
                        observer.unobserve(img);
                    }
                }
            });
        });
        
        images.forEach(img => {
            if (img.dataset.src) {
                imageObserver.observe(img);
            }
        });
    }
    
    // Smooth scroll for anchor links
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function (e) {
            e.preventDefault();
            const target = document.querySelector(this.getAttribute('href'));
            if (target) {
                target.scrollIntoView({
                    behavior: 'smooth',
                    block: 'start'
                });
            }
        });
    });
    
    // Auto-resize textareas
    const textareas = document.querySelectorAll('textarea');
    textareas.forEach(textarea => {
        textarea.addEventListener('input', function() {
            this.style.height = 'auto';
            this.style.height = (this.scrollHeight) + 'px';
        });
    });
});

function showProgressBar() {
    const progressContainer = document.createElement('div');
    progressContainer.className = 'progress-container';
    progressContainer.innerHTML = '<div class="progress-bar-custom" id="progressBar"></div>';
    document.body.appendChild(progressContainer);
}

function updateProgress(percentage) {
    const progressBar = document.getElementById('progressBar');
    if (progressBar) {
        progressBar.style.width = percentage + '%';
    }
}

function hideProgressBar() {
    const progressContainer = document.querySelector('.progress-container');
    if (progressContainer) {
        progressContainer.remove();
    }
}

// Utility functions
function copyToClipboard(text) {
    navigator.clipboard.writeText(text).then(() => {
        showToast('Copied to clipboard!', 'success');
    }).catch(() => {
        showToast('Failed to copy', 'error');
    });
}

function showToast(message, type = 'info') {
    const toast = document.createElement('div');
    toast.className = `toast-notification toast-${type}`;
    toast.textContent = message;
    
    const style = {
        position: 'fixed',
        top: '20px',
        right: '20px',
        padding: '12px 24px',
        borderRadius: '8px',
        color: 'white',
        fontWeight: '500',
        zIndex: '10000',
        opacity: '0',
        transition: 'opacity 0.3s'
    };
    
    const colors = {
        success: '#28a745',
        error: '#dc3545',
        warning: '#ffc107',
        info: '#17a2b8'
    };
    
    Object.assign(toast.style, style);
    toast.style.backgroundColor = colors[type] || colors.info;
    
    document.body.appendChild(toast);
    
    // Animate in
    setTimeout(() => toast.style.opacity = '1', 100);
    
    // Remove after delay
    setTimeout(() => {
        toast.style.opacity = '0';
        setTimeout(() => toast.remove(), 300);
    }, 3000);
}
