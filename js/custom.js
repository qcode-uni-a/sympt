// Ensure MathJax is processed after navigation or content change
function typesetMath() {
    if (typeof MathJax !== "undefined") {
        MathJax.Hub.Queue(["Typeset", MathJax.Hub]);
    }
}

// Trigger MathJax on initial page load
window.onload = function() {
    typesetMath();
};

// Trigger MathJax on internal page navigation or AJAX-style content change
document.addEventListener("DOMContentLoaded", function() {
    typesetMath();
});

// For dynamically updated content (AJAX-style navigation), trigger MathJax
document.addEventListener('pjax:end', function() {
    typesetMath();
});
