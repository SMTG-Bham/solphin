// Swaps the two front-page diagrams when the theme changes: the light artwork
// is unreadable on a dark background and vice versa. index.rst emits both
// images and hides the dark one; this picks the right one from there on.
document.addEventListener("DOMContentLoaded", function () {
    const btn = document.getElementById("wagtail-theme");

    const light = document.getElementById("diagram-light");
    const dark = document.getElementById("diagram-dark");

    // conf.py loads this through html_js_files, which applies to every page,
    // but the two diagrams only exist on the front page. Without this guard
    // update() dereferences null on every other page in the docs.
    if (!btn || !light || !dark) {
        return;
    }

    function update() {
        // sphinx_wagtail_theme's blocking.js does
        // `document.body.classList.toggle("theme-dark", ...)`. documentElement
        // is checked as well so this keeps working if the theme moves it.
        const isDark =
            document.documentElement.classList.contains("theme-dark") ||
            document.body.classList.contains("theme-dark");

        if (isDark) {
            light.style.display = "none";
            dark.style.display = "block";
        } else {
            light.style.display = "block";
            dark.style.display = "none";
        }
    }

    // The button does not flip the class itself -- it dispatches
    // theme:toggle-theme-mode, and blocking.js sets the class off that event.
    // Reading it back on the next tick avoids racing that handler.
    btn.addEventListener("click", () => {
        setTimeout(update, 50);
    });

    update();
});
