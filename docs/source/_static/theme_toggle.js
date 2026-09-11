// Swaps light/dark artwork when the theme changes: the light versions are
// unreadable on a dark background and vice versa. The pages emit both images
// of each pair, tagged with the classes below, and hide the dark ones; this
// picks the right one of every pair from there on.
document.addEventListener("DOMContentLoaded", function () {
    const btn = document.getElementById("wagtail-theme");

    const light = document.querySelectorAll(".theme-light-only");
    const dark = document.querySelectorAll(".theme-dark-only");

    // conf.py loads this through html_js_files, which applies to every page,
    // but the paired images only exist on some of them. Nothing to do where
    // the theme button or the images are absent.
    if (!btn || (!light.length && !dark.length)) {
        return;
    }

    function update() {
        // sphinx_wagtail_theme's blocking.js does
        // `document.body.classList.toggle("theme-dark", ...)`. documentElement
        // is checked as well so this keeps working if the theme moves it.
        const isDark =
            document.documentElement.classList.contains("theme-dark") ||
            document.body.classList.contains("theme-dark");

        light.forEach((el) => {
            el.style.display = isDark ? "none" : "block";
        });
        dark.forEach((el) => {
            el.style.display = isDark ? "block" : "none";
        });
    }

    // The button does not flip the class itself -- it dispatches
    // theme:toggle-theme-mode, and blocking.js sets the class off that event.
    // Reading it back on the next tick avoids racing that handler.
    btn.addEventListener("click", () => {
        setTimeout(update, 50);
    });

    update();
});
