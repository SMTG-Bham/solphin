document.addEventListener("DOMContentLoaded", function () {
    const btn = document.getElementById("wagtail-theme");

    const light = document.querySelector(".diagram-light");
    const dark = document.querySelector(".diagram-dark");

    function update() {
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

    btn.addEventListener("click", () => {
        setTimeout(update, 50);
    });

    update();
});