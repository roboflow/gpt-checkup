let dropdowns = document.querySelectorAll(".dropdown");
let i;
for (i = 0; i < dropdowns.length; i++) {
    dropdowns[i].addEventListener("click", function (event) {
        this.classList.toggle("active");
        let content = this.nextElementSibling;

        console.log("before: " + content.style.display);
        if (content.style.padding !== "0px var(--spacing-sizing-6") {
            content.style.padding = "0px var(--spacing-sizing-6";
        } else {
            content.style.padding = "0 var(--spacing-sizing-6) var(--spacing-sizing-4) var(--spacing-sizing-6)";
        }
        console.log("after: " + content.style.display);

        if (content.style.maxHeight != "0px") {
            content.style.maxHeight = "0px";
        } else {
            content.style.maxHeight = content.scrollHeight + "px";
        }
    });
}
let show_passing_btn = document.querySelector("#show_passing_btn");
show_passing_btn.addEventListener("click", function (event) {
    this.classList.toggle("show_passing_btn_active");
    if (this.textContent === "Show") {
        this.textContent = "Hide";
    } else {
        this.textContent = "Show";
    }

    let content = document.querySelector("#passing_cards");
    if (content.style.overflow === "visible" || !content.style.overflow) {
        content.style.overflow = "hidden";
    } else {
        content.style.overflow = "visible";
    }
    if (content.style.maxHeight != "0px" || !content.style.maxHeight) {
        content.style.maxHeight = "0px";
    } else {
        content.style.maxHeight = content.scrollHeight + "px";
    }
});
