let dropdowns = document.querySelectorAll(".dropdown");
let i;
for (i = 0; i < dropdowns.length; i++) {
    dropdowns[i].addEventListener("click", function (event) {
        this.classList.toggle("active");

        let content = this.nextElementSibling;
        // if (this.classList.contains("dropdown_learn")) {
        //     content = this.nextElementSibling;
        // } else {
        //     content = this.previousElementSibling;
        // }
        // console.log(this.classList);

        console.log("before: " + content.style.display);
        // flex, null
        // none, content.style.display
        if (content.style.display === "none") {
            content.style.display = "flex";
        } else {
            content.style.display = "none";
        }
        console.log("after: " + content.style.display);

        if (content.style.maxHeight != "0px") {
            content.style.maxHeight = "0px";
            // content.style.padding = "0px";
        } else {
            content.style.maxHeight = content.scrollHeight + "px";
            // content.style.padding = "0 var(--spacing-sizing-6) var(--spacing-sizing-4) var(--spacing-sizing-6)";
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
    if (content.style.overflow === "visible") {
        content.style.overflow = "hidden";
    } else {
        content.style.overflow = "visible";
    }
    if (content.style.maxHeight) {
        content.style.maxHeight = null;
    } else {
        content.style.maxHeight = "300vh";
    }
});
