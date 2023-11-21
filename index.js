function toggleChevron(dropdown) {
    let elem = dropdown.querySelector("#chevron");
    elem.classList.toggle("fa-chevron-down");
    elem.classList.toggle("fa-chevron-up");
    // if (elem.classList.contains("fa-chevron-down")) {
    //     elem.classList.remove("fa-chevron-down");
    //     elem.classList.add("fa-chevron-up");
    // } else {
    //     elem.classList.remove("fa-chevron-up");
    //     elem.classList.add("fa-chevron-down");
    // }
}

let dropdowns = document.querySelectorAll(".dropdown");
let i;
for (i = 0; i < dropdowns.length; i++) {
    dropdowns[i].addEventListener("click", function (event) {
        this.classList.toggle("active");

        let content;
        content = this.nextElementSibling;
        // if (this.classList.contains("dropdown_learn")) {
        //     content = this.nextElementSibling;
        // } else {
        //     content = this.previousElementSibling;
        // }
        // console.log(this.classList);

        if (content.style.display === "flex") {
            content.style.display = "none";
        } else {
            content.style.display = "flex";
        }

        if (content.style.maxHeight) {
            content.style.maxHeight = null;
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

    let content = document.querySelector("#passing_tests");
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
