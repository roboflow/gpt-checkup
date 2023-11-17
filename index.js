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
        var content = this.nextElementSibling;

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
