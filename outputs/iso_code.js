const countries = document.querySelectorAll("country")

countries.forEach(c => {
    c.addEventListener("click", e => {
        console.log(e.target.classList[1])

    })


})
