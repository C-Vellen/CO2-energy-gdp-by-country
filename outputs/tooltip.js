const svg = document.querySelector("svg")
const svgNS = "http://www.w3.org/2000/svg"

const countries = svg.querySelectorAll(".country")
const units = {
    "co2": "Mt/year",
    "energy": "kWh/year",
    "gdp": "$/year",
    "population": "",
    "co2_per_unit_energy": "kg/kWh",
    "energy_per_gdp": "kWh/$(2011)",
    "gdp_per_capita": "$(2011)/capita/year",
}

const tooltip = document.createElementNS(svgNS, "g")
const tooltipBorder = document.createElementNS(svgNS, "rect")
const tooltipText = document.createElementNS(svgNS, "g")
svg.appendChild(tooltip)
tooltip.appendChild(tooltipBorder)
tooltip.appendChild(tooltipText)

// tooltip template
tooltip.setAttribute('transform', 'translate(1000, 100)')
tooltipBorder.setAttribute('x', '0')
tooltipBorder.setAttribute('y', '0')
tooltipBorder.setAttribute('width', '500')
tooltipBorder.setAttribute('height', '200')
tooltipBorder.setAttribute('fill', '#fff')
tooltipBorder.setAttribute('stroke', '#000')
tooltip.setAttribute('opacity', '0')
tooltipText.setAttributeNS(null, 'x', '20')
tooltipText.setAttributeNS(null, 'y', '30')
tooltipText.setAttributeNS(null, 'font-family', 'Helvetica')
let i = 0
const country1 = countries[0]
for (item in country1.dataset) {
    i += 1
    const line = document.createElementNS(svgNS, "text")
    tooltipText.appendChild(line)
    line.setAttributeNS(null, 'x', '5')
    line.setAttributeNS(null, 'y', `${30 * i}`)
    line.setAttributeNS(null, 'font-size', '20')
    line.textContent = `${item}: ${country1.dataset[item]}`
}

// conversion coordonnées pointeur / coordonnées svg
const cursorPoint = (e) => {
    const pt = svg.createSVGPoint()
    pt.x = e.clientX; pt.y = e.clientY
    return pt.matrixTransform(svg.getScreenCTM().inverse())
}


function tooltipDisplay(e) {

    const pointerX = cursorPoint(e).x - 250
    const pointerY = cursorPoint(e).y + 50
    e.target.setAttribute("opacity", "1.0")
    tooltip.setAttribute("transform", `translate(${pointerX},${pointerY})`)
    tooltip.setAttribute('opacity', '0.9')

    const dataObject = {}
    for (item in e.target.dataset) {
        dataObject[item] = e.target.dataset[item]
    }
    const lines = tooltip.querySelectorAll("text")
    const minLength = Math.min(lines.length, Object.keys(dataObject).length)

    tooltipBorder.setAttribute('height', `${(minLength + 1) * 30}`)

    for (let i = 0; i < minLength; i++) {
        const line = lines[i]
        const key = Object.keys(dataObject)[i]
        const value = Object.values(dataObject)[i]
        line.textContent = `${key.replace('_per_', '/')}: ${value}`
        if (key == "isocode") { line.textContent = line.textContent.slice(0, 12) }
        if (Object.keys(units).includes(key)) { line.textContent += ` ${units[key]}` }
    }
    console.log("**", minLength)
}

function tooltipHide(e) {
    e.target.setAttribute("opacity", "0.8")
    tooltip.setAttribute('opacity', '0')
    const lines = tooltip.querySelectorAll("text")
    lines.forEach((line) => {
        line.textContent = ""
    })
}

countries.forEach(c => { c.addEventListener("pointermove", tooltipDisplay) })
countries.forEach(c => { c.addEventListener("pointerleave", tooltipHide) })

