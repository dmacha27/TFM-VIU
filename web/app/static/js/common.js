export function getElementsInDFS(root) {
    let elements = [];

    function dfs(node) {
        node.childNodes.forEach(child => {
            if (child.nodeType === Node.ELEMENT_NODE && child.tagName.toLowerCase() !== 'span') {
                if (child.tagName.toLowerCase() !== 'div') {
                    elements.push(child);
                }
                dfs(child);
            }
        });
    }

    dfs(root);
    return elements;
}

export function resetPseudocodeHighligth(pseudocode_elements_gbili, pseudocode_elements_rgcli, pseudocode_elements_lgc) {

    pseudocode_elements_gbili.forEach(element => {
        element.style.backgroundColor = '';
    });

    pseudocode_elements_rgcli.forEach(element => {
        element.style.backgroundColor = '';
    });

    pseudocode_elements_lgc.forEach(element => {
        element.style.backgroundColor = '';
    });

}

export function generateAllExplanations(num_steps, titles, explanations) {

    let html = '';

    for (let i = 0; i < num_steps; i++) {
        html += `
        <div id="stepExplanation${i}" class="p-3 border bg-light text-center rounded shadow-sm mb-1" style="display: none">
            <div class="accordion" id="accordion">
                <div class="accordion-item">
                    <h2 class="accordion-header" id="headingOne">
                        <button class="stepTitle${i} accordion-button collapsed" type="button" data-bs-toggle="collapse"
                                data-bs-target="#collapseOne${i}" aria-controls="collapseOne${i}">
                                ${titles[i]}
                        </button>
                    </h2>
                    <div id="collapseOne${i}" class="accordion-collapse collapse" aria-labelledby="headingOne"
                         data-bs-parent="#accordion">
                        <div id="explanation${i}" class="accordion-body">
                        ${explanations[i]}
                        </div>
                    </div>
                </div>
            </div>
        </div>
        <div id="stepInfo${i}" class="p-3 border bg-light text-center rounded shadow-sm" style="display: none">
        </div>`;
    }
    return html;
}

export function generateTabs(tabs) {
    let navTabs = '<ul class="nav nav-tabs" id="myTab" role="tablist">';
    let tabContent = '<div class="tab-content" id="myTabContent">';

    tabs.forEach((name, index) => {
        let isActive = index === 0 ? 'active' : '';
        let isSelected = index === 0 ? 'true' : 'false';
        navTabs += `
            <li class="nav-item" role="presentation">
                <button class="nav-link ${isActive}" id="${name.toLowerCase()}-tab" data-bs-toggle="tab" data-bs-target="#${name.toLowerCase()}" type="button" role="tab" aria-controls="${name.toLowerCase()}" aria-selected="${isSelected}">${name}</button>
            </li>
        `;
        tabContent += `
            <div class="tab-pane fade show ${isActive}" id="${name.toLowerCase()}" role="tabpanel" aria-labelledby="${name.toLowerCase()}-tab">
            </div>
        `;
    });

    navTabs += '</ul>';
    tabContent += '</div>';

    return navTabs + tabContent;
}

export function drawLegend(legend, mapping, colorScale) {
    legend.innerHTML = '';

    for (const key in mapping) {
        let item = `<div class="d-flex align-items-center m-1">
            <div class="legend-dot" style="background-color: ${colorScale(parseInt(key))}"></div>
            <div>${mapping[key]}</div>
            </div>`
        legend.innerHTML += item;
    }
}

export function resetHighlight(nodeGroup, linkGroup) {
    nodeGroup.selectAll("circle")
        .transition()
        .duration(20)
        .attr("r", 5)
        .style("stroke", "#999");

    linkGroup.selectAll("line")
        .attr("stroke-width", 1);

    linkGroup.selectAll("line")
        .transition()
        .duration(20)
        .attr("stroke", "#999");
}

export function updateNodes(nodeGroup, colorScale, data) {

    let changed = [];

    nodeGroup.selectAll("circle")
        .each(function (d) {

            let old = d.label;
            let current = data["nodes"][d.index].label;
            d.label = current

            // El primero controla si se ha etiquetado un nodo, el segundo controla que no se guarde uno que se ha desetiquetado (paso hacia atrás)
            if ((old !== current) && (current !== -1)) {
                changed.push(d.index)
            }
        })
        .attr("fill", d => d.label === -1 ? "#808080" : colorScale(d.label))

    nodeGroup.selectAll("circle")
        .filter(d => changed.includes(d.index))
        .transition()
        .duration(300)
        .attr("r", 10)
        .transition()
        .duration(300)
        .attr("r", 5);
}

export function highlightNodes(nodeGroup, nodeIndex) {
    nodeGroup.selectAll("circle")
        .filter(function (d) {
            return nodeIndex.includes(d.index);
        })
        .transition()
        .duration(20)
        .attr("r", 8);
}

export function highlightNodesAndSimilar(nodeGroup, linkGroup, nodeIndex, similarIndex) {
    nodeGroup.selectAll("circle")
        .filter(function (d) {
            return d.index === nodeIndex;
        })
        .transition()
        .duration(20)
        .attr("r", 12)
        .style("stroke", "red");

    nodeGroup.selectAll("circle")
        .filter(function (d) {
            return similarIndex.includes(d.index);
        })
        .transition()
        .duration(20)
        .attr("r", 8)
        .style("stroke", "yellow");
}

export function highlightNearestNodes(nodeGroup, nodeIndex, sortedNeighbors) {

    nodeGroup.selectAll("circle")
        .filter(function (d) {
            return sortedNeighbors.includes(d.index);
        })
        .transition()
        .duration(20)
        .attr("r", 8)
        .style("stroke", "yellow");

    nodeGroup.selectAll("circle")
        .filter(function (d) {
            return d.index === nodeIndex;
        })
        .transition()
        .duration(20)
        .attr("r", 9)
        .style("stroke", "red");


}