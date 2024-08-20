import {
    pseudocode_elements_gbili,
    pseudocode_elements_lgc,
    pseudocode_elements_rgcli
} from './start.js';

import {
    drawLegend,
    generateAllExplanations,
    generateTabs,
    highlightNearestNodes,
    highlightNodes,
    resetHighlight,
    resetPseudocodeHighligth,
    updateNodes
} from "./common.js";

import {
    plotAffinity,
    plotFinal,
    plotIteration,
    plotS
} from "./lgc.js";

const colorScale = d3.scaleOrdinal(d3.schemeCategory10);

function plotdistanceMatrix(stepInfo, nodeGroup, linkGroup, D) {
    stepInfo.innerHTML = "";

    let table = document.createElement("table");
    table.id = "distanceTable";
    table.className = "display";

    let thead = document.createElement("thead");
    let headerRow = document.createElement("tr");
    let th = document.createElement("th");
    th.innerHTML = "";
    headerRow.appendChild(th);
    for (let i = 0; i < D.length; i++) {
        let th = document.createElement("th");
        th.innerHTML = i;
        headerRow.appendChild(th);
    }
    thead.appendChild(headerRow);
    table.appendChild(thead);

    let tbody = document.createElement("tbody");
    for (let i = 0; i < D.length; i++) {
        let row = document.createElement("tr");
        let th = document.createElement("th");
        th.innerHTML = i;
        row.appendChild(th);
        for (let j = 0; j < D.length; j++) {
            let td = document.createElement("td");
            td.innerHTML = D[i][j];
            row.appendChild(td);
        }

        row.addEventListener('click', function () {
            resetHighlight(nodeGroup, linkGroup);
            highlightNodes(nodeGroup, [i]);
        });

        tbody.appendChild(row);
    }
    table.appendChild(tbody);

    stepInfo.appendChild(table);


    $(document).ready(function () {
        $('#distanceTable').DataTable({
            responsive: true,
            scrollX: true,
            scrollY: '600px',
            scrollCollapse: true,
            paging: true,
            searching: false,
            ordering: false
        });

    });

}

function initializeNodes(data, nodesMap) {
    data.nodes.forEach(node => {
        if (!nodesMap.has(node.id)) {
            nodesMap.set(node.id, {...node});
        } else {
            let existingNode = nodesMap.get(node.id);
            node.x = existingNode.x;
            node.y = existingNode.y;
            node.fx = existingNode.fx;
            node.fy = existingNode.fy;
        }
    });
}

export function drawGBILI(steps, components_semi, components_graph, mapping) {

    function activateExplanation(num_step) {

        let tables = {
            0: '#distanceTable',
            1: '#neighborsTable',
            2: '#mneighborsTable',
            5: '#similarityTable',
            6: '#dTable',
            7: '#inferenceTable',
            8: '#finallabelsTable'
        };


        for (let i = 0; i < document.querySelectorAll('[id^="stepExplanation"]').length; i++) {
            let stepExplanation = document.getElementById(`stepExplanation${i}`);
            let stepInfo = document.getElementById(`stepInfo${i}`);

            if (i === num_step) {
                stepExplanation.style.display = 'block';
                stepInfo.style.display = 'block';
                if (i in tables) {
                    $(document).ready(function () {
                        $(tables[i]).DataTable().columns.adjust();
                    });
                }
            } else {
                stepExplanation.style.display = 'none';
                stepInfo.style.display = 'none';
            }
        }
    }

    function plotneighborsMatrix(stepInfo, nodeGroup, linkGroup, D) {

        stepInfo.innerHTML = `
          <div class="alert alert-info" role="alert">
              <p>Seleccione una fila de la tabla para mostrar el vecindario.</p>
          </div>
         `;

        let table = document.createElement("table");
        table.id = "neighborsTable";
        table.className = "display";

        let thead = document.createElement("thead");
        let headerRow = document.createElement("tr");
        let th = document.createElement("th");
        th.innerHTML = "";
        headerRow.appendChild(th);
        for (let i = 0; i < D[0].length; i++) {
            let th = document.createElement("th");
            th.innerHTML = i;
            headerRow.appendChild(th);
        }
        thead.appendChild(headerRow);
        table.appendChild(thead);

        let tbody = document.createElement("tbody");
        for (let i = 0; i < D.length; i++) {
            let row = document.createElement("tr");
            row.classList.add("user-select-none");
            let th = document.createElement("th");
            th.innerHTML = i;
            row.appendChild(th);
            for (let j = 0; j < D[i].length; j++) {
                let td = document.createElement("td");
                td.innerHTML = D[i][j];
                row.appendChild(td);
            }
            row.addEventListener('click', function () {
                resetHighlight(nodeGroup, linkGroup);
                highlightNearestNodes(nodeGroup, i, D[i]);
            });
            tbody.appendChild(row);
        }
        table.appendChild(tbody);

        stepInfo.appendChild(table);

        $(document).ready(function () {
            $('#neighborsTable').DataTable({
                responsive: true,
                scrollX: true,
                scrollY: '600px',
                scrollCollapse: true,
                paging: true,
                searching: false,
                columnDefs: [
                    {targets: 1, visible: false}
                ],
                ordering: false
            });

        });

    }

    function plotmutualneighborsMatrix(stepInfo, nodeGroup, linkGroup, M) {
        stepInfo.innerHTML = `
          <div class="alert alert-info" role="alert">
              <p>Seleccione una fila de la tabla para mostrar los vecinos mutuos.</p>
          </div>
         `;

        let table_mutual = document.createElement("table");
        table_mutual.id = "mneighborsTable";
        table_mutual.className = "display";

        let thead_mutual = document.createElement("thead");
        let headerRow_mutual = document.createElement("tr");
        let th_empty_mutual = document.createElement("th");
        th_empty_mutual.innerHTML = "";
        headerRow_mutual.appendChild(th_empty_mutual);

        let th_l = document.createElement("th");
        th_l.innerHTML = "Vecinos mutuos";
        headerRow_mutual.appendChild(th_l);

        thead_mutual.appendChild(headerRow_mutual);
        table_mutual.appendChild(thead_mutual);

        let tbody_mutual = document.createElement("tbody");
        for (let i = 0; i < M.length; i++) {
            let row_mutual = document.createElement("tr");
            let th_row_header_mutual = document.createElement("th");
            th_row_header_mutual.innerHTML = i;
            row_mutual.appendChild(th_row_header_mutual);

            let td_mutual = document.createElement("td");
            td_mutual.innerHTML = "[" + M[i] + "]";
            row_mutual.appendChild(td_mutual);

            row_mutual.addEventListener('click', function () {
                resetHighlight(nodeGroup, linkGroup);
                highlightNearestNodes(nodeGroup, i, M[i]);
            });

            tbody_mutual.appendChild(row_mutual);
        }
        table_mutual.appendChild(tbody_mutual);

        stepInfo.appendChild(table_mutual);

        $(document).ready(function () {
            $('#mneighborsTable').DataTable({
                responsive: true,
                scrollX: true,
                scrollY: '600px',
                scrollCollapse: true,
                paging: true,
                searching: false,
                ordering: false,
                columnDefs: [
                    {className: "dt-head-center", targets: '_all'}
                ]
            });

            stepInfo.querySelector(".dt-scroll-headInner").classList.add("m-auto");
            $('#mneighborsTable').DataTable().columns.adjust();

        });

    }

    function plotsemiGraph(stepInfo) {

        stepInfo.innerHTML = `
                    <div class="alert alert-info" role="alert">
                        <p>Seleccione un nodo del grafo para mostrar información de componente.</p>
                    </div>
                `;
    }

    function highlightComponent(components, nodeId) {
        resetHighlight(nodeGroup, linkGroup);

        const componentId = components[nodeId];
        const nodesInComponent = steps[3].nodes.filter(n => components[n.id] === componentId);

        nodeGroup.selectAll("circle")
            .transition()
            .duration(20)
            .attr("r", d => nodesInComponent.some(n => n.id === d.id) ? 8 : 5)

        linkGroup.selectAll("line")
            .transition()
            .duration(20)
            .attr("stroke-width", d => nodesInComponent.some(n => n.id === d.source.id) && nodesInComponent.some(n => n.id === d.target.id) ? 2 : 1);

        infoComponent(componentId, nodesInComponent);
    }

    function infoComponent(componentId, nodesInComponent) {

        let stepInfo = document.getElementById(`stepInfo${currentStep}`);

        if (currentStep !== 4) {
            stepInfo.innerHTML = `
                    <div class="alert alert-info" role="alert">
                        <p>Seleccione un nodo del grafo para mostrar información.</p>
                    </div>
                `;
        }

        const isLabeled = nodesInComponent.some(node => node.label !== -1);

        let container = document.createElement("div");
        container.className = "card p-3 mb-3 overflow-auto";
        container.style.backgroundColor = isLabeled ? '#d4edda' : '#f8f9fa';
        container.style.height = "250px";

        let title = document.createElement("h2");
        title.className = "card-title text-center";
        title.innerHTML = `Componente ${componentId}`;
        container.appendChild(title);

        let componentInfo = document.createElement("div");
        componentInfo.className = "component-info";

        let nodeCount = document.createElement("p");
        nodeCount.className = "card-text";
        nodeCount.innerHTML = `<strong>Número de nodos:</strong> ${nodesInComponent.length}`;
        componentInfo.appendChild(nodeCount);

        let nodeList = document.createElement("ul");
        nodeList.className = "list-group list-group-flush";
        nodesInComponent.forEach(node => {
            let listItem = document.createElement("li");
            listItem.className = "list-group-item";
            listItem.innerHTML = `<strong>ID:</strong> ${node.id}, <strong>Label:</strong> ${node.label in mapping ? mapping[node.label] : 'Sin etiqueta'}`;
            nodeList.appendChild(listItem);
        });
        componentInfo.appendChild(nodeList);

        container.appendChild(componentInfo);
        stepInfo.appendChild(container);
    }


    function infoFinal(stepInfo) {

        stepInfo.innerHTML = "";

        let unionsContainer = document.createElement("div");
        unionsContainer.id = "unionsContainer";
        unionsContainer.className = "card p-3 mb-3";

        let unions = steps[4]["unions"];

        unions.sort(function (a, b) {
            return a[0] - b[0];
        });

        let ul = document.createElement("ul");
        ul.className = "list-group";
        unions.forEach(union => {
            let li = document.createElement("li");
            li.className = "list-group-item";
            const p = document.createElement('p');

            const link = document.createElement('a');
            link.href = "#";
            link.style.textDecoration = "underline";
            link.textContent = `Unión de componente ${union[0]} con componente ${union[1]}`;
            link.addEventListener('click', function (event) {
                event.preventDefault();
                highlightUnion(components_semi, union[0], union[1]);
            });

            p.appendChild(link);

            const showFinalComponentBtn = document.createElement('a');
            showFinalComponentBtn.href = "#";
            showFinalComponentBtn.style.textDecoration = "underline";
            let nodeId = components_semi.indexOf(union[0]);
            showFinalComponentBtn.textContent = `Ver nueva componente ${components_graph[nodeId]}`;

            showFinalComponentBtn.addEventListener('click', function (event) {
                event.stopPropagation();

                resetHighlight(nodeGroup, linkGroup)

                infoFinal(document.getElementById(`stepInfo${currentStep}`));
                highlightComponent(components_graph, nodeId);
            });

            li.appendChild(p);
            li.appendChild(showFinalComponentBtn);

            ul.appendChild(li);
        });

        unionsContainer.appendChild(ul);
        stepInfo.appendChild(unionsContainer);

        let aux = document.createElement("div");
        aux.className = "alert alert-info";
        aux.setAttribute("role", "alert");
        aux.innerHTML = "<p>Seleccione un nodo del grafo para mostrar información de componente.</p>";

        stepInfo.appendChild(aux);
    }

    function highlightUnion(components, component1, component2) {
        resetHighlight(nodeGroup, linkGroup);

        const nodesInComponent1 = steps[3].nodes.filter(n => components[n.id] === component1);
        const nodesInComponent2 = steps[3].nodes.filter(n => components[n.id] === component2);

        nodeGroup.selectAll("circle")
            .transition()
            .duration(20)
            .attr("r", d => (nodesInComponent1.some(n => n.id === d.id) || nodesInComponent2.some(n => n.id === d.id)) ? 8 : 5);

        linkGroup.selectAll("line")
            .transition()
            .duration(20)
            .attr("stroke", d => (nodesInComponent1.some(n => n.id === d.source.id) && nodesInComponent2.some(n => n.id === d.target.id) || nodesInComponent2.some(n => n.id === d.source.id) && nodesInComponent1.some(n => n.id === d.target.id)) ? "red" : "#999");
    }

    const width = 500;
    const height = 500;

    let currentStep = 0;

    let zoom = d3.zoom()
        .on("zoom", (event) => {
            svgGroup.attr("transform", event.transform);
        });

    let svg = d3.select("svg")
        .attr("width", width)
        .attr("height", height)
        .attr("viewBox", [-width / 2, -height / 2, width, height])
        .attr("style", "max-width: 100%; height: auto;")
        .call(zoom);

    let svgGroup = svg.append("g");

    let linkGroup = svgGroup.append("g")
        .attr("class", "links")
        .attr("stroke", "#999")
        .attr("stroke-opacity", 0.6);

    let nodeGroup = svgGroup.append("g")
        .attr("class", "nodes");

    let simulation;

    let nodesMap = new Map();

    document.querySelector('#resetZoomBtn').addEventListener('click', () => {
        svg.transition().duration(750).call(zoom.transform, d3.zoomIdentity);
    });

    let Tooltip = d3.select("body")
        .append("div")
        .attr("class", "tooltip")
        .style("opacity", 0);

    let mouseover = function (event, d) {
        Tooltip.style("opacity", 1);
    }

    let mousemove = function (event, d) {
        Tooltip
            .html("Nodo: " + d.id + "<br>Etiqueta: " +
                (d.label !== -1 ? "<span style='color:" + colorScale(d.label) + "'>" + mapping[d.label] + "</span>" : "Sin etiqueta"))
            .style("left", (event.pageX) + "px")
            .style("top", (event.pageY) + "px");
    }

    let mouseleave = function (event, d) {
        Tooltip.style("opacity", 0);
    }

    function drawInitialGraph(data, nodeGroup, linkGroup) {

        const nodes = data.nodes.map(d => ({...d}));
        const links = data.links.map(d => ({...d}));

        simulation = d3.forceSimulation(nodes)
            .force("charge", d3.forceManyBody())
            .force("x", d3.forceX())
            .force("y", d3.forceY())
            .on("tick", ticked);

        const link = linkGroup.selectAll("line")
            .data(links, d => `${d.source}-${d.target}`)
            .enter().append("line")
            .attr("class", "link")
            .attr("stroke-width", d => Math.sqrt(d.value));

        const node = nodeGroup.selectAll("circle")
            .data(nodes, d => d.id)
            .enter().append("circle")
            .attr("class", "node")
            .attr("stroke", "#999")
            .attr("stroke-width", 1)
            .attr("r", 5)
            .attr("fill", d => d.label === -1 ? "#808080" : colorScale(d.label))
            .on("click", (event, d) => {
                if (currentStep === 3) {
                    highlightComponent(components_semi, d.id);
                } else if (currentStep === 4) {
                    infoFinal(document.getElementById(`stepInfo${currentStep}`));
                    highlightComponent(components_graph, d.id);
                }
                event.stopPropagation();
            })
            .on("mouseover", mouseover)
            .on("mousemove", mousemove)
            .on("mouseleave", mouseleave)
            .call(d3.drag()
                .on("start", dragstarted)
                .on("drag", dragged)
                .on("end", dragended));

        function ticked() {
            linkGroup.selectAll("line")
                .attr("x1", d => d.source.x)
                .attr("y1", d => d.source.y)
                .attr("x2", d => d.target.x)
                .attr("y2", d => d.target.y);

            nodeGroup.selectAll("circle")
                .attr("cx", d => d.x)
                .attr("cy", d => d.y);
        }

        function dragstarted(event, d) {
            if (!event.active) simulation.alphaTarget(0.3).restart();
            d.fx = d.x;
            d.fy = d.y;
        }

        function dragged(event, d) {
            d.fx = event.x;
            d.fy = event.y;
        }

        function dragended(event, d) {
            if (!event.active) simulation.alphaTarget(0);
            d.fx = null;
            d.fy = null;
        }

        node.raise();
    }

    function updateLinks(data) {
        const links = data.links.map(d => ({...d}));

        const link = linkGroup.selectAll("line")
            .data(links, d => `${d.source}-${d.target}`);

        link.exit().remove();

        link.enter().append("line")
            .attr("class", "link")
            .attr("stroke-width", d => 1)
            .merge(link);

        simulation.force("link", d3.forceLink(links)
            .id(d => d.id)
            .strength(d => d.value * 0.1))
            .alpha(0.3).restart();
    }

    function updateGraph() {
        const currentData = steps[currentStep];
        initializeNodes(currentData, nodesMap);
        updateLinks(currentData);
    }

    document.getElementById('prevBtn').addEventListener('click', () => {
        if (currentStep > 0) {
            currentStep--;
            resetHighlight(nodeGroup, linkGroup);
            updateGraph();
            activateExplanation(currentStep);

            if (currentStep === 0) {

                resetPseudocodeHighligth(pseudocode_elements_gbili, pseudocode_elements_rgcli, pseudocode_elements_lgc);
                pseudocode_elements_gbili[2].style.backgroundColor = '#ffe975';

            } else if (currentStep === 1) {

                resetPseudocodeHighligth(pseudocode_elements_gbili, pseudocode_elements_rgcli, pseudocode_elements_lgc);

                for (let i = 6; i < 13; i++) {
                    pseudocode_elements_gbili[i].style.backgroundColor = '#ffe975'
                }

            } else if (currentStep === 2) {

                resetPseudocodeHighligth(pseudocode_elements_gbili, pseudocode_elements_rgcli, pseudocode_elements_lgc);

                for (let i = 13; i < 20; i++) {
                    pseudocode_elements_gbili[i].style.backgroundColor = '#ffe975'
                }

            } else if (currentStep === 3) {

                document.getElementById("nextBtn").innerHTML = "Siguiente";

                resetPseudocodeHighligth(pseudocode_elements_gbili, pseudocode_elements_rgcli, pseudocode_elements_lgc);

                for (let i = 20; i < 27; i++) {
                    pseudocode_elements_gbili[i].style.backgroundColor = '#ffe975'
                }
                pseudocode_elements_gbili[28].style.backgroundColor = '#ffe975'

                plotsemiGraph(document.getElementById(`stepInfo${currentStep}`));

            } else if (currentStep === 4) {

                document.getElementById("nextBtn").innerHTML = "Inferencia";
                document.getElementById("prevBtn").innerHTML = "Anterior";

                document.getElementById("gbili_pseudocode_container").style.display = "block";
                document.getElementById("lgc_pseudocode_container").style.display = "none";

                resetPseudocodeHighligth(pseudocode_elements_gbili, pseudocode_elements_rgcli, pseudocode_elements_lgc);

                for (let i = 29; i < 39; i++) {
                    pseudocode_elements_gbili[i].style.backgroundColor = '#ffe975'
                }

                infoFinal(document.getElementById(`stepInfo${currentStep}`));

            } else if (currentStep === 5) {

                document.getElementById("nextBtn").innerHTML = "Siguiente";
                document.getElementById("prevBtn").innerHTML = "Grafo";

                resetPseudocodeHighligth(pseudocode_elements_gbili, pseudocode_elements_rgcli, pseudocode_elements_lgc);
                pseudocode_elements_lgc[2].style.backgroundColor = '#ffe975'

            } else if (currentStep === 6) {

                resetPseudocodeHighligth(pseudocode_elements_gbili, pseudocode_elements_rgcli, pseudocode_elements_lgc);
                updateNodes(nodeGroup, colorScale, steps[currentStep]);
                pseudocode_elements_lgc[3].style.backgroundColor = '#ffe975'

            } else if (currentStep === 7) {
                updateNodes(nodeGroup, colorScale, steps[4]);
                document.getElementById("controlIteration").value = 0;
                document.getElementById("controlIterationLabel").innerText = 0;

                resetPseudocodeHighligth(pseudocode_elements_gbili, pseudocode_elements_rgcli, pseudocode_elements_lgc);
                pseudocode_elements_lgc[4].style.backgroundColor = '#ffe975'

            }
        }
    });

    document.getElementById('nextBtn').addEventListener('click', () => {
        if (currentStep < steps.length - 1) {
            currentStep++;
            resetHighlight(nodeGroup, linkGroup);
            updateGraph();
            activateExplanation(currentStep);

            if (currentStep === 1) {

                resetPseudocodeHighligth(pseudocode_elements_gbili, pseudocode_elements_rgcli, pseudocode_elements_lgc);

                for (let i = 6; i < 13; i++) {
                    pseudocode_elements_gbili[i].style.backgroundColor = '#ffe975'
                }

            } else if (currentStep === 2) {

                resetPseudocodeHighligth(pseudocode_elements_gbili, pseudocode_elements_rgcli, pseudocode_elements_lgc);

                for (let i = 13; i < 20; i++) {
                    pseudocode_elements_gbili[i].style.backgroundColor = '#ffe975'
                }

            } else if (currentStep === 3) {

                resetPseudocodeHighligth(pseudocode_elements_gbili, pseudocode_elements_rgcli, pseudocode_elements_lgc);

                for (let i = 20; i < 27; i++) {
                    pseudocode_elements_gbili[i].style.backgroundColor = '#ffe975'
                }
                pseudocode_elements_gbili[28].style.backgroundColor = '#ffe975'

                plotsemiGraph(document.getElementById(`stepInfo${currentStep}`));

            } else if (currentStep === 4) {

                document.getElementById("nextBtn").innerHTML = "Inferencia";
                document.getElementById("prevBtn").innerHTML = "Anterior";

                resetPseudocodeHighligth(pseudocode_elements_gbili, pseudocode_elements_rgcli, pseudocode_elements_lgc);

                for (let i = 29; i < 39; i++) {
                    pseudocode_elements_gbili[i].style.backgroundColor = '#ffe975'
                }

                infoFinal(document.getElementById(`stepInfo${currentStep}`));

            } else if (currentStep === 5) {

                document.getElementById("nextBtn").innerHTML = "Siguiente";
                document.getElementById("prevBtn").innerHTML = "Grafo";

                document.getElementById("gbili_pseudocode_container").style.display = "none";
                document.getElementById("lgc_pseudocode_container").style.display = "block";

                resetPseudocodeHighligth(pseudocode_elements_gbili, pseudocode_elements_rgcli, pseudocode_elements_lgc);
                pseudocode_elements_lgc[2].style.backgroundColor = '#ffe975'

            } else if (currentStep === 6) {

                document.getElementById("prevBtn").innerHTML = "Anterior";

                resetPseudocodeHighligth(pseudocode_elements_gbili, pseudocode_elements_rgcli, pseudocode_elements_lgc);
                pseudocode_elements_lgc[3].style.backgroundColor = '#ffe975'

            } else if (currentStep === 7) {

                resetPseudocodeHighligth(pseudocode_elements_gbili, pseudocode_elements_rgcli, pseudocode_elements_lgc);
                document.getElementById("controlIteration").value = 0;
                document.getElementById("controlIterationLabel").innerText = 0;

                pseudocode_elements_lgc[4].style.backgroundColor = '#ffe975'

            } else if (currentStep === 8) {
                updateNodes(nodeGroup, colorScale, steps[currentStep]);

                resetPseudocodeHighligth(pseudocode_elements_gbili, pseudocode_elements_rgcli, pseudocode_elements_lgc);
                pseudocode_elements_lgc[5].style.backgroundColor = '#ffe975'
            }
        }
    });

    svg.on("click", () => {
        resetHighlight(nodeGroup, linkGroup);
    });

    //Inicialización
    let allExplanations = document.getElementById("allExplanations");


    let titles = [
        'Cálculo de distancias',
        'Vecinos más cercanos',
        'Vecinos mutuos',
        'Inicio construcción grafo',
        'Grafo final',
        'Matriz de similitud',
        'Matriz S',
        'Inferencia',
        'Final'
    ];

    let explanations = [
        'En este paso se calcula la distancia a pares de cada nodo. Esta información servirá para calcular los vecinos más cercanos (y posteriormente, los vecinos mutuos).',
        'En este paso se obtienen los k vecinos más cercanos de cada nodo. El grafo representa el enlazamiento de cada nodo con sus vecinos más cercanos.',
        'En este paso se obtienen los vecinos (j) de cada nodo (i) que son mutuos. Es decir, que ese vecino (j) también tiene, entre sus k vecinos más cercanos, al nodo (i). El grafo representa el enlazamiento de estos vecinos que son mutuos.',
        'En este paso se comienza a crear el grafo final a partir de los vecinos mutuos. La intuición es obtener un vecino mutuo (j) de un nodo (i) en el que la distancia desde el nodo (i) hasta cualquier otro nodo etiquetado sea mínima <strong>pasando por el vecino (j)</strong>. Esto lo que intenta es conectar nodos que están cerca de los otros nodos etiquetados, pues durante la inferencia, nodos cercanos tienden a tener la misma etiqueta. Utilizando los vecinos mutuos se consigue reforzar esta idea, pues dos nodos que son vecinos más cercanos mutuos, son con seguridad mucho más cercanos entre ellos que con otros. <br><br> En este paso también se obtienen las componentes. Una componente es un subgrafo completamente conectado (existe un camino entre cualquier nodo) máximo (no se le puede añadir un nodo sin violar la conectividad). Este paso sirve para detectar componentes con ningún dato etiquetado.',
        'En este paso se genera el grafo final. Para los nodos que se encuentran en componentes en las que no hay nodos etiquetados, se comprueba si algún vecino cercano sí que se encuentra en una componente etiquetada y en ese caso se crea un enlace entre ellos.',
        'La matriz de similitud es una matriz cuadrada y simétrica que representa los nodos que están unidos por un enlace (y que se consideran similares). <br>La matriz de etiquetas es la matriz que se verá modificada a lo largo del proceso iterativo de inferencia. Inicialmente (en este paso) contiene el conjunto de datos original. Para los datos etiquetados se marca la etiqueta correspondiente mediante un 1 en la posición.',
        'La matriz S es una versión normalizada de la matriz de similitud (W). Esto es necesario porque los nodos con alta conectividad podrían tener más importancia que otros en el proceso de inferencia.',
        'La inferencia consiste en un proceso iterativo con operaciones matriciales. La matriz Y es la matriz de etiquetado original. <br> El deslizador inferior permite visualizar las distintas etiquetas que se asignan a los puntos del grafo durante el proceso. Dicha información se encuentra en la tabla, en cada celda, el número superior representa la etiqueta y el inferior la "confianza" en dicha etiqueta.',
        'En este momento todos los nodos han sido etiquetados según el proceso de inferencia. Se realiza el proceso de <i>argmax</i> para obtener la etiqueta final.'
    ];

    allExplanations.innerHTML = generateAllExplanations(steps.length, titles, explanations);

    plotdistanceMatrix(document.getElementById(`stepInfo${0}`), nodeGroup, linkGroup, steps[0]["distance"]);

    plotneighborsMatrix(document.getElementById(`stepInfo${1}`), nodeGroup, linkGroup, steps[1]["neighbors"]);

    plotmutualneighborsMatrix(document.getElementById(`stepInfo${2}`), nodeGroup, linkGroup, steps[2]["mneighbors"]);

    plotsemiGraph(document.getElementById(`stepInfo${3}`));

    infoFinal(document.getElementById(`stepInfo${4}`));

    plotAffinity(document.getElementById(`stepInfo${5}`), nodeGroup, linkGroup, steps[5]["W"], steps[5]["F"], mapping);

    plotS(document.getElementById(`stepInfo${6}`), steps[6]["D"], steps[6]["D_sqrt_inv"], steps[6]["S"]);

    plotIteration(document.getElementById(`stepInfo${7}`), nodeGroup, linkGroup, colorScale, steps[7]["F_history"], steps[7]["pred_history"], mapping);

    plotFinal(document.getElementById(`stepInfo${8}`), nodeGroup, linkGroup, steps[8]["F_final"], steps[8]["pred_final"], mapping, colorScale);

    drawInitialGraph(steps[currentStep], nodeGroup, linkGroup);
    drawLegend(document.getElementById("legend"), mapping, colorScale);

    document.getElementById("gbili_pseudocode_container").style.display = "block";
    resetPseudocodeHighligth(pseudocode_elements_gbili, pseudocode_elements_rgcli, pseudocode_elements_lgc);
    pseudocode_elements_gbili[2].style.backgroundColor = '#ffe975';

    activateExplanation(currentStep);

}

export function fetchGBILI(archivo, target_name, p_unlabeled, k, alpha, max_iter, threshold) {
    let formData = new FormData();
    formData.append('file', archivo);
    formData.append('target_name', target_name);
    formData.append('p_unlabeled', p_unlabeled);
    formData.append('k', k);
    formData.append('alpha', alpha);
    formData.append('max_iter', max_iter);
    formData.append('threshold', threshold);

    return fetch("/gbili_data", {
        method: 'POST',
        body: formData
    })
        .then(response => {
            if (!response.ok) {

                document.getElementById("loader").remove();
                $(".alert").show('medium');
                setTimeout(function () {
                    $(".alert").hide('medium');
                    location.reload();
                }, 3000);

                return response.text().then(text => {
                    document.getElementById("error_msg").innerHTML = text;
                    throw new Error('Error en la petición');
                });
            }
            return response.json();
        })
        .then(data => {
            let steps = [];
            steps.push(data["gbili"]["dataset"]);
            steps.push(data["gbili"]["knn"]);
            steps.push(data["gbili"]["m_knn"]);
            steps.push(data["gbili"]["semi_graph"]);
            steps.push(data["gbili"]["graph"]);
            steps.push(data["lgc"]["affinity"]);
            steps.push(data["lgc"]["S"]);
            steps.push(data["lgc"]["iteration"]);
            steps.push(data["lgc"]["labels"]);

            let components_semi = data["gbili"]["semi_graph"]["components"];
            let components_graph = data["gbili"]["graph"]["components"];

            let mapping = data["mapping"];

            return {steps, components_semi, components_graph, mapping};
        })
        .catch(error => {
            throw error;
        });
}


export function drawRGCLI(steps, mapping) {

    function activateExplanation(num_step) {

        let tables = {
            0: '#distanceTable',
            1: '#knnTable',
            3: '#similarityTable',
            4: '#dTable',
            5: '#inferenceTable',
            6: '#finallabelsTable'
        };


        for (let i = 0; i < document.querySelectorAll('[id^="stepExplanation"]').length; i++) {
            let stepExplanation = document.getElementById(`stepExplanation${i}`);
            let stepInfo = document.getElementById(`stepInfo${i}`);

            if (i === num_step) {
                stepExplanation.style.display = 'block';
                stepInfo.style.display = 'block';
                if (i in tables) {
                    $(document).ready(function () {
                        $(tables[i]).DataTable().columns.adjust();
                    });
                }
            } else {
                stepExplanation.style.display = 'none';
                stepInfo.style.display = 'none';
            }
        }
    }

    function plotnearestNeighbors(kNN, L, F, num_step) {

        let stepInfo = document.getElementById(`stepInfo${num_step}`);
        stepInfo.innerHTML = generateTabs(["kNN", "L", "F"]);

        let knn = document.getElementById("knn");
        let l = document.getElementById("l");
        let f = document.getElementById("f");

        // Crear tabla para knn
        let table_knn = document.createElement("table");
        table_knn.id = "knnTable";
        table_knn.className = "display";

        let thead_knn = document.createElement("thead");
        let headerRow_knn = document.createElement("tr");
        let th_empty_knn = document.createElement("th");
        th_empty_knn.innerHTML = "";
        headerRow_knn.appendChild(th_empty_knn);
        for (let i = 0; i < kNN[0].length; i++) {
            let th_knn = document.createElement("th");
            th_knn.innerHTML = i;
            headerRow_knn.appendChild(th_knn);
        }
        thead_knn.appendChild(headerRow_knn);
        table_knn.appendChild(thead_knn);

        let tbody_knn = document.createElement("tbody");
        for (let i = 0; i < kNN.length; i++) {
            let row_knn = document.createElement("tr");
            let th_row_header_knn = document.createElement("th");
            th_row_header_knn.innerHTML = i;
            row_knn.appendChild(th_row_header_knn);

            for (let j = 0; j < kNN[i].length; j++) {
                let td_knn = document.createElement("td");
                td_knn.innerHTML = kNN[i][j];
                row_knn.appendChild(td_knn);
            }

            row_knn.addEventListener('click', function () {
                resetHighlight(nodeGroup, linkGroup);
                highlightNearestNodes(nodeGroup, i, kNN[i]);
            });

            tbody_knn.appendChild(row_knn);
        }
        table_knn.appendChild(tbody_knn);

        knn.appendChild(table_knn);

        // Crear tabla para L
        let table_l = document.createElement("table");
        table_l.id = "lTable";
        table_l.className = "display";

        let thead_l = document.createElement("thead");
        let headerRow_l = document.createElement("tr");
        let th_empty_l = document.createElement("th");
        th_empty_l.innerHTML = "";
        headerRow_l.appendChild(th_empty_l);

        let th_l = document.createElement("th");
        th_l.innerHTML = "Etiquetado cercano";
        headerRow_l.appendChild(th_l);

        thead_l.appendChild(headerRow_l);
        table_l.appendChild(thead_l);

        let tbody_l = document.createElement("tbody");
        for (let i = 0; i < L.length; i++) {
            let row_l = document.createElement("tr");
            let th_row_header_l = document.createElement("th");
            th_row_header_l.innerHTML = i;
            row_l.appendChild(th_row_header_l);

            let td_l = document.createElement("td");
            td_l.innerHTML = L[i];
            row_l.appendChild(td_l);

            row_l.addEventListener('click', function () {
                resetHighlight(nodeGroup, linkGroup);
                highlightNearestNodes(nodeGroup, i, [L[i]]);
                //highlightNodes(nodeGroup, [i, L[i]]);
            });

            tbody_l.appendChild(row_l);
        }
        table_l.appendChild(tbody_l);

        l.appendChild(table_l);

        // Crear tabla para F
        let table_f = document.createElement("table");
        table_f.id = "fTable";
        table_f.className = "display";

        let thead_f = document.createElement("thead");
        let headerRow_f = document.createElement("tr");
        let th_empty_f = document.createElement("th");
        th_empty_f.innerHTML = "";
        headerRow_f.appendChild(th_empty_f);

        let th_f = document.createElement("th");
        th_f.innerHTML = "k más lejano";
        headerRow_f.appendChild(th_f);

        thead_f.appendChild(headerRow_f);
        table_f.appendChild(thead_f);

        let tbody_f = document.createElement("tbody");
        for (let i = 0; i < F.length; i++) {
            let row_f = document.createElement("tr");
            let th_row_header_f = document.createElement("th");
            th_row_header_f.innerHTML = i;
            row_f.appendChild(th_row_header_f);

            let td_f = document.createElement("td");

            td_f.innerHTML = F[i];
            row_f.appendChild(td_f);

            row_f.addEventListener('click', function () {
                resetHighlight(nodeGroup, linkGroup);
                highlightNearestNodes(nodeGroup, i, [F[i]]);
                //highlightNodes(nodeGroup, [i, F[i]]);
            });

            tbody_f.appendChild(row_f);
        }
        table_f.appendChild(tbody_f);

        f.appendChild(table_f);

        $(document).ready(function () {
            $('#knnTable').DataTable({
                responsive: true,
                scrollX: true,
                scrollY: '600px',
                scrollCollapse: true,
                paging: true,
                searching: false,
                ordering: false,
                columnDefs: [
                    {className: "dt-head-center", targets: '_all'}
                ]
            });

            document.querySelector("#knn .dt-scroll-headInner").classList.add("m-auto")

            $('#lTable').DataTable({
                responsive: true,
                scrollX: true,
                scrollY: '600px',
                scrollCollapse: true,
                paging: true,
                searching: false,
                ordering: false,
                columnDefs: [
                    {className: "dt-head-center", targets: '_all'}
                ]
            });

            $('#fTable').DataTable({
                responsive: true,
                scrollX: true,
                scrollY: '600px',
                scrollCollapse: true,
                paging: true,
                searching: false,
                ordering: false,
                columnDefs: [
                    {className: "dt-head-center", targets: '_all'}
                ]
            });

            $(document).on('shown.bs.tab', function (e) {
                $('#knnTable').DataTable().columns.adjust();
                $('#lTable').DataTable().columns.adjust();
                $('#fTable').DataTable().columns.adjust();
            })


            document.querySelector("#l .dt-scroll-headInner").classList.add("m-auto");
            document.querySelector("#f .dt-scroll-headInner").classList.add("m-auto");
        });

    }

    const width = 500;
    const height = 500;

    let currentStep = 0;

    let zoom = d3.zoom()
        .on("zoom", (event) => {
            svgGroup.attr("transform", event.transform);
        });

    let svg = d3.select("svg")
        .attr("width", width)
        .attr("height", height)
        .attr("viewBox", [-width / 2, -height / 2, width, height])
        .attr("style", "max-width: 100%; height: auto;")
        .call(zoom);

    let svgGroup = svg.append("g");

    let linkGroup = svgGroup.append("g")
        .attr("class", "links")
        .attr("stroke", "#999")
        .attr("stroke-opacity", 0.6);

    let nodeGroup = svgGroup.append("g")
        .attr("class", "nodes");

    let simulation;

    let nodesMap = new Map();

    document.querySelector('#resetZoomBtn').addEventListener('click', () => {
        svg.transition().duration(750).call(zoom.transform, d3.zoomIdentity);
    });

    let Tooltip = d3.select("body")
        .append("div")
        .attr("class", "tooltip")
        .style("opacity", 0);

    let mouseover = function (event, d) {
        Tooltip.style("opacity", 1);
    }

    let mousemove = function (event, d) {
        Tooltip
            .html("Nodo: " + d.id + "<br>Etiqueta: " + (d.label !== -1 ? mapping[d.label] : "Sin etiqueta"))
            .style("left", (event.pageX) + "px")
            .style("top", (event.pageY) + "px");
    }

    let mouseleave = function (event, d) {
        Tooltip.style("opacity", 0);
    }

    function drawInitialGraph(data, nodeGroup, linkGroup) {

        const nodes = data.nodes.map(d => ({...d}));
        const links = data.links.map(d => ({...d}));

        simulation = d3.forceSimulation(nodes)
            .force("charge", d3.forceManyBody())
            .force("x", d3.forceX())
            .force("y", d3.forceY())
            .on("tick", ticked);

        const link = linkGroup.selectAll("line")
            .data(links, d => `${d.source}-${d.target}`)
            .enter().append("line")
            .attr("class", "link")
            .attr("stroke-width", d => Math.sqrt(d.value));

        const node = nodeGroup.selectAll("circle")
            .data(nodes, d => d.id)
            .enter().append("circle")
            .attr("class", "node")
            .attr("stroke", "#999")
            .attr("stroke-width", 1)
            .attr("r", 5)
            .attr("fill", d => d.label === -1 ? "#808080" : colorScale(d.label))
            .on("mouseover", mouseover)
            .on("mousemove", mousemove)
            .on("mouseleave", mouseleave)
            .call(d3.drag()
                .on("start", dragstarted)
                .on("drag", dragged)
                .on("end", dragended));


        function ticked() {
            linkGroup.selectAll("line")
                .attr("x1", d => d.source.x)
                .attr("y1", d => d.source.y)
                .attr("x2", d => d.target.x)
                .attr("y2", d => d.target.y);

            nodeGroup.selectAll("circle")
                .attr("cx", d => d.x)
                .attr("cy", d => d.y);
        }

        function dragstarted(event, d) {
            if (!event.active) simulation.alphaTarget(0.3).restart();
            d.fx = d.x;
            d.fy = d.y;
        }

        function dragged(event, d) {
            d.fx = event.x;
            d.fy = event.y;
        }

        function dragended(event, d) {
            if (!event.active) simulation.alphaTarget(0);
            d.fx = null;
            d.fy = null;
        }

        node.raise();
    }

    function updateLinks(data) {
        const links = data.links.map(d => ({...d}));

        const link = linkGroup.selectAll("line")
            .data(links, d => `${d.source}-${d.target}`);

        link.exit().remove();

        link.enter().append("line")
            .attr("class", "link")
            .attr("stroke-width", d => 1)
            .merge(link);

        simulation.force("link", d3.forceLink(links)
            .id(d => d.id)
            .strength(d => d.value * 0.1))
            .alpha(0.3).restart();
    }

    function updateGraph() {
        const currentData = steps[currentStep];
        initializeNodes(currentData, nodesMap);
        updateLinks(currentData);
    }


    document.getElementById('prevBtn').addEventListener('click', () => {
        if (currentStep > 0) {
            currentStep--;
            resetHighlight(nodeGroup, linkGroup);
            updateGraph();
            activateExplanation(currentStep);

            if (currentStep === 0) {
                resetPseudocodeHighligth(pseudocode_elements_gbili, pseudocode_elements_rgcli, pseudocode_elements_lgc);
                pseudocode_elements_rgcli[1].style.backgroundColor = '#ffe975';

            } else if (currentStep === 1) {

                resetPseudocodeHighligth(pseudocode_elements_gbili, pseudocode_elements_rgcli, pseudocode_elements_lgc);
                for (let i = 11; i < 15; i++) {
                    pseudocode_elements_rgcli[i].style.backgroundColor = '#ffe975';
                }
                for (let i = 19; i < 26; i++) {
                    pseudocode_elements_rgcli[i].style.backgroundColor = '#ffe975';
                }

                document.getElementById("nextBtn").innerHTML = "Siguiente";

            } else if (currentStep === 2) {

                resetPseudocodeHighligth(pseudocode_elements_gbili, pseudocode_elements_rgcli, pseudocode_elements_lgc);
                for (let i = 15; i < 19; i++) {
                    pseudocode_elements_rgcli[i].style.backgroundColor = '#ffe975';
                }
                for (let i = 26; i < 40; i++) {
                    pseudocode_elements_rgcli[i].style.backgroundColor = '#ffe975';
                }

                document.getElementById("nextBtn").innerHTML = "Inferencia";
                document.getElementById("prevBtn").innerHTML = "Anterior";

                document.getElementById("rgcli_pseudocode_container").style.display = "block";
                document.getElementById("lgc_pseudocode_container").style.display = "none";

            } else if (currentStep === 3) {

                document.getElementById("nextBtn").innerHTML = "Siguiente";
                document.getElementById("prevBtn").innerHTML = "Grafo";

                resetPseudocodeHighligth(pseudocode_elements_gbili, pseudocode_elements_rgcli, pseudocode_elements_lgc);
                pseudocode_elements_lgc[2].style.backgroundColor = '#ffe975'

            } else if (currentStep === 4) {

                resetPseudocodeHighligth(pseudocode_elements_gbili, pseudocode_elements_rgcli, pseudocode_elements_lgc);
                updateNodes(nodeGroup, colorScale, steps[currentStep]);
                pseudocode_elements_lgc[3].style.backgroundColor = '#ffe975'

            } else if (currentStep === 5) {
                updateNodes(nodeGroup, colorScale, steps[4]);
                document.getElementById("controlIteration").value = 0;
                document.getElementById("controlIterationLabel").innerText = 0;

                resetPseudocodeHighligth(pseudocode_elements_gbili, pseudocode_elements_rgcli, pseudocode_elements_lgc);
                pseudocode_elements_lgc[4].style.backgroundColor = '#ffe975'

            }
        }
    });

    document.getElementById('nextBtn').addEventListener('click', () => {
        if (currentStep < steps.length - 1) {
            currentStep++;
            resetHighlight(nodeGroup, linkGroup);
            updateGraph();
            activateExplanation(currentStep);

            if (currentStep === 1) {

                resetPseudocodeHighligth(pseudocode_elements_gbili, pseudocode_elements_rgcli, pseudocode_elements_lgc);
                for (let i = 11; i < 15; i++) {
                    pseudocode_elements_rgcli[i].style.backgroundColor = '#ffe975';
                }
                for (let i = 19; i < 26; i++) {
                    pseudocode_elements_rgcli[i].style.backgroundColor = '#ffe975';
                }


            } else if (currentStep === 2) {

                document.getElementById("nextBtn").innerHTML = "Inferencia";

                resetPseudocodeHighligth(pseudocode_elements_gbili, pseudocode_elements_rgcli, pseudocode_elements_lgc);
                for (let i = 15; i < 19; i++) {
                    pseudocode_elements_rgcli[i].style.backgroundColor = '#ffe975';
                }
                for (let i = 26; i < 40; i++) {
                    pseudocode_elements_rgcli[i].style.backgroundColor = '#ffe975';
                }

            } else if (currentStep === 3) {

                document.getElementById("nextBtn").innerHTML = "Siguiente";
                document.getElementById("prevBtn").innerHTML = "Grafo";

                document.getElementById("rgcli_pseudocode_container").style.display = "none";
                document.getElementById("lgc_pseudocode_container").style.display = "block";

                resetPseudocodeHighligth(pseudocode_elements_gbili, pseudocode_elements_rgcli, pseudocode_elements_lgc);
                pseudocode_elements_lgc[2].style.backgroundColor = '#ffe975'

            } else if (currentStep === 4) {

                document.getElementById("prevBtn").innerHTML = "Anterior";

                resetPseudocodeHighligth(pseudocode_elements_gbili, pseudocode_elements_rgcli, pseudocode_elements_lgc);
                pseudocode_elements_lgc[3].style.backgroundColor = '#ffe975'

            } else if (currentStep === 5) {

                resetPseudocodeHighligth(pseudocode_elements_gbili, pseudocode_elements_rgcli, pseudocode_elements_lgc);
                document.getElementById("controlIteration").value = 0;
                document.getElementById("controlIterationLabel").innerText = 0;

                pseudocode_elements_lgc[4].style.backgroundColor = '#ffe975'

            } else if (currentStep === 6) {
                updateNodes(nodeGroup, colorScale, steps[currentStep]);

                resetPseudocodeHighligth(pseudocode_elements_gbili, pseudocode_elements_rgcli, pseudocode_elements_lgc);
                pseudocode_elements_lgc[5].style.backgroundColor = '#ffe975'
            }
        }
    });

    svg.on("click", () => {
        resetHighlight(nodeGroup, linkGroup);
    });

    //Inicialización
    let allExplanations = document.getElementById("allExplanations");

    let titles = [
        'Conjunto de datos',
        'Vecinos más cercanos',
        'Grafo final',
        'Matriz de similitud',
        'Matriz S',
        'Inferencia',
        'Final'
    ];

    let explanations = [
        'La información de las distancias servirá para calcular los vecinos más cercanos.',
        'En este paso se obtienen los vecinos más cercanos de cada nodo, así como el puntos etiquetado más cercanos y el k más lejano.',
        'En este paso se genera el grafo final. Se calculan los vecinos más cercanos mutuos (y sus puntuaciones epsilon ) y establece ki enlaces con los vecinos que minimizan la puntuación epsilon. <br> Las puntuaciones se calculan como la distancia del un nodo (i) a un vecino más cercano mutuo (j) mas la distancia de este último (j) al nodo etiquetado más cercano de él. ',
        'La matriz de similitud es una matriz cuadrada y simétrica que representa los nodos que están unidos por un enlace (y que se consideran similares). <br>La matriz de etiquetas es la matriz que se verá modificada a lo largo del proceso iterativo de inferencia. Inicialmente (en este paso) contiene el conjunto de datos original. Para los datos etiquetados se marca la etiqueta correspondiente mediante un 1 en la posición.',
        'La matriz S es una versión normalizada de la matriz de similitud (W). Esto es necesario porque los nodos con alta conectividad podrían tener más importancia que otros en el proceso de inferencia.',
        'La inferencia consiste en un proceso iterativo con operaciones matriciales. La matriz Y es la matriz de etiquetado original. <br> El deslizador inferior permite visualizar las distintas etiquetas que se asignan a los puntos del grafo durante el proceso. Dicha información se encuentra en la tabla, en cada celda, el número superior representa la etiqueta y el inferior la "confianza" en dicha etiqueta.',
        'En este momento todos los nodos han sido etiquetados según el proceso de inferencia. Se realiza el proceso de <i>argmax</i> para obtener la etiqueta final.'
    ];

    allExplanations.innerHTML = generateAllExplanations(steps.length, titles, explanations);

    plotdistanceMatrix(document.getElementById(`stepInfo${0}`), nodeGroup, linkGroup, steps[0]["distance"]);

    plotnearestNeighbors(steps[1]["kNN"], steps[1]["L"], steps[1]["F"], 1);

    plotAffinity(document.getElementById(`stepInfo${3}`), nodeGroup, linkGroup, steps[3]["W"], steps[3]["F"], mapping);

    plotS(document.getElementById(`stepInfo${4}`), steps[4]["D"], steps[4]["D_sqrt_inv"], steps[4]["S"]);

    plotIteration(document.getElementById(`stepInfo${5}`), nodeGroup, linkGroup, colorScale, steps[5]["F_history"], steps[5]["pred_history"], mapping);

    plotFinal(document.getElementById(`stepInfo${6}`), nodeGroup, linkGroup, steps[6]["F_final"], steps[6]["pred_final"], mapping, colorScale);

    drawInitialGraph(steps[currentStep], nodeGroup, linkGroup);
    drawLegend(document.getElementById("legend"), mapping, colorScale);

    document.getElementById("rgcli_pseudocode_container").style.display = "block";
    resetPseudocodeHighligth(pseudocode_elements_gbili, pseudocode_elements_rgcli, pseudocode_elements_lgc);
    pseudocode_elements_rgcli[1].style.backgroundColor = '#ffe975';

    activateExplanation(currentStep);
}


export function fetchRGCLI(archivo, target_name, p_unlabeled, k_e, k_i, nt, alpha, max_iter, threshold) {
    let formData = new FormData();
    formData.append('file', archivo);
    formData.append('target_name', target_name);
    formData.append('p_unlabeled', p_unlabeled);
    formData.append('k_e', k_e);
    formData.append('k_i', k_i);
    formData.append('nt', nt);
    formData.append('alpha', alpha);
    formData.append('max_iter', max_iter);
    formData.append('threshold', threshold);

    return fetch("/rgcli_data", {
        method: 'POST',
        body: formData
    })
        .then(response => {
            if (!response.ok) {

                document.getElementById("loader").remove();
                $(".alert").show('medium');
                setTimeout(function () {
                    $(".alert").hide('medium');
                    location.reload();
                }, 3000);

                return response.text().then(text => {
                    document.getElementById("error_msg").innerHTML = text;
                    throw new Error('Error en la petición');
                });
            }
            return response.json();
        })
        .then(data => {
            let steps = [];
            steps.push(data["rgcli"]["dataset"]);
            steps.push(data["rgcli"]["searchknn"]);
            steps.push(data["rgcli"]["graph"]);
            steps.push(data["lgc"]["affinity"]);
            steps.push(data["lgc"]["S"]);
            steps.push(data["lgc"]["iteration"]);
            steps.push(data["lgc"]["labels"]);

            let mapping = data["mapping"];

            return {steps, mapping};
        })
        .catch(error => {
            throw error;
        });
}