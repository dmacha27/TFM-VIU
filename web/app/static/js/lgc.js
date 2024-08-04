import {generateTabs, highlightNodes, highlightNodesAndSimilar, resetHighlight} from "./common.js";

export function plotAffinity(stepInfo, nodeGroup, linkGroup, W, F, mapping) {

    let aux = document.createElement("div");
    aux.className = "alert alert-info p-0 pt-1";
    aux.innerHTML = "<p>Seleccione una fila para mostrar información en el grafo.</p>";

    stepInfo.appendChild(aux);

    stepInfo.innerHTML += generateTabs(["Similitud", "Etiquetas"]);

    let similarity = document.getElementById("similitud");
    let labels = document.getElementById("etiquetas");

    let table_similarity = document.createElement("table");
    table_similarity.id = "similarityTable";
    table_similarity.className = "display";

    let thead_similarity = document.createElement("thead");
    let headerRow_similarity = document.createElement("tr");
    let th_similarity = document.createElement("th");
    th_similarity.innerHTML = "";
    headerRow_similarity.appendChild(th_similarity);
    for (let i = 0; i < W.length; i++) {
        let th_similarity = document.createElement("th");
        th_similarity.innerHTML = i;
        headerRow_similarity.appendChild(th_similarity);
    }
    thead_similarity.appendChild(headerRow_similarity);
    table_similarity.appendChild(thead_similarity);

    let tbody_similarity = document.createElement("tbody");
    for (let i = 0; i < W.length; i++) {
        let row_similarity = document.createElement("tr");
        let th_similarity = document.createElement("th");
        th_similarity.innerHTML = i;
        row_similarity.appendChild(th_similarity);

        let similar = [];
        for (let j = 0; j < W.length; j++) {
            let td_similarity = document.createElement("td");
            td_similarity.innerHTML = W[i][j];
            if (W[i][j] === 1) {
                similar.push(j);
                td_similarity.style.backgroundColor = 'green';
            }
            row_similarity.appendChild(td_similarity);
        }

        row_similarity.addEventListener('click', function () {
            resetHighlight(nodeGroup, linkGroup);
            highlightNodesAndSimilar(nodeGroup, linkGroup, i, similar);
        });

        tbody_similarity.appendChild(row_similarity);
    }
    table_similarity.appendChild(tbody_similarity);

    similarity.appendChild(table_similarity);


    let table_labels = document.createElement("table");
    table_labels.id = "labelsTable";
    table_labels.className = "display";

    let thead_labels = document.createElement("thead");
    let headerRow_labels = document.createElement("tr");
    let th_empty = document.createElement("th");
    th_empty.innerHTML = "";
    headerRow_labels.appendChild(th_empty);

    for (let i = 0; i < F[0].length; i++) {
        let th_labels = document.createElement("th");
        th_labels.innerHTML = mapping[i];
        headerRow_labels.appendChild(th_labels);
    }

    thead_labels.appendChild(headerRow_labels);
    table_labels.appendChild(thead_labels);

    let tbody_labels = document.createElement("tbody");
    for (let i = 0; i < F.length; i++) {
        let row_labels = document.createElement("tr");
        let th_row_header = document.createElement("th");
        th_row_header.innerHTML = i;
        row_labels.appendChild(th_row_header);

        for (let j = 0; j < F[i].length; j++) {
            let td_labels = document.createElement("td");
            td_labels.innerHTML = F[i][j];
            row_labels.appendChild(td_labels);
        }
        row_labels.addEventListener('click', function () {
            resetHighlight(nodeGroup, linkGroup);
            highlightNodes(nodeGroup, [i]);
        });

        tbody_labels.appendChild(row_labels);
    }

    table_labels.appendChild(tbody_labels);

    labels.appendChild(table_labels);

    $(document).ready(function () {
        $('#similarityTable').DataTable({
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

        $('#labelsTable').DataTable({
            responsive: true,
            scrollX: true,
            scrollY: '600px',
            scrollCollapse: true,
            paging: false,
            searching: false,
            ordering: false,
            columnDefs: [
                {className: "dt-head-center", targets: '_all'}
            ]
        });

        $(document).on('shown.bs.tab', function (e) {
            $('#labelsTable').DataTable().columns.adjust();
        })

        document.querySelector("#etiquetas .dt-scroll-headInner").classList.add("m-auto");

    });

}

export function plotS(stepInfo, D, D_sqrt_inv, S) {
    stepInfo.innerHTML = generateTabs(["D", "D_sqrt_inv", "S"]);

    let d = document.getElementById("d");
    let d_sqrt_inv = document.getElementById("d_sqrt_inv");
    let s = document.getElementById("s");

    // Crear tabla para D
    let table_d = document.createElement("table");
    table_d.id = "dTable";
    table_d.className = "display";

    let thead_d = document.createElement("thead");
    let headerRow_d = document.createElement("tr");
    let th_empty_d = document.createElement("th");
    th_empty_d.innerHTML = "";
    headerRow_d.appendChild(th_empty_d);
    for (let i = 0; i < D.length; i++) {
        let th_d = document.createElement("th");
        th_d.innerHTML = i;
        headerRow_d.appendChild(th_d);
    }
    thead_d.appendChild(headerRow_d);
    table_d.appendChild(thead_d);

    let tbody_d = document.createElement("tbody");
    for (let i = 0; i < D.length; i++) {
        let row_d = document.createElement("tr");
        let th_row_header_d = document.createElement("th");
        th_row_header_d.innerHTML = i;
        row_d.appendChild(th_row_header_d);

        for (let j = 0; j < D[i].length; j++) {
            let td_d = document.createElement("td");
            td_d.innerHTML = D[i][j];
            row_d.appendChild(td_d);
        }
        tbody_d.appendChild(row_d);
    }
    table_d.appendChild(tbody_d);

    d.appendChild(table_d);

    // Crear tabla para D_sqrt_inv
    let table_d_sqrt_inv = document.createElement("table");
    table_d_sqrt_inv.id = "d_sqrt_invTable";
    table_d_sqrt_inv.className = "display";

    let thead_d_sqrt_inv = document.createElement("thead");
    let headerRow_d_sqrt_inv = document.createElement("tr");
    let th_empty_d_sqrt_inv = document.createElement("th");
    th_empty_d_sqrt_inv.innerHTML = "";
    headerRow_d_sqrt_inv.appendChild(th_empty_d_sqrt_inv);
    for (let i = 0; i < D_sqrt_inv.length; i++) {
        let th_d_sqrt_inv = document.createElement("th");
        th_d_sqrt_inv.innerHTML = i;
        headerRow_d_sqrt_inv.appendChild(th_d_sqrt_inv);
    }
    thead_d_sqrt_inv.appendChild(headerRow_d_sqrt_inv);
    table_d_sqrt_inv.appendChild(thead_d_sqrt_inv);

    let tbody_d_sqrt_inv = document.createElement("tbody");
    for (let i = 0; i < D_sqrt_inv.length; i++) {
        let row_d_sqrt_inv = document.createElement("tr");
        let th_row_header_d_sqrt_inv = document.createElement("th");
        th_row_header_d_sqrt_inv.innerHTML = i;
        row_d_sqrt_inv.appendChild(th_row_header_d_sqrt_inv);

        for (let j = 0; j < D_sqrt_inv[i].length; j++) {
            let td_d_sqrt_inv = document.createElement("td");
            td_d_sqrt_inv.innerHTML = D_sqrt_inv[i][j];
            row_d_sqrt_inv.appendChild(td_d_sqrt_inv);
        }
        tbody_d_sqrt_inv.appendChild(row_d_sqrt_inv);
    }
    table_d_sqrt_inv.appendChild(tbody_d_sqrt_inv);

    d_sqrt_inv.appendChild(table_d_sqrt_inv);

    // Crear tabla para S
    let table_s = document.createElement("table");
    table_s.id = "sTable";
    table_s.className = "display";

    let thead_s = document.createElement("thead");
    let headerRow_s = document.createElement("tr");
    let th_empty_s = document.createElement("th");
    th_empty_s.innerHTML = "";
    headerRow_s.appendChild(th_empty_s);
    for (let i = 0; i < S.length; i++) {
        let th_s = document.createElement("th");
        th_s.innerHTML = i;
        headerRow_s.appendChild(th_s);
    }
    thead_s.appendChild(headerRow_s);
    table_s.appendChild(thead_s);

    let tbody_s = document.createElement("tbody");
    for (let i = 0; i < S.length; i++) {
        let row_s = document.createElement("tr");
        let th_row_header_s = document.createElement("th");
        th_row_header_s.innerHTML = i;
        row_s.appendChild(th_row_header_s);

        for (let j = 0; j < S[i].length; j++) {
            let td_s = document.createElement("td");
            td_s.innerHTML = S[i][j];
            row_s.appendChild(td_s);
        }
        tbody_s.appendChild(row_s);
    }
    table_s.appendChild(tbody_s);

    s.appendChild(table_s);

    $(document).ready(function () {
        $('#dTable').DataTable({
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

        $('#d_sqrt_invTable').DataTable({
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

        $('#sTable').DataTable({
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
            $('#dTable').DataTable().columns.adjust();
            $('#d_sqrt_invTable').DataTable().columns.adjust();
            $('#sTable').DataTable().columns.adjust();
        })

    });


}

function updateNodesGradient(nodeGroup, colorScale, F, pred) {
    nodeGroup.selectAll("circle")
        .each(function (d) {
            d.label = pred[d.index];
        })
        .attr("fill", d => {
            if (d.label === -1) {
                return "#808080";
            } else {
                let baseColor = d3.color(colorScale(d.label));

                return d3.interpolateRgb("white", baseColor)(F[d.index][pred[d.index]] === 0 ? 0.1 : F[d.index][pred[d.index]]);
            }
        });
}

export function plotIteration(stepInfo, nodeGroup, linkGroup, colorScale, F_history, pred_history, mapping) {
    stepInfo.innerHTML = `
            <div class="d-flex">
                <input type="range" class="form-range mx-2" id="controlIteration" min="0" max=${F_history.length - 1} value="0">
                <span id="controlIterationLabel" class="badge bg-primary px-3 mx-auto">0</span>
            </div>
        `;

    updateNodesGradient(nodeGroup, colorScale, F_history[0], pred_history[0]);
    document.getElementById('controlIteration').addEventListener('input', function (event) {
        let value = event.target.value;
        document.getElementById('controlIterationLabel').innerText = value;

        updateNodesGradient(nodeGroup, colorScale, F_history[value], pred_history[value]);

    });

    let num_nodes = pred_history[0].length

    let table = document.createElement("table");
    table.id = "inferenceTable";
    table.className = "display";

    let thead = document.createElement("thead");
    let headerRow = document.createElement("tr");
    let th = document.createElement("th");
    th.innerHTML = "";
    headerRow.appendChild(th);
    for (let i = 0; i < F_history.length; i++) {
        let th = document.createElement("th");
        th.innerHTML = "Iteración: " + i.toString();
        headerRow.appendChild(th);
    }
    thead.appendChild(headerRow);
    table.appendChild(thead);

    let tbody = document.createElement("tbody");
    for (let i = 0; i < num_nodes; i++) {
        let row = document.createElement("tr");
        row.classList.add("user-select-none");
        let th = document.createElement("th");
        th.innerHTML = i;
        row.appendChild(th);
        for (let j = 0; j < F_history.length; j++) {
            let td = document.createElement("td");
            let color = pred_history[j][i] === -1 ? '#999' : colorScale(pred_history[j][i]);
            let label = pred_history[j][i] === -1 ? 'Sin etiqueta' : mapping[pred_history[j][i]];

            td.innerHTML = `<span style="color: ${color};">${label}</span><br>` +
                (pred_history[j][i] === -1 ? 1 : F_history[j][i][pred_history[j][i]]).toString();

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
        $('#inferenceTable').DataTable({
            responsive: true,
            scrollX: true,
            scrollY: '600px',
            scrollCollapse: true,
            paging: false,
            searching: false,
            ordering: false,
            scroller: true
        });
    });
}

export function plotFinal(stepInfo, nodeGroup, linkGroup, F, pred, mapping, colorScale) {
    let rowDiv = document.createElement('div');
    rowDiv.classList.add('d-flex', 'flex-column', 'flex-md-row', 'justify-content-around');

    let col1Div = document.createElement('div');
    col1Div.classList.add('d-flex', 'flex-column', 'justify-content-center', 'align-items-center', 'p-1');

    let col2Div = document.createElement('div');
    col2Div.classList.add('d-flex', 'flex-column', 'justify-content-center', 'align-items-center', 'p-1');
    let arrow = document.createElement('div');
    arrow.innerHTML = '→';
    col2Div.appendChild(arrow);

    let col3Div = document.createElement('div');
    col3Div.classList.add('d-flex', 'flex-column', 'justify-content-center', 'align-items-center', 'p-1');

    rowDiv.appendChild(col1Div);
    rowDiv.appendChild(col2Div);
    rowDiv.appendChild(col3Div);

    stepInfo.appendChild(rowDiv);

    let table_labels = document.createElement("table");
    table_labels.id = "finallabelsTable";
    table_labels.className = "display";

    let thead_labels = document.createElement("thead");
    let headerRow_labels = document.createElement("tr");
    let th_empty = document.createElement("th");
    th_empty.innerHTML = "";
    headerRow_labels.appendChild(th_empty);

    for (let i = 0; i < F[0].length; i++) {
        let th_labels = document.createElement("th");
        th_labels.innerHTML = mapping[i];
        headerRow_labels.appendChild(th_labels);
    }

    thead_labels.appendChild(headerRow_labels);
    table_labels.appendChild(thead_labels);

    let tbody_labels = document.createElement("tbody");
    for (let i = 0; i < F.length; i++) {
        let row_labels = document.createElement("tr");
        let th_row_header = document.createElement("th");
        th_row_header.innerHTML = i;
        row_labels.appendChild(th_row_header);

        for (let j = 0; j < F[i].length; j++) {
            let td_labels = document.createElement("td");
            td_labels.innerHTML = F[i][j];
            row_labels.appendChild(td_labels);
        }
        row_labels.addEventListener('click', function () {
            resetHighlight(nodeGroup, linkGroup);
            highlightNodes(nodeGroup, [i]);
        });

        tbody_labels.appendChild(row_labels);
    }

    table_labels.appendChild(tbody_labels);
    col1Div.appendChild(table_labels);

    $(document).ready(function () {
        $('#finallabelsTable').DataTable({
            responsive: true,
            scrollX: true,
            scrollY: '600px',
            scrollCollapse: true,
            paging: false,
            searching: false,
            ordering: false,
            columnDefs: [
                {className: "dt-head-center", targets: '_all'}
            ],
            initComplete: function () {
                document.getElementById("loader").remove();
            }
        });

        $('#finallabelsTable').DataTable().columns.adjust();

        col1Div.querySelector(".dt-scroll-headInner").style.removeProperty('width');
        col1Div.querySelector(".dt-scroll-headInner").classList.add("m-auto");
    });

    let tableContainer = document.createElement("div");
    tableContainer.style.maxHeight = "600px";
    tableContainer.style.width = "150px";
    tableContainer.style.overflowY = "auto";

    let table_pred = document.createElement("table");
    table_pred.id = "finalPredTable";
    table_pred.className = "table table-striped";

    let thead_pred = document.createElement("thead");
    thead_pred.className = "thead-dark";
    thead_pred.style.backgroundColor = "transparent";
    let headerRow_pred = document.createElement("tr");

    let th_empty_pred = document.createElement("th");
    th_empty_pred.scope = "col";
    th_empty_pred.innerHTML = " ";
    headerRow_pred.appendChild(th_empty_pred);

    let th_pred = document.createElement("th");
    th_pred.scope = "col";
    th_pred.innerHTML = "Etiqueta";
    headerRow_pred.appendChild(th_pred);

    thead_pred.appendChild(headerRow_pred);
    table_pred.appendChild(thead_pred);

    let tbody_pred = document.createElement("tbody");
    tbody_pred.style.backgroundColor = "transparent";

    for (let i = 0; i < pred.length; i++) {
        let row_pred = document.createElement("tr");

        let th_row_header = document.createElement("th");
        th_row_header.scope = "row";
        th_row_header.innerHTML = i;
        row_pred.appendChild(th_row_header);

        let td_pred = document.createElement("td");
        let color = pred[i] === -1 ? '#999' : colorScale(pred[i]);
        let label = pred[i] === -1 ? 'Sin etiqueta' : mapping[pred[i]];

        td_pred.innerHTML = `<span style="color: ${color};">${label}</span>`;
        row_pred.appendChild(td_pred);

        row_pred.addEventListener('click', function () {
            resetHighlight(nodeGroup, linkGroup);
            highlightNodes(nodeGroup, [i]);
        });

        tbody_pred.appendChild(row_pred);
    }

    table_pred.appendChild(tbody_pred);
    tableContainer.appendChild(table_pred);
    col3Div.appendChild(tableContainer);
}
