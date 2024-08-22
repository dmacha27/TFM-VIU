import {fetchGBILI, fetchRGCLI, drawGBILI, drawRGCLI} from './draw.js';
import {getElementsInDFS} from "./common.js";

export let pseudocode_elements_gbili;
export let pseudocode_elements_rgcli;
export let pseudocode_elements_lgc;

let archivo;
let target_name;

document.addEventListener('DOMContentLoaded', function () {
    ['dragleave', 'drop', 'dragenter', 'dragover'].forEach(function (evento) {
        document.addEventListener(evento, function (e) {
            e.preventDefault();
        }, false);
    });

    let area = document.getElementById('soltar');
    area.param = 'soltar';

    let progreso = document.getElementById('progreso');
    let porcentaje_progreso = document.getElementById('porcentaje_progreso');
    let nombre_fichero = document.getElementById('nombre_fichero');
    let classSelector = document.getElementById('classSelector');
    let selectGraphBtn = document.getElementById("selectGraphBtn");
    selectGraphBtn.disabled = true;

    let boton = document.getElementById('archivo');
    boton.param = 'boton';

    area.addEventListener('drop', subir, false)
    boton.addEventListener('change', subir)

    function subir(e) {
        e.preventDefault();

        progreso.style.width = "0" + "%";
        progreso.setAttribute("aria-valuenow", "0");
        porcentaje_progreso.textContent = "0" + "%";

        if (e.currentTarget.param === 'soltar') {
            archivo = e.dataTransfer.files; // arrastrar y soltar
        } else {
            archivo = e.target.files; // botón de subida
        }
        if (archivo.length >= 2) {
            return false;
        }

        nombre_fichero.textContent = archivo[0].name;

        enviarArchivo(archivo[0]);
    }

    function enviarArchivo(file) {
        let xhr = new XMLHttpRequest();
        xhr.open('post', '/upload', true);

        xhr.onreadystatechange = function () {
            if (xhr.readyState === 4) {
                if (xhr.status === 200) {
                    try {
                        let response = JSON.parse(xhr.responseText);

                        classSelector.innerHTML = "";

                        let default_option = document.createElement('option');
                        default_option.value = "";
                        classSelector.appendChild(default_option);

                        response.forEach(function (nombre) {
                            let option = document.createElement('option');
                            option.value = nombre;
                            option.textContent = nombre;
                            classSelector.appendChild(option);
                        });

                        document.getElementById("card_subida_class").style.display = "block";
                        selectGraphBtn.disabled = false;
                    } catch (error) {
                        console.error("Error parsing response:", error);
                    }
                } else {
                    console.error("Upload failed with status:", xhr.status);
                }
            }
        };

        xhr.upload.onprogress = function (evento) {
            if (evento.lengthComputable) {
                let porcentaje = Math.floor(evento.loaded / evento.total * 100);
                progreso.style.width = porcentaje + "%";
                progreso.setAttribute("aria-valuenow", porcentaje.toString());
                porcentaje_progreso.textContent = porcentaje + "%";
            }
        };

        let params = new FormData();
        params.append('file', file);
        xhr.send(params);
    }


    document.getElementById('fichero_prueba').addEventListener('click', () => descargarFichero('iris'));
    document.getElementById('fichero_iris').addEventListener('click', () => descargarFichero('iris'));
    document.getElementById('fichero_breast').addEventListener('click', () => descargarFichero('breast'));
    document.getElementById('fichero_diabetes').addEventListener('click', () => descargarFichero('diabetes'));

    function descargarFichero(nombre) {
        fetch('/descargar_fichero?nombre=' + nombre)
            .then(response => response.blob())
            .then(blob => {
                archivo = [new File([blob], nombre + ".arff", {type: ""}),];

                progreso.style.width = "100" + "%";
                progreso.setAttribute("aria-valuenow", "100");
                porcentaje_progreso.textContent = "100" + "%";

                enviarArchivo(archivo[0]);

                document.getElementById("selectGraphBtn").disabled = false;

                document.getElementById("nombre_fichero").textContent = nombre + ".arff";

            })
            .catch(error => console.error('Error al descargar el fichero:', error));

    }

    document.getElementById('selectGraphBtn').addEventListener('click', function () {
        document.getElementById('uploadDataset').style.setProperty('display', 'none', 'important');

        document.getElementById('graphButtons').style.display = 'block';
    });

    document.getElementById('gbiliBtn').addEventListener('click', function (event) {
        event.preventDefault();
        target_name = classSelector.value;

        if (target_name === "") {

            classSelector.setCustomValidity('Debe seleccionar una opción.');
            classSelector.reportValidity(); // Mostrar el mensaje de error
            return;

        } else {

            document.getElementById('graphButtons').style.setProperty('display', 'none', 'important');
            pseudocode.renderElement(document.getElementById("gbili_pseudocode"),
                {lineNumber: true});
            pseudocode_elements_gbili = getElementsInDFS(document.querySelector('#gbili_pseudocode_container .ps-root'));

            pseudocode_elements_rgcli = [];

            pseudocode.renderElement(document.getElementById("lgc_pseudocode"),
                {lineNumber: true});
            pseudocode_elements_lgc = getElementsInDFS(document.querySelector('#lgc_pseudocode_container .ps-root'));

            let p_unlabeled = document.getElementById('p_unlabeled').value;
            let alpha = document.getElementById('alpha').value;
            let iter_max = document.getElementById('iter_max').value;
            let threshold = document.getElementById('threshold').value;
            let k = document.getElementById('k').value;

            fetchGBILI(archivo[0], target_name, p_unlabeled, k, alpha, iter_max, threshold)
                .then(({steps, components_semi, components_graph, mapping}) => {
                    drawGBILI(steps, components_semi, components_graph, mapping);
                })
                .catch(error => {
                    console.error('Error:', error);
                });

            document.getElementById('contentContainer').style.display = 'block';
        }
    });

    document.getElementById('rgcliBtn').addEventListener('click', function (event) {
        event.preventDefault();
        target_name = classSelector.value;

        if (target_name === "") {

            classSelector.setCustomValidity('Debe seleccionar una opción.');
            classSelector.reportValidity(); // Mostrar el mensaje de error
            return;

        } else {

            document.getElementById('graphButtons').style.setProperty('display', 'none', 'important');
            pseudocode_elements_gbili = [];

            pseudocode.renderElement(document.getElementById("rgcli_pseudocode"), {lineNumber: true});
            pseudocode_elements_rgcli = getElementsInDFS(document.querySelector('#rgcli_pseudocode_container .ps-root'));

            pseudocode.renderElement(document.getElementById("lgc_pseudocode"),
                {lineNumber: true});
            pseudocode_elements_lgc = getElementsInDFS(document.querySelector('#lgc_pseudocode_container .ps-root'));

            let p_unlabeled = document.getElementById('p_unlabeled').value;
            let alpha = document.getElementById('alpha').value;
            let iter_max = document.getElementById('iter_max').value;
            let threshold = document.getElementById('threshold').value;
            let k_e = document.getElementById('k_e').value;
            let k_i = document.getElementById('k_i').value;
            let nt = document.getElementById('nt').value;

            fetchRGCLI(archivo[0], target_name, p_unlabeled, k_e, k_i, nt, alpha, iter_max, threshold)
                .then(({steps, mapping}) => {
                    drawRGCLI(steps, mapping);
                })
                .catch(error => {
                    console.error('Error:', error);
                });

            document.getElementById('contentContainer').style.display = 'block';

        }
    });

});
