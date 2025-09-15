var coll = document.getElementsByClassName("collapsible");
var i;
var collectionID = -1;

// Setup tooltips

$.noConflict();
jQuery( document ).ready(function () {
    jQuery('#modelName').tooltip({'trigger':'hover', 'title': 'Enter Name for shepard data object where training results will be stored', placement: 'right'});
    jQuery('#AUTOENCODER_TYPE').tooltip({'trigger':'hover', 'title': 'Select the model you want to train', placement: 'right'});
    jQuery('#BATCH_SIZE').tooltip({'trigger':'hover', 'title': 'Mini Batch Size used to fit the model, e.g Using a BUFFER_SIZE = 32 and a BATCH_SIZE = 8 means the buffer will be fitted in chunks of 8 samples.', placement: 'right'});
    jQuery('#EPOCHS').tooltip({'trigger':'hover', 'title': 'Set the number of Epochs (iterations) for the training cycle of one data batch.', placement: 'right'});
    jQuery('#BUFFER_SIZE').tooltip({'trigger':'hover', 'title': 'Number of samples to be buffered and passed through the model at once, e.g. BUFFER SIZE = 1 means only one sample will be collected to fit the model. In case of a Recurrent Autoencoder the buffer size represents the number of sequences that are collected and trained at once. The length of this sequences can be set with the parameter SEQUENCE LENGTH', placement: 'right'});
    jQuery('#BUFFER_MODE').tooltip({'trigger':'hover', 'title': 'Sets the mechanism how the Buffer will be updated. Replace', placement: 'right'});
    jQuery('#DROPOUT_RATE').tooltip({'trigger':'hover', 'title': 'Dropout Rate is the fraction of the input units to drop randomly in each layer of the autoencoder to prevent overfitting.', placement: 'right'});
    jQuery('#L2_LAMBDA').tooltip({'trigger':'hover', 'title': 'L2 Regularization Lambda value to prevent overfitting.', placement: 'right'});
    jQuery('#SEQUENCE_LENGTH').tooltip({'trigger':'hover', 'title': 'Set the length of one time series to be used for the training in case of a Recurrent Autoencoder.', placement: 'right'});
});


window.onload = function() {
    readCollections();
}

function queryCollectionData() {
    // URL des Endpunkts
    collectionID = document.getElementById('collectionID').value;
    var url = SHEPARD_BACKEND + '/collections/' + collectionID;
    var headers = new Headers();
    headers.append('X-API-KEY', API_KEY);

    // Daten abrufen
    fetch(url, {
        method: 'GET',
        headers: headers
        })
        .then(response => response.json())
        .then(data => {
            document.getElementById('structuredContainerId').value = data.attributes.__structuredContainerId__;
            document.getElementById('fileContainerId').value = data.attributes.__fileContainerId__;
            document.getElementById('timeseriesContainerId').value = data.attributes.__timeseriesContainerId__;
        })
    .catch(error => console.error('Error:', error));
}

function onOPCUAInput()  {
    var val = document.getElementById("selectValue").value;
    var opts = document.getElementById('valuesPath').children;
    for (var i = 0; i < opts.length; i++) {
      if (opts[i].value === val) {
        // Set the selected OPC path based on the selected option
        document.getElementById('OPCpath').value = opts[i].text;
        console.log(document.getElementById('OPCpath').value);
        break;
      }
    }
}

// Async function to browse OPC UA variables and handle server communication
async function browseVariablesOPCUA(rootNode) {
    let opcURI = document.getElementById('opcUAServer').value;
    let port = document.getElementById('opcUAPort').value;
    let username = document.getElementById('user_OPCUA').value;
    let password = document.getElementById('password_OPCUA').value;
    let authMethod = document.getElementById('securityMode').value;
    document.getElementById('loadingGif').style.removeProperty('display');

    // Fetch security and server data based on authentication method
    if (authMethod == 'User/Password') {
        let paramsUser = new URLSearchParams({
            userName: username,
            passwort: password,
            scenarioID: '8ded8336-d37b-4443-a8af-5ef0cae656a8',
            CredentialID: '671'
        });
        fetch(ORCHESTRA_WEBSERVICE + "/security/user?" + paramsUser.toString(), {
            method: 'POST'
        }).catch(error => {
            console.error('Fehler beim Abrufen der Daten:', error);
        });

        let paramsServer = new URLSearchParams({
            IP: opcURI,
            port: port,
            authMethod: 'UserPassword'
        });
        fetch(ORCHESTRA_WEBSERVICE + "/server?" + paramsServer.toString(), {
            method: 'POST'
        }).catch(error => {
            console.error('Fehler beim Abrufen der Daten:', error);
        });
    } else {
        // let paramsServer = new URLSearchParams({
        //     IP: opcURI,
        //     port: port,
        //     authMethod: 'Anonymous'
        // });
        // fetch("http://localhost:8019/opcua/server?" + paramsServer.toString(), {
        //     method: 'POST'
        // }).catch(error => {
        //     console.error('Fehler beim Abrufen der Daten:', error);
        // });
    }

    // Fetch OPC UA variable nodes and process them
    try {
        let paramsQuery = new URLSearchParams({
            rootNode: rootNode.value
        });
        let response = await fetch(ORCHESTRA_WEBSERVICE + "/variables/root?" + paramsQuery.toString(), {
            method: 'GET'
        });
        let data = await response.json();
        addVarToTable(data);
        document.getElementById('loadingGif').style.display = 'none';
    } catch (Exception) {
        console.error('Fehler beim Abrufen der Daten:', Exception);
    }
    fillDatalist(); // Call the function to prefill variables
}

// Function to process and populate the variable nodes into a dropdown
function processVariableNodes(json) {
    var variableNodes = json.variableNode;
    var datalist = document.getElementById('valuesPath');
    // Populate the dropdown with variable nodes
    for (var i = 0; i < variableNodes.length; i++) {
        var variableNode = variableNodes[i];
        var option = document.createElement('option');
        option.value = variableNode.nodeId;
        option.text = variableNode.path;
        datalist.appendChild(option);
    }
}

function countNodeIds() {
    var nodeIds = document.getElementById('nodeIds').value;
    var count = nodeIds.split(',,,').length;
    document.getElementById('nodeIdsCount').value = count;
}

function addVarToTable(json) {
    document.getElementById('selectedVariables').style.removeProperty('display');
    var table = document.getElementById('variableBody');
    clearTable(table);
    for (var i = 0; i < json.variableNode.length; i++) {
        insertVariableRow(i, json.variableNode[i].nodename, json.variableNode[i].nodeId, json.variableNode[i].type, table);
    }
}

function insertVariableRow(index, nodename, nodeId, type, table) {
    var row = table.insertRow(-1);
    row.id = "variableRow_" + index;
    var cell1 = row.insertCell(0);
    cell1.id = "variableRow_NodeName_" + index;
    var cell2 = row.insertCell(1);
    var input = document.createElement('input');
    input.type = 'text';
    input.name = 'variableRow_nodeId_' + index;
    input.id = 'variableRow_nodeId_' + index;
    input.value = nodeId;
    input.style.width = ((input.value.length + 2) * 8) + 'px';
    input.style.border = 'none';
    input.readOnly = true;
    cell2.appendChild(input);
    var cell3 = row.insertCell(2);
    cell3.name = "variableRow_type_" + index;
    var cell4 = row.insertCell(3);
    cell1.innerHTML = nodename;
    cell3.innerHTML = type;
    cell4.innerHTML = '<input class="form-check-input" type="checkbox" name="variable_' + index + '" id="variable_' + index + '" checked>';
}

function uncheckAllVariables() {
    var checkboxes = document.querySelectorAll('input[id^="variable_"]');
    checkboxes.forEach(function(checkbox) {
        checkbox.checked = false;
    });
}
//
function deleteRow(button) {
    var row = button.parentNode.parentNode;
    row.parentNode.removeChild(row);
}

function countTableRows(table) {
    var rowCount = table.rows.length;
    console.log("Number of rows in the table: " + rowCount);
    return rowCount;
}

function filterTable() {
    var input, filter, table, tr, td, i, j, txtValue;
    input = document.getElementById("searchInput");
    filter = input.value.toUpperCase();
    table = document.getElementById("selectedVariables");
    tr = table.getElementsByTagName("tr");

    for (i = 1; i < tr.length; i++) {
        tr[i].style.display = "none";
        td = tr[i].getElementsByTagName("td");
        for (j = 0; j < td.length; j++) {
            if (td[j].id.indexOf("variableRow_NodeName_") == 0 ) {
                txtValue = td[j].textContent || td[j].innerText;
                if (txtValue.toUpperCase().indexOf(filter) > -1) {
                    tr[i].style.display = "";
                    break;
                }
            }
        }
    }
}

function applyFilteredCheckboxes(checkValue) {
    var input, filter, table, tr, td, i, j, txtValue;
    input = document.getElementById("searchInput");
    filter = input.value.toUpperCase();
    table = document.getElementById("selectedVariables");
    tr = table.getElementsByTagName("tr");

    for (i = 1; i < tr.length; i++) {
        tr[i].style.display = "none";
        td = tr[i].getElementsByTagName("td");
        for (j = 0; j < td.length; j++) {
            if (td[j].id.indexOf("variableRow_NodeName_") == 0 ) {
                txtValue = td[j].textContent || td[j].innerText;
                if (txtValue.toUpperCase().indexOf(filter) > -1) {
                    tr[i].style.display = "";
                    var checkbox = tr[i].querySelector('input[id^="variable_"]');
                    if (checkbox) {
                        checkbox.checked = checkValue;
                    }
                    break;
                }
            }
        }
    }
}

function modelTypeSelected(modelType) {
    if (modelType == 'lstm' || modelType == 'conv' || modelType == 'attention') {
        document.getElementById('rnnAutoencoderParams').style.removeProperty('display');
    } else {
        document.getElementById('rnnAutoencoderParams').style.display = 'none';
        document.getElementById('bufferModeRow').style.removeProperty('display');
    }
}

function addManually() {
    var nodeID = document.getElementById('nodeId_manually').value;
    var nodeName = document.getElementById('variableName_manually').value;
    var nodeType = document.getElementById('dataType_manually').value;
    var table = document.getElementById('variableBody');
    var rowCount = countTableRows(table);
    insertVariableRow(rowCount, nodeName, nodeID, nodeType, table);
    document.getElementById('nodeId_manually').value = "";
    document.getElementById('variableName_manually').value = "";
    document.getElementById('dataType_manually').value = "";
    fillDatalist(); // Call the function to prefill variables
}

function uploadCSV() {
    var file = document.getElementById('csvFile').files[0];
    var reader = new FileReader();
    reader.onload = function(e) {
        var text = reader.result;
        var lines = text.split('\n');
        var table = document.getElementById('variableBody');
        clearTable(table);
        for (var i = 0; i < lines.length; i++) {
            var line = parseCSVLine(lines[i]);
            if (line.length == 3) {
                insertVariableRow(i, line[0], line[1], line[2], table);
            }
        }
    };
    reader.readAsText(file);
    fillDatalist(); // Call the function to prefill variables
}

function parseCSVLine(line) {
    var result = [];
    var insideQuotes = false;
    var value = '';
    for (var i = 0; i < line.length; i++) {
        var char = line[i];
        if (char === '"') {
            insideQuotes = !insideQuotes;
        } else if (char === ';' && !insideQuotes) {
            result.push(value);
            value = '';
        } else {
            value += char;
        }
    }
    result.push(value);
    return result;
}

function clearTable(table) {
    table.innerHTML = "";
}

function fillDatalist() {
    let dataType = document.getElementById('dataSourceType').value;
    let headers = null;
    if (dataType == 'CSV') {
        let list = document.getElementById('headers').value.split(',,,');

        headers = list.map(function (element) {
            if (element.length > 0) {
                let cleaned = element.split('[');
                return cleaned[0];
            }
        });
    } else if (dataType == 'OPC UA') {
        headers = document.querySelectorAll('[id^="variableRow_nodeId_"]');
    } else {
        console.error('Invalid data type selected');
        return;
    }
    var datalist = document.getElementById('variableOptions');
    datalist.innerHTML = ''; // Clear existing options
    for (var i = 0; i < headers.length-1; i++) {
        var option = document.createElement('option');
        if (dataType == 'CSV') {
            // option.innerHTML = headers[i];
            option.value = headers[i];
        } else if (dataType == 'OPC UA') {
            option.value = headers[i].value;
        } else {
            console.error('Invalid data type selected');
            return;
        }
        datalist.appendChild(option);
    }
}