const API_KEY = "eyJhbGciOiJSUzI1NiJ9.eyJzdWIiOiJhMmViYTIwMi0zMzhmLTQ0ODUtOTU2Ny1lNWE4MDkyMWRmMmIiLCJpc3MiOiJodHRwOi8vYmFja2VuZC5zaGVwYXJkLmRldi5kcy1sYWIub3JnL3NoZXBhcmQvYXBpLyIsIm5iZiI6MTc1Mzc5OTYwMSwiaWF0IjoxNzUzNzk5NjAxLCJqdGkiOiJiYjZjM2Y4Ni1jMDdmLTQ1ODMtYTNiYS1kZmU0NWUyZGYwODkifQ.NStqMAl3tss8CH5_qK-OjTtoPrfN8D9D_ckn5FLUKf6JUPkxZuW5z6CYuB_dZZyX4Wvd222S9PFpVc7btsb2Nhy46GeBur8-L2bzvVy9IqLwZ1ATQ3-MT6h8e9pJJtukLy073qpd76qTgHQDzt9DFrOwzBQCEUxYOacksWxfs40xUyH2yVy1DjiOTEDYvPFlscLAo_w8t3Jfmb8WUZ4zsmjmNdXJ45KhoWoW_lrdwny8azq4VOymh7KHJ2kNXIh3CYsZwJiyfCToz_SQsA_zCfImORvwNZdR_h2_obNOljybeBBalIwjX9F2vRyyx3Y59yYE9ts9DH4lr6Syst9Dfw";
const SHEPARD_BACKEND = "https://backend.shepard.dev.ds-lab.org/shepard/api";
const ORCHESTRA_WEBSERVICE = "http://localhost:8019/opcua";


function queryCollectionData() {
    // URL des Endpunkts
    let collectionId = document.getElementById('collectionID').value;
    var url = SHEPARD_BACKEND + '/collections/' + collectionId;
    var headers = new Headers();
    headers.append('X-API-KEY', API_KEY);

    // Daten abrufen
    fetch(url, {
        method: 'GET',
        headers: headers
        })
        .then(response => response.json())
        .then(data => {
            var endpoint = SHEPARD_BACKEND + '/search';
            // Header with the API key
            var headers = new Headers();
            headers.append('X-API-KEY', API_KEY);
            headers.append('Content-Type', 'application/json');
            const query = {
                "searchParams": {
                    "query": "{ \"property\": \"attributes.__class__\", \"operator\": \"eq\", \"value\": \"model\"}",
                    "queryType": "DataObject"
                },
                "scopes": [{
                    "collectionId": collectionId,
                    "traversalRules": ["children", "parents", "predecessors", "successors"]
                }]
            };
            data = searchShepard(query)
            data.then(data => {
                var dropdown = document.getElementById('modelSelection');
                dropdown.innerHTML = '';
                var option = document.createElement('option');
                option.value = -1;
                option.text = 'No Model selected';
                dropdown.appendChild(option);
                for (var i = 0; i < data.results.length; i++) {
                    getModelVersions(data.results[i]);
                }
            });
        })
    .catch(error => console.error('Error:', error));
    document.getElementById('dataObjects').style.removeProperty('display');
}

function searchShepard(query) {
    var endpoint = SHEPARD_BACKEND + '/search';
    // Header with the API key
    var headers = new Headers();
    headers.append('X-API-KEY', API_KEY);
    headers.append('Content-Type', 'application/json');
    // Send POST request and return the JSON response
    return fetch(endpoint, {
        method: 'POST',
        headers: headers,
        body: JSON.stringify(query)
    })
    .then(response => response.json())
    .catch(error => {
        console.error('Error fetching data:', error);
        return null;
    });
}

function getModelVersions(model) {
    let collectionId = document.getElementById('collectionID').value;
    let query = {
        "searchParams": {
            "query": "{ \"property\": \"name\", \"operator\": \"contains\", \"value\": \"autoencoder_configs\"}",
            "queryType": "Reference"
        },
        "scopes": [{
            "collectionId": collectionId,
            "dataObjectId": model.id,
            "traversalRules": ["children"]
        }]
    };
    data = searchShepard(query);
    data.then(data => {
        //use first refrence as there should only be one
        reference = data.results[0];
        if (data.results.length > 1) {
            console.warn('More than one model version found, this should not happen');
            console.log(data.results);
        }
        var endpoint = SHEPARD_BACKEND + '/collections/' + collectionId + '/dataObjects/' + model.id + '/structuredDataReferences/' + reference.id;
        // Header with the API key
        var headers = new Headers();
        headers.append('X-API-KEY', API_KEY);
        // Send GET request
        fetch(endpoint, {
            method: 'GET',
            headers: headers,
        })
        .then(response => response.json())
        .then(data => {
            var dropdown = document.getElementById('modelSelection');
            for (let i = 0; i < data.structuredDataOids.length; i++) {
                var endpoint = SHEPARD_BACKEND + '/structuredDataContainers/' + data.structuredDataContainerId + '/payload/' + data.structuredDataOids[i];
                // Header with the API key
                var headers = new Headers();
                headers.append('X-API-KEY', API_KEY);
                // Send GET request
                fetch(endpoint, {
                    method: 'GET',
                    headers: headers,
                }).then(response => response.json())
                .then(payload => {
                    var option = document.createElement('option');
                    option.value = data.structuredDataContainerId + ";" + payload.structuredData.oid + ";" + i + ";" + model.id;
                    option.text = model.name + " (" + payload.structuredData.name + ")";
                    dropdown.appendChild(option);
                })
                .catch(error => {
                    console.error('Error fetching data:', error);
                });
            }
        })
        .catch(error => {
            console.error('Error fetching data:', error);
        });
    });
}

function dataSourceSelected(dataSourceType) {
    // Get the selected data source and enable the input fields
    // document.getElementById('sourceName').removeAttribute("disabled");
    // document.getElementById('sourceDescription').removeAttribute("disabled");
    if (dataSourceType == 'OPC UA') {

        // Enable the OPC UA input fields and disable the REST input fields
        document.getElementById('OPCUA').style.removeProperty("display");
        document.getElementById('timerConfigDiv').style.removeProperty("display");
        document.getElementById('TrainingEnvironment').style.removeProperty("display");
        document.getElementById('CSV').style.display = "none";
        // document.getElementById('REST_values').style.display = "none";
        document.getElementById('handleInvalidData').style.removeProperty("display");

    } else if(dataSourceType == 'CSV') {

        // Enable the REST input fields and disable the OPC UA input fields
        // document.getElementById('REST_values').style.removeProperty("display");
        document.getElementById('OPCUA').style.display = "none";
        document.getElementById('timerConfigDiv').style.display = "none";
        document.getElementById('TrainingEnvironment').style.removeProperty("display");
        document.getElementById('CSV').style.removeProperty("display");
        document.getElementById('handleInvalidData').style.removeProperty("display");

    } else {
        // Disable all input fields
        document.getElementById('OPCUA').style.display = "none";
        document.getElementById('timerConfigDiv').style.display = "none";
        // document.getElementById('REST_values').style.display = "none";
        document.getElementById('TrainingEnvironment').style.display = "none";
        document.getElementById('CSV').style.display = "none";
        document.getElementById('handleInvalidData').style.display = "none";
    }
}

function updateSelectedModel(dropdown) {
    let vals = dropdown.value.split(";");
    let structuredDataContainerId = vals[0];
    let modelOid = vals[1];
    let name = dropdown.options[dropdown.selectedIndex].text;
    let versionIndex = vals[2];
    let dataObjectId = vals[3];
    let collectionId = document.getElementById('collectionID').value;

    // in Case of Detection execute following code
    try {
        let version_increment = parseInt(versionIndex) + 1;
        document.getElementById('modelVersion').value = "v" + version_increment;
        read_validation_loss(dataObjectId, versionIndex);
        document.getElementById('structDataModelOid').value = modelOid;
    } catch (error) {
        console.error('Error updating model version:', error);
    }
    if (dataObjectId == -1) {
        document.getElementById('OPCUA').style.display = "none";
        document.getElementById('timerConfigDiv').style.display = "none";
        document.getElementById('TrainingEnvironment').style.display = "none";
        document.getElementById('rnnAutoencoderParams').style.display = 'none';
        document.getElementById('dataObjectName').value = "";
        document.getElementById('dataObjectID').value = "";
        document.getElementById('created').value = "";
        document.getElementById('modelType').value = "";
        document.getElementById('validationPerformance').value = "";
    } else {
        var endpoint = SHEPARD_BACKEND + '/structuredDataContainers/' + structuredDataContainerId + '/payload/' + modelOid;
        // Header with the API key
        var headers = new Headers();
        headers.append('X-API-KEY', API_KEY);
        // Send GET request
        fetch(endpoint, {
            method: 'GET',
            headers: headers,
        })
        .then(response => response.json())
        .then(data => {
            // Create dropdown menu
            let payload = JSON.parse(data.payload);
            document.getElementById('dataObjectName').value = name;
            document.getElementById('created').value = data.structuredData.createdAt;
            document.getElementById('modelType').value = payload.model_type;
            if (payload.model_type == 'lstm' || payload.model_type == 'conv' || payload.model_type == 'attention') {
                document.getElementById('rnnAutoencoderParams').style.removeProperty('display');
            } else {
                document.getElementById('rnnAutoencoderParams').style.display = 'none';
            }
            document.getElementById('validationPerformance').value = payload.test_loss;
            try {
                document.getElementById('recommendedDetectionThreshold').value = payload.recommended_threshold_mse[0];
            } catch (error) {
                console.log('No recommended detection threshold found');
            }
        });
        let query = {
            "searchParams": {
                "query": "{ \"property\": \"name\", \"operator\": \"contains\", \"value\": \"training_config\"}",
                "queryType": "Reference"
            },
            "scopes": [{
                "collectionId": collectionId,
                "dataObjectId": dataObjectId,
                "traversalRules": ["children"]
            }]
        };
        let results = searchShepard(query);
        results.then(data => {
            let result = data.results[0];
            if (data.results.length > 1) {
                console.warn('More than one model version found, this should not happen');
                console.log(data.results);
            }
            var endpoint = SHEPARD_BACKEND + '/collections/' + collectionId + '/dataObjects/' + dataObjectId + '/structuredDataReferences/' + result.id;
            // Header with the API key
            var headers = new Headers();
            headers.append('X-API-KEY', API_KEY);
            // Send GET request
            fetch(endpoint, {
                method: 'GET',
                headers: headers,
            })
            .then(response => response.json())
            .then(data => {
                //get Oid from at model index
                let config = data.structuredDataOids[versionIndex];
                var endpoint = SHEPARD_BACKEND + '/structuredDataContainers/' + data.structuredDataContainerId + '/payload/' + config;
                fetch(endpoint, {
                    method: 'GET',
                    headers: headers,
                }).then(response => response.json())
                .then(payload => {
                    let json = JSON.parse(payload.payload);
                    document.getElementById('BUFFER_SIZE').value = json.BUFFER_LENGTH;
                    document.getElementById('BATCH_SIZE').value = json.BATCH_SIZE;
                    document.getElementById('EPOCHS').value = json.EPOCHS;
                    document.getElementById('BUFFER_MODE').value = json.BUFFER_MODE;
                    try {
                        document.getElementById('SEQUENCE_LENGTH').value = json.SEQUENCE_LENGTH;
                    } catch (error) {
                        console.log('No sequence length found');
                    }
                    if (dataSourceType == 'OPC UA') {
                        document.getElementById('nodeIds').value = json.nodeIds.replaceAll(",,,", "\n");
                        document.getElementById('nodeIdsCount').value = json.numFeatures;
                        let serverAndPort = payjsonload.OpcUaServer.replace("opc.tcp://", "").split(":");
                        document.getElementById('opcUAServer').value = serverAndPort[0];
                        document.getElementById('opcUAPort').value = serverAndPort[1];
                    }
                });
            });
        })
        .catch(error => {
            console.error('Error fetching data:', error);
        });
        endpoint = SHEPARD_BACKEND + '/collections/' + collectionId + '/dataObjects/' + dataObjectId;
        document.getElementById('dataObjectID').value = dataObjectId;
        fetch(endpoint, {
            method: 'GET',
            headers: headers,
        }).then(response => response.json())
        .then(data => {
            let dataSourceType = data.attributes.__dataSourceType__;
            document.getElementById('dataSourceType').value = dataSourceType;
            dataSourceSelected(dataSourceType);
        }).catch(error => {
            console.error('Error fetching data:', error);
        });
    }
}

function updateAuth() {
    if (document.getElementById('securityMode').value == 'User/Password') {
        document.getElementById('user_OPCUA').removeAttribute("disabled");
        document.getElementById('password_OPCUA').removeAttribute("disabled");
    } else {
        document.getElementById('user_OPCUA').setAttribute("disabled", "disabled");
        document.getElementById('password_OPCUA').setAttribute("disabled", "disabled");
    }
}

function readCollections() {
    console.log("Reading collections");
    // REST-Endpunkt URL
    var endpoint = SHEPARD_BACKEND + '/collections';
    
    // Header mit dem API-Schlüssel
    var headers = new Headers();
    headers.append('X-API-KEY', API_KEY);

    // GET-Request senden
    fetch(endpoint, {
        method: 'GET',
        headers: headers
    })
    .then(response => response.json())
    .then(data => {
        // Dropdown-Menü erstellen
        var dropdown = document.getElementById('collectionID');
        var option = document.createElement('option');
        option.value = -1;
        option.text = 'No Collection selected';
        dropdown.appendChild(option);
        for (var i = 0; i < data.length; i++) {
                var item = data[i];
                var option = document.createElement('option');
                option.value = item.id;
                option.text = item.name;
                dropdown.appendChild(option);
        }
    })
    .catch(error => {
        console.error('Fehler beim Abrufen der Daten:', error);
    });
}