var allScenarios;

window.onload = function() {
    // REST-Endpunkt URL
    let endpoint = ORCHESTRA_WEBSERVICE + '/getScenarios';
    callOrchestraWebservice(endpoint)
    .then(response => response.json())
    .then(data => {
        allScenarios = data;
        console.log("Scenarios loaded:", allScenarios);
    })
    .catch(error => {
        console.error('Fehler beim Abrufen der Daten:', error);
    });
}

function filterScenarios(group) {
    document.getElementById("dataObjects").style.removeProperty("display");
    let dropdown = document.getElementById('modelSelection');
    dropdown.innerHTML = ''; // Clear existing options
    let defaultOption = document.createElement('option');
    defaultOption.value = '';
    defaultOption.textContent = 'Select a scenario';
    dropdown.appendChild(defaultOption);
    for (let i = 0; i<allScenarios.length; i++) {
        let scenario = allScenarios[i];
        if (scenario.groups.some(e => e.groupName == group)) {
            let option = document.createElement('option');
            option.value = scenario.scenarioID;
            option.textContent = scenario.name;
            dropdown.appendChild(option);
        }
    }
}

function fillScenarioData(scenarioId) {
    let scenario = allScenarios.find(s => s.scenarioID === scenarioId);
    document.getElementById('processName').value = scenario.name.substring(3);
    document.getElementById('scenarioID').value = scenario.scenarioID;
    document.getElementById('created').value = scenario.deployedAt;
}

function callOrchestraWebservice(endpoint, method = 'GET', body = null) {
    const headers = new Headers();
    if (body) {
        headers.append('Content-Type', 'application/json');
    }
    return fetch(endpoint, {
        method: method,
        headers: headers,
        body: body ? JSON.stringify(body) : null
    });
}