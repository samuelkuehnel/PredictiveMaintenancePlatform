var CHART;

var xVals;
var yVals;


// Setup tooltips

$.noConflict();
jQuery( document ).ready(function () {
    jQuery('#modelName').tooltip({'trigger':'hover', 'title': 'Enter Name for shepard data object where detection results will be stored', placement: 'right'});
    jQuery('#DETECTION_THRESHOLD').tooltip({'trigger':'hover', 'title': 'Threshold for reconstruction error to classify anomalies (between 0 and 1)', placement: 'right'});
    jQuery('#CONTINUE_TRAINING').tooltip({'trigger':'hover', 'title': 'Select to train the model during detection phase. This helps adjusting the model to changing machine conditons.', placement: 'right'});
    jQuery('#BATCH_SIZE').tooltip({'trigger':'hover', 'title': 'Mini Batch Size used to fit the model, e.g Using a BUFFER_SIZE = 32 and a BATCH_SIZE = 8 means the buffer will be fitted in chunks of 8 samples.', placement: 'right'});
    jQuery('#EPOCHS').tooltip({'trigger':'hover', 'title': 'Set the number of Epochs (iterations) for the training cycle of one data batch.', placement: 'right'});
    jQuery('#BUFFER_SIZE').tooltip({'trigger':'hover', 'title': 'Number of samples to be buffered and passed through the model at once, e.g. BUFFER SIZE = 1 means only one sample will be collected to fit the model. In case of a Recurrent Autoencoder the buffer size represents the number of sequences that are collected and trained at once. The length of this sequences can be set with the parameter SEQUENCE LENGTH', placement: 'right'});
    jQuery('#BUFFER_MODE').tooltip({'trigger':'hover', 'title': 'Sets the mechanism how the Buffer will be updated. Replace', placement: 'right'});
    jQuery('#SEQUENCE_LENGTH').tooltip({'trigger':'hover', 'title': 'Set the length of one time series to be used for the training in case of a Recurrent Autoencoder.', placement: 'right'});
});

window.onload = function() {
    readCollections();
}

function updateSelectedModelData(dropdown) {
    updateSelectedModel(dropdown);
}

function updateRetrainingParams(checkbox) {
    if (checkbox.checked) {
        document.getElementById('retrainingParams').style.removeProperty('display');
    } else {
        document.getElementById('retrainingParams').style.display = 'none';
    }
}

function read_validation_loss(dataObjectId, index) {
     let collectionId = document.getElementById('collectionID').value;
    let query = {
        "searchParams": {
            "query": "{ \"property\": \"name\", \"operator\": \"contains\", \"value\": \"validation_loss\"}",
            "queryType": "Reference"
        },
        "scopes": [{
            "collectionId": collectionId,
            "dataObjectId": dataObjectId,
            "traversalRules": ["children"]
        }]
    };
    searchShepard(query).then(results => {
        let reference = results.results[0];
        console.log(query);
        console.log(results);
        if (results.results.length > 1) {
            console.warn('More than one model version found, this should not happen');
            console.log(results.results);
        }
                console.log(reference);

        var endpoint = SHEPARD_BACKEND + '/collections/' + collectionId + '/dataObjects/' + dataObjectId + '/timeseriesReferences/' + reference.id;
        headers = new Headers();
        headers.append('X-API-KEY', API_KEY);

        // Send GET request
        fetch(endpoint, {
            method: 'GET',
            headers: headers,
        })
        .then(response => response.json())
        .then(ref => {
            let timeseries = ref.timeseries[index];
            let paramsQuery = new URLSearchParams({
                measurement: timeseries.measurement,
                start: ref.start,
                end: ref.end,
                symbolic_name: timeseries.symbolicName,
                location: timeseries.location,
                device: timeseries.device,
                field: timeseries.field
            });
            endpoint = SHEPARD_BACKEND + '/timeseriesContainers/' + ref.timeseriesContainerId + '/payload?' + paramsQuery.toString();
            fetch(endpoint, {
                method: 'GET',
                headers: headers,
            })
            .then(response => response.json())
            .then(data => {
                let points = data.points;
                console.log(points);
                let xValues = [];
                let yValues = [];
                points.forEach(point => {
                    xValues.push(point.timestamp)
                    yValues.push(parseFloat(point.value));
                });
                console.log(xValues, yValues);
                xVals = xValues;
                yVals = yValues;
                plot_losses(null);
            })
            .catch(error => {
                console.error('Error fetching data:', error);
            });
        })
        .catch(error => {
            console.error('Error fetching data:', error);
        });
    })
    .catch(error => {
        console.error('Error fetching data:', error);
    });
}

function plot_losses(threshold) {
    if (CHART != null) {
        CHART.destroy();
    }
    dataSets = [{
        borderColor: "blue",
        data: yVals,
        fill: false,
        label: "validation loss"
    }]
    if (threshold != null) {
        dataSets.push({
            label: 'Threshold',
            borderColor: "red",
            data: new Array(xVals.length).fill(threshold),
            fill: false
        });
    }
    
    CHART = new Chart("lossChart", {
        type: "line",
        data: {
          labels: xVals,
          datasets: dataSets
        },
        options:{
            legend: {display: true}
            
        }
      });
    
}

function thresholdMethodSelected(thresholdMethod) {
    var thresholdElement = document.getElementById('DETECTION_THRESHOLD');
    if (thresholdMethod == 'max') {
        thresholdElement.value = Math.max(...yVals);
        thresholdElement.setAttribute('readonly', 'readonly');
        document.getElementById('stdsRow').style.display = 'none';
        document.getElementById('percentileRow').style.display = 'none';
        plot_losses(thresholdElement.value);
    } else if (thresholdMethod == 'percentile') {
        document.getElementById('percentileRow').style.removeProperty('display');
        document.getElementById('stdsRow').style.display = 'none';
    } else if (thresholdMethod == 'custom') {
        thresholdElement.removeAttribute('readonly');
        document.getElementById('percentileRow').style.display = 'none';
        document.getElementById('stdsRow').style.display = 'none';
    } else if (thresholdMethod == 'mean_stds') {
        document.getElementById('stdsRow').style.removeProperty('display');
        document.getElementById('percentileRow').style.display = 'none';
    } else {
        document.getElementById('percentileRow').style.display = 'none';
        document.getElementById('stdsRow').style.display = 'none';
    }
}

function calculateQuantile(q) {
    if (yVals && yVals.length > 0) {
        var a = yVals.slice();

        // Sort the array into ascending order
        data = a.sort();

        // Work out the position in the array of the percentile point
        var p = ((data.length) - 1) * q;
        var b = Math.ceil(p);

        let quantile = 0;
        // Check if index is an integer
        if (b == p && data[b+1] !== undefined){
            quantile =  0.5 * (parseFloat(data[b]) + parseFloat(data[b+1]));
        }else{
            quantile =  parseFloat(data[b]);
        }
        document.getElementById('DETECTION_THRESHOLD').value = quantile;
        plot_losses(quantile);
    } else {
        console.error('yVals is empty or undefined.');
    }
}

function caluclateMeanAndStds(stds) {
    if (yVals && yVals.length > 0) {
        let mean = yVals.reduce((a, b) => a + b, 0) / yVals.length;
        let stdsArray = yVals.map(x => Math.pow(x - mean, 2));
        let stdsValue = Math.sqrt(stdsArray.reduce((a, b) => a + b, 0) / stdsArray.length);
        document.getElementById('DETECTION_THRESHOLD').value = mean + stds * stdsValue;
        plot_losses(mean + stds * stdsValue);
    } else {
        console.error('yVals is empty or undefined.');
    }
}