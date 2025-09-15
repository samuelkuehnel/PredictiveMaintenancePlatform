var exceptionMapping = {};
var exceptionCount = 0;
jQuery( document ).ready(function () {
    document.getElementById("INVALID_DATA_HANDLING").addEventListener("change", function(event) {
        invalidDataHandlingSelected(event.target.value);
    });
    document.getElementById("INVALID_DATA_HANDLING_EXCEPTION").addEventListener("change", function(event) {
        invalidDataHandlingSelectedException(event.target.value);
    });
    document.getElementById("addRowButton").addEventListener("click", addRowWithButton);
});

function invalidDataHandlingSelected(value) {
    if (value === "remove") {
        document.getElementById("replaceInvalidData").style.display = "none";
    } else {
        document.getElementById("replaceInvalidData").style.removeProperty("display");
    }
}

function invalidDataHandlingSelectedException(value) {
    if (value === "remove") {
        document.getElementById("replaceInvalidDataException").style.display = "none";
    } else {
        document.getElementById("replaceInvalidDataException").style.removeProperty("display");
    }
}

function addRowWithButton() {
    var table = document.getElementById('variableBodyExceptions');
    var row = table.insertRow(-1);
    var data_handling = document.getElementById('INVALID_DATA_HANDLING_EXCEPTION').value;
    var replace_value = document.getElementById('REPLACE_VALUE_EXCEPTION').value;
    var variableName = document.getElementById('variableSelection').value;
    // var variableColumn = document.getElementById('variableSelection').innerHTML;
    // console.log(variableColumn);
    // Add three columns with sample data
    // Variable Name
    let cellVariable = row.insertCell(0);
    let inputElement = document.createElement('input');
    inputElement.type = 'text';
    inputElement.className = 'form-control';
    inputElement.name = 'exceptions_VariableName_' + exceptionCount;
    inputElement.value = exceptionMapping[variableName];
    if (inputElement.value == "undefined") {
        inputElement.value = variableName;
    }
    inputElement.hidden = true;
    cellVariable.appendChild(inputElement);
    inputElement = document.createElement('input');
    inputElement.type = 'text';
    inputElement.className = 'form-control';
    inputElement.value = variableName;
    cellVariable.appendChild(inputElement);


    
    //Add data Handling cell
    let cellDataHandling = row.insertCell(1);
    inputElement = document.createElement('input');
    inputElement.type = 'text';
    inputElement.className = 'form-control';
    inputElement.name = 'exceptions_handling_' + exceptionCount;
    inputElement.value = data_handling;
    cellDataHandling.appendChild(inputElement);

    //Add replace value cell
    let cellReplaceValue = row.insertCell(2);
    inputElement = document.createElement('input');
    inputElement.type = 'text';
    inputElement.className = 'form-control';
    inputElement.name = 'exceptions_replaceValue_' + exceptionCount;
    inputElement.value = replace_value;
    cellReplaceValue.appendChild(inputElement);

    // Add the fourth column with a delete button
    var deleteCell = row.insertCell(3);
    var deleteButton = document.createElement('button');
    deleteButton.innerHTML = 'Delete';
    deleteButton.className = 'btn btn-danger';
    deleteButton.type = 'button';
    deleteButton.onclick = function () {
        table.deleteRow(row.rowIndex-1);
    };
    deleteCell.appendChild(deleteButton);
    exceptionCount++;
    // Clear the input fields after adding the row
    document.getElementById('variableSelection').value = '';
    document.getElementById('INVALID_DATA_HANDLING_EXCEPTION').value = 'Choose Invalid Data Handling';
    document.getElementById('REPLACE_VALUE_EXCEPTION').value = '';
}