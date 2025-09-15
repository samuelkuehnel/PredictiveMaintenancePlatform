var exceptionMapping = {};
function readCSVFile(target) {
    const file = target.files[0];
    if (file) {
        const reader = new FileReader();
        reader.onload = function (e) {
            const text = e.target.result;
            let header_present = document.getElementById('headerCheckbox').checked;
            console.log(header_present);
            const rows = text.split('\n').slice(0, 6); // Get first 6 lines
            let table = document.createElement('table');
            let headerRow = table.createTHead().insertRow(0);
            let headers = rows[0].split(',');
            document.getElementById('headers').value = ''; // Clear previous headers
            if (header_present) {
                let i = 0;
                headers.forEach(header => {
                    // html += `<th>${header}</th>`;
                    let cell = document.createElement('th');
                    cell.innerHTML = header; // Set the header text
                    headerRow.appendChild(cell); // Append the cell to the header row
                    document.getElementById('headers').value += header + ',,,'; // Store headers in an input
                    exceptionMapping[header] = `column${i}`;
                    i++;
                });
                rows.shift(); // Remove the header row from the data
            } else {
                for (let i = 0; i < headers.length; i++) {
                    let cell = document.createElement('th');
                    cell.innerHTML = `column${i + 1}`; // Set the header text
                    headerRow.appendChild(cell); // Append the cell to the header row
                    document.getElementById('headers').value += 'column' + i + ',,,'; // Store headers in an input
                    exceptionMapping[`column${i + 1}`] = `column${i}`;
                }
                rows.pop(); // Remove the last row if no header is present
            }
            fillDatalist(); // Call the function to prefill variables
            // html += '</tr>';
            const columnCount = headers.length; // Get the number of columns
            document.getElementById('csv_content').innerHTML = ''; // Clear previous content
            document.getElementById('columnCount').value = columnCount; // Store the column count in an input
            let body = table.createTBody();
            rows.forEach(row => {
                let tableRow = body.insertRow(-1);
                // html += '<tr>';
                const cells = row.split(',');
                cells.forEach(cell => {
                    let tableCell = tableRow.insertCell(-1);
                    tableCell.innerHTML = cell; // Set the cell text
                });
                // html += '</tr>';
            });
            // html += '</table>';
            document.getElementById('csv_content').appendChild(table); // Append the table to the div
            document.getElementById('csvTextContent').value = text; // Store the entire CSV content in an input
        };
        reader.readAsText(file);
    }
};

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
    for (var i = 0; i < headers.length - 1; i++) {
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