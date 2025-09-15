// Setup tooltips

$.noConflict();
jQuery( document ).ready(function () {
    jQuery('#modelName').tooltip({'trigger':'hover', 'title': 'Enter Name for shepard data object where training results will be stored', placement: 'right'});
    jQuery('#BATCH_SIZE').tooltip({'trigger':'hover', 'title': 'Mini Batch Size used to fit the model, e.g Using a BUFFER_SIZE = 32 and a BATCH_SIZE = 8 means the buffer will be fitted in chunks of 8 samples.', placement: 'right'});
    jQuery('#EPOCHS').tooltip({'trigger':'hover', 'title': 'Set the number of Epochs (iterations) for the training cycle of one data batch.', placement: 'right'});
    jQuery('#BUFFER_SIZE').tooltip({'trigger':'hover', 'title': 'Number of samples to be buffered and passed through the model at once, e.g. BUFFER SIZE = 1 means only one sample will be collected to fit the model. In case of a Recurrent Autoencoder the buffer size represents the number of sequences that are collected and trained at once. The length of this sequences can be set with the parameter SEQUENCE LENGTH', placement: 'right'});
    jQuery('#BUFFER_MODE').tooltip({'trigger':'hover', 'title': 'Sets the mechanism how the Buffer will be updated. Replace', placement: 'right'});
    jQuery('#SEQUENCE_LENGTH').tooltip({'trigger':'hover', 'title': 'Set the length of one time series to be used for the training in case of a Recurrent Autoencoder.', placement: 'right'});
});

window.onload = function() {
    readCollections();
}    