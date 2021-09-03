// After deploying the model, A few steps are required to perform on-device inference.
// 1.) Copy the below code and paste it on the right hand side pane of the Espruino web-based IDE.
// 2.) Upload the code to bangle.js by pressing the deploy button. 
// 3.) Now, use the REPL window of the ide to start the "accel" event listener and pass recStart function as an argument to start the prediction process. 
//     To start the event listener, use the command, Bangle.on("accel", recStart);

// Requiring necessary modules.
var storage = require("Storage");
var tf = require("tensorflow");

//Reading the model and the activity names.
var model = storage.read(".tfmodel");
var modelnames = storage.read(".tfnames").split(",");

//converting the model to a flatstring as the tf.create method accepts
//model as either a flatstring or flatarray.
model = E.toArrayBuffer(model);
model = E.toString(model);

//Creating the interpreter to perform inference.
var tflite_model = tf.create(2352, model);

//The input to the interpreter is given as a 1D array,
var samples= 150;
var acceleration_values = [];
var accelIdx = 0;



function indexOfMaximum(prev, curr) {
  return prev > curr ? prev : curr;
}

function recordAcceleration(a) {
  acceleration_values.push((a.x));
  acceleration_values.push((a.y));
  acceleration_values.push((a.z));
  accelIdx = accelIdx + 3;
  if(accelIdx >= samples) {
    prediction();
  }
}

function recStart() {
  Bangle.removeListener('accel', recordAcceleration);
  Bangle.accelWr(0x18,0b01110100); // stand-by mode, +-8g.
  Bangle.accelWr(0x1B,0b01000010); // 50Hz output, ODR/2 filter.
  Bangle.accelWr(0x18,0b11110100); // operating mode, +-8g.
  Bangle.setPollInterval(20);  // 20ms poll interval
  Bangle.on('accel', recordAcceleration);
}

function recStop() {
  Bangle.setPollInterval(80); // default poll interval
  Bangle.accelWr(0x18,0b01101100); // off, +-4g
  Bangle.accelWr(0x1B,0x0); // default 12.5hz output
  Bangle.accelWr(0x18,0b11101100); // +-4g
}

function prediction() {
  Bangle.removeListener('accel', recordAcceleration);

  // Setting input for the tflite Model.
  tflite_model.getInput().set(acceleration_values);
  tflite_model.invoke();

  // Getting Output from the Model.
  let output = tflite_model.getOutput();

  // Output Classification.
  maximumNum = output.reduce(indexOfMaximum, undefined);
  let idx = output.indexOf(maximumNum);
  let activity = modelnames[idx];
  console.log(activity);

  // Displaying the activity information on the Watch.
  E.showMessage(activity);
  setTimeout(function() {
    g.clear();
  }, 1000);
  acceleration_values = [];
  accelIdx = 0;
  Bangle.on("accel", recordAcceleration);
}

// detects the press on BTN1 and stops the prediction process.
setWatch(function() {
  Bangle.removeAllListeners();
  recStop();
  acceleration_values = [];
  accelIdx = 0;
  g.clear();
}, BTN1, {edge:"rising", debounce:50, repeat:true});
