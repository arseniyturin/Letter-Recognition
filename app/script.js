/*

	Author: Arseny Turin
	Date: 20/10/2020

*/

// Simple ajax request implementation
function ajax(data, url, method="POST", content="application/x-www-form-urlencoded") {
	const xhttp = new XMLHttpRequest();
	xhttp.open(method, url, true);
	xhttp.setRequestHeader("Content-type", content);
	xhttp.send(data);
	xhttp.onreadystatechange = () => {
		if (this.readyState == 4 && this.status == 200) { console.log(this.responseText); }
	};
}

// Function returns index of the largest number in the array
function argMax(array) {
	return [].map.call(array, (x, i) => [x, i]).reduce((r, a) => (a[0] > r[0] ? a : r))[1];
}

// Main class that responsible for drawing on the canvas and predicting a letter with TensorflowJS
class letterRecognition {

	constructor(canvas) {

		// Alphabet is used to return result of the prediction as a letter
		this.alphabet = {
			0: "A", 1: "B", 2: "C", 3: "D", 4: "E", 5: "F", 6: "G", 7: "H", 8: "I", 9: "J",
			10: "K", 11: "L", 12: "M", 13: "N", 14: "O", 15: "P", 16: "Q", 17: "R", 18: "S",
			19: "T", 20: "U", 21: "V", 22: "W", 23: "X", 24: "Y", 25: "Z"
		}

		// Most of these variables describe context settings
		// and access DOM elements
		this.coord = {'x': 0, 'y': 0}
		this.draw = false;
		this.canvas = document.getElementById(canvas);
		this.context = this.canvas.getContext('2d');
		this.context.canvas.width  = this.canvas.offsetWidth;
		this.context.canvas.height = this.canvas.offsetHeight;
		this.context.lineWidth = 6;
		this.context.lineCap = 'round';
		this.context.strokeStyle = 'black';
		this.smallCanvas  = document.createElement('canvas');
		this.smallContext = this.smallCanvas.getContext('2d');
		this.smallContext.canvas.width = 28;
		this.smallContext.canvas.height = 28;
		this.go = document.getElementById('go');
		this.clear = document.getElementById('clear');
		this.result = document.getElementById('result');
		this.images = document.getElementById('images');

		// Assign functions to DOM events
		this.canvas.addEventListener('touchstart', this.touchstart.bind(this));
		this.canvas.addEventListener('touchmove', this.touchmove.bind(this));
		this.canvas.addEventListener('mousedown', this.mousedown.bind(this));
		this.canvas.addEventListener('mousemove', this.mousemove.bind(this));
		this.canvas.addEventListener('mouseup', this.mouseup.bind(this));
		this.go.addEventListener('click', this.predict.bind(this));
		this.clear.addEventListener('click', this.clearCanvas.bind(this));
	}

	// Clears canvas
	clearCanvas() {
		this.smallContext.clearRect(0,0,this.smallCanvas.width,this.smallCanvas.height);
		this.context.clearRect(0,0,this.canvas.width,this.canvas.height);
	}

	// Returns cursor position
	getPosition(event) {
		this.coord.x = event.clientX - this.canvas.offsetLeft + window.pageXOffset;
		this.coord.y = event.clientY - this.canvas.offsetTop + window.pageYOffset;
	}

	// Returns touch position
	getMobilePosition(event) {
		this.coord.x = event.touches[0].clientX - this.canvas.offsetLeft + window.pageXOffset;
		this.coord.y = event.touches[0].clientY - this.canvas.offsetTop + window.pageYOffset;
	}

	// Activate drawign with finger
	touchstart(event) {
		event.preventDefault();
		this.getMobilePosition(event);
	}

	// Drawing with finger
	touchmove(event) {
		event.preventDefault();
		this.context.beginPath();
		this.context.moveTo(this.coord.x, this.coord.y);
		this.getMobilePosition(event);
		this.context.lineTo(this.coord.x , this.coord.y);
		this.context.stroke();
		this.context.closePath();
	}

	// Activate drawing with mouse
	mousedown() {
		this.getPosition(event);
		this.draw = true;
	}

	// Drawing with mouse
	mousemove(event) {
		if(!this.draw) return;
		this.context.beginPath();
		this.context.moveTo(this.coord.x, this.coord.y);
		this.getPosition(event);
		this.context.lineTo(this.coord.x , this.coord.y);
		this.context.stroke();
		this.context.closePath();
	}

	// Stop drawing with mouse
	mouseup() {
		this.draw = false;
	}

	// Predicting letter
	predict() {

		/**
		1. Preprocessing. Canvas returns RGBA image of 112 x 112 px, which in fact is
		   1d array with (50176,) shape, because 112 * 112 * 4 = 50176. Our model input is a
			 tensor with shape (28, 28, 1). Hence first we must rescale and reshape
			 image to use it in the model. This achieved by using .drawImage() on hidden
			 smaller canvas' context and several array manipulations.
		2. Model predict.
		3. Return result back to the webpage along with predicted classes to the console
		   for analysis and debugging purpose.
		*/

		// '?' -> '*'
		this.result.innerText = '*';
		// Temorary arrays for processing data
		let one_d = [];
		let two_d = [];
		let canvasImg = new Image();

		canvasImg.src = this.canvas.toDataURL();
		canvasImg.addEventListener('load', transform.bind(this));

		// Because our model is trained on images 28x28
		// first we have to rescale drawing and transform it
		// into tensor with (28, 28, 1) shape
		function transform() {

			// This method scale image from 112x112 to 28x28
			// Because we can't use .drawImage directly on Canvas
			// we create Image and use it as an input.
			this.smallContext.drawImage(canvasImg, 0, 0, this.canvas.width, this.canvas.height, 0, 0, this.smallCanvas.width, this.smallCanvas.height);

			// Retrieve small image in form of array. Initial shape is (3136,)
			let smallCanvasData = this.smallContext.getImageData(0, 0, 28, 28).data;

			// Pulling every 4th element from initial array.
			for (let i = 3; i <= smallCanvasData.length; i+=4) { one_d.push(smallCanvasData[i]); }

			// Transforming (784,) => (28,28)
			for (let i = 0; i < one_d.length; i+=28) { two_d.push(one_d.slice(i, i+28)); }

			// Transforming JS array to Tensor
			let tensor = tf.tensor([two_d]);

			// Min-Max scaling
			let min = tensor.min();
			let max = tensor.max();
			let scaled_tensor = tensor.sub(min).div(max.sub(min));

			// Make predictions
			let pred = model.predict(scaled_tensor).dataSync();

			// Show prediction in the <div id="result">
			this.result.innerText = this.alphabet[argMax(pred)];

			// Predict classes
			let classes = {};
			for(let i = 0; i < pred.length; i++) {
				classes[this.alphabet[i]] = pred[i].toFixed(2);
			}
			console.log(classes);
		}
	}
}
