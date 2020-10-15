/* Function returns index of maximum value of Array */
function argMax(array) {
  return [].map.call(array, (x, i) => [x, i]).reduce((r, a) => (a[0] > r[0] ? a : r))[1];
}

/* Main function for handling canvas drawing and making predictions */
;(function() {

	const alphabet = {
		0: "A", 1: "B", 2: "C", 3: "D", 4: "E", 5: "F", 6: "G", 7: "H", 8: "I", 9: "J",
		10: "K", 11: "L", 12: "M", 13: "N", 14: "O", 15: "P", 16: "Q", 17: "R", 18: "S",
		19: "T", 20: "U", 21: "V", 22: "W", 23: "X", 24: "Y", 25: "Z"
	}

	// Cursor coordinates
	let coord = {x: 0, y: 0}
	// State of the drawing
	let paint = false;

	/* Global function, takes id of the canvas and does all the magic */
	function draw(id) {

		/* Canvases */
		const canvas = document.getElementById(id);
		const context = canvas.getContext('2d');
		const smallCanvas  = document.createElement('canvas');
		const smallContext = smallCanvas.getContext('2d');

		/* DOM */
		const go = document.getElementById('go');
		const clear = document.getElementById('clear');
		const result = document.getElementById('result');
		const images = document.getElementById('images');

		smallContext.canvas.width = 28;
		smallContext.canvas.height = 28;
		context.canvas.width  = canvas.offsetWidth;
		context.canvas.height = canvas.offsetHeight;
		context.lineWidth = 6;
		context.lineCap = 'round';
		context.strokeStyle = 'black';

		go.addEventListener('click', predict);
		clear.addEventListener('click', clear_canvas);

		/* Get cursor position */
		function getPosition(event) {
			coord.x = event.clientX - canvas.offsetLeft + window.pageXOffset;
			coord.y = event.clientY - canvas.offsetTop + window.pageYOffset;
		}

		/* Get mobile cursor position */
		function getMobilePosition(event) {
			coord.x = event.touches[0].clientX - canvas.offsetLeft + window.pageXOffset;
			coord.y = event.touches[0].clientY - canvas.offsetTop + window.pageYOffset;
		}

		/* Clears two canvases */
		function clear_canvas() {
			smallContext.clearRect(0,0,smallCanvas.width,smallCanvas.height);
			context.clearRect(0,0,canvas.width,canvas.height);
		}

		// Mobile start drawing
		canvas.addEventListener('touchstart', function(event){
			event.preventDefault();
			getMobilePosition(event);
		});

		// Mobile drawing
		canvas.addEventListener('touchmove', function(event){
			event.preventDefault();
			context.beginPath();
			context.moveTo(coord.x, coord.y);
			getMobilePosition(event);
			context.lineTo(coord.x , coord.y);
			context.stroke();
			context.closePath();
		});

		/* 1. Activate drawing state */
		canvas.addEventListener('mousedown', function(){
			getPosition(event);
			paint = true;
		});

		/* 2. Drawing in action */
		canvas.addEventListener('mousemove', function(){
			if(!paint) return;
			context.beginPath();
			context.moveTo(coord.x, coord.y);
			getPosition(event);
			context.lineTo(coord.x , coord.y);
			context.stroke();
			context.closePath();
		});

		/* 3. Stop drawing */
		canvas.addEventListener('mouseup', function() {
			paint = false;
		});

		/**
		Function takes array, transforms it to tensor [28,28,1] and
		makes a prediction with TensorflowJS CNN
		*/
		function predict(){

			// '?' -> '*'
			result.innerText = '*';
			// Temorary arrays for processing data
			let one_d = [];
			let two_d = [];

			const originalImage = new Image();
			originalImage.src = canvas.toDataURL();

			originalImage.onload = function() {

				smallContext.drawImage(originalImage, 0, 0, canvas.width, canvas.height, 0, 0, smallCanvas.width, smallCanvas.height);

				imageArray = smallContext.getImageData(0, 0, 28, 28).data;
				for (let i = 3; i <= imageArray.length; i+=4) { one_d.push(imageArray[i]); }
				for (let i = 0; i < one_d.length; i+=28) { two_d.push(one_d.slice(i, i+28)); }

				let tensor = tf.tensor([two_d]);
				let min = tensor.min();
				let max = tensor.max();

				let scaled_tensor = tensor.sub(min).div(max.sub(min));
				let pred = model.predict(scaled_tensor).dataSync();

				result.innerText = alphabet[argMax(pred)];

				let c = {};

				for(let i = 0; i < pred.length; i++) {
					c[alphabet[i]] = pred[i].toFixed(2);
				}
				console.log(c);
			}
		}
	}
	// Make draw first order function
	window.draw = draw;

})();
