
let xTrain;
let yTrain;
let scaleFactor = 10
function setup(){
    createCanvas(28*scaleFactor,28*scaleFactor);
    //Convert data from mnist.js to format for network to learn
    mnist = mnist.get();
    xTrain = mnist.slice(0,8000).map(x => x.input);
    yTrain = mnist.slice(0,8000).map(x => x.output);
    background(0);
    displayRandomDigit();
}

function draw(){
    //background(0);
    //rect(50,50,50,50);
}

function displayRandomDigit(){
    let digit = tf.tensor(random(xTrain).map(v=>v*255),[1,28,28,1]);
    digit = tf.image.resizeNearestNeighbor(digit, [28*scaleFactor,28*scaleFactor]);
    digit = digit.flatten().arraySync();
    loadPixels()
    let i = 0; 
    for (let x = 0; x < 28*scaleFactor; x++) {
        for (let y = 0; y < 28*scaleFactor; y++) {
            set(y,x,digit[i]);
            i++;
        }
    }
    updatePixels();
}