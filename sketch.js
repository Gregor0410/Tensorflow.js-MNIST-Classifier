
let xTrain;
let yTrain;

function setup(){
    createCanvas(28,28);
    //Convert data from mnist.js to format for network to learn
    mnist = mnist.get();
    xTrain = mnist.slice(0,8000).map(x => x.input);
    yTrain = mnist.slice(0,8000).map(x => x.output);
    loadPixels();
    displayRandomDigit();
}

function draw(){
    //background(0);
    //rect(50,50,50,50);
}

function displayRandomDigit(){
    let digit = tf.tensor1d(random(xTrain));
    pixels =  digit.reshape([28,28]).array();
    updatePixels();
}