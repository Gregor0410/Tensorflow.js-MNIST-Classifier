
let xTrain;
let yTrain;
let xTest;
let yTest;
let scaleFactor = 10;
let net;
let score;
function setup(){
    let canvas = createCanvas(28*scaleFactor,28*scaleFactor);
    canvas.parent('c');
    //Convert data from mnist.js to format for network to learn
    mnist = mnist.get();
    xTrain = mnist.slice(0,8000).map(x => x.input);
    yTrain = mnist.slice(0,8000).map(x => x.output);
    xTest = mnist.slice(8000,).map(x => x.input);
    yTest = mnist.slice(8000,).map(x => x.output);
    net = tf.sequential();
    net.add(tf.layers.dense({units:128,inputShape:[28*28],activation:'relu'}));
    net.add(tf.layers.dense({units:10, activation:'softmax'}));
    net.compile({optimizer:'adam',metrics:['accuracy','accuracy'],loss:'categoricalCrossentropy'});
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
function train(){
    let tXTrain=tf.tensor2d(xTrain);
    let tYTrain=tf.tensor2d(yTrain);
    history = net.fit(tXTrain,tYTrain,{epochs:1}).then(console.log(history));
}
function test(){
    let tXTest=tf.tensor2d(xTest);
    let tYTest=tf.tensor2d(yTest);
    score = net.evaluate(tXTest,tYTest);
    console.log(score.map(v=>v.arraySync()));
}