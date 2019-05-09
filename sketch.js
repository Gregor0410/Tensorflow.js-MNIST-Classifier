
let xTrain;
let yTrain;
let xTest;
let yTest;
let scaleFactor = 10;
let net;
let mnist;
let score;
let outputBox;
function setup(){
    let canvas = createCanvas(28*scaleFactor,28*scaleFactor);
    canvas.parent('c');
    outputBox = document.getElementById('output')
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

function displayRandomDigit(){
    let input = random(xTest);
    let digit = tf.tensor(input.map(v=>v*255),[1,28,28,1]);
    let prediction = net.predict(tf.tensor(input).expandDims(0)).argMax(1).arraySync();
    outputBox.innerHTML = "The network predicts: " + prediction.join();
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
async function train(){
    let tXTrain=tf.tensor2d(xTrain);
    let tYTrain=tf.tensor2d(yTrain);
    history = await net.fit(tXTrain,tYTrain,{epochs:1}).then(
    (console.log(history)));
}
async function test(){
    let tXTest=tf.tensor2d(xTest);
    let tYTest=tf.tensor2d(yTest);
    score = await net.evaluate(tXTest,tYTest).then(
    console.log(score.map(v=>v.arraySync())));
}